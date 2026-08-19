from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from tqdm.auto import tqdm

from baselines import compute_average_image, fit_sigmas_from_average_image
from cnn_site import (
    build_site_crop_cache,
    compute_cnn_preprocessing,
    make_random_split,
    make_site_loader,
    set_random_seed,
)
from mf_site import best_threshold_by_fidelity, classification_fidelity
from utils import get_device


default_data_dir = Path("dataset_3x3")
default_model_dir = Path("models_3x3")
default_output_dir = Path("output_3x3")

lightweight_cnn_site_method = "lightweight_cnn_site"
lightweight_cnn_site_architecture = "tiny_cnn_site_3conv_pool_8fc"
lightweight_cnn_site_neighbor_method = "lightweight_cnn_site_neighbor"
lightweight_cnn_site_neighbor_architecture = "tiny_cnn_site_3conv_pool_neighbor8_8fc"
lightweight_cnn_site_neighbor_weighted_method = "lightweight_cnn_site_neighbor_weighted"
lightweight_cnn_site_neighbor_weighted_architecture = (
    "tiny_cnn_site_3conv_pool_psf_neighbor8_8fc"
)
# Late-injection variants: the neighbour features enter the *second* dense
# layer instead of the first. Su et al. inject exposure time this way, and the
# spacing-conditioned model here does too, so this is the comparison that says
# whether injection point matters for neighbours.
lightweight_cnn_site_neighbor_late_method = "lightweight_cnn_site_neighbor_late"
lightweight_cnn_site_neighbor_late_architecture = (
    "tiny_cnn_site_3conv_pool_8fc_neighbor8"
)
lightweight_cnn_site_neighbor_weighted_late_method = (
    "lightweight_cnn_site_neighbor_weighted_late"
)
lightweight_cnn_site_neighbor_weighted_late_architecture = (
    "tiny_cnn_site_3conv_pool_8fc_psf_neighbor8"
)

# Where the neighbour features join the classifier.
#   "early"  concatenated onto the flattened conv features -> Linear(flat+8, 8)
#   "late"   concatenated onto the hidden activations      -> Linear(8+8, 2)
neighbor_injection_early = "early"
neighbor_injection_late = "late"

selected_architectures = (
    # lightweight_cnn_site_architecture,
    lightweight_cnn_site_neighbor_architecture,
    # lightweight_cnn_site_neighbor_weighted_architecture,
)

crop_size = 10
learning_rate = 1e-4
epochs = 300
early_stopping_patience = 20
early_stopping_min_delta = 1e-5
batch_size = 64
eval_batch_size = 2048
num_workers = 10
base_seed = 42
device = "cuda:1"
use_site_crop_cache = True
site_crop_cache_batch_size = 512

neighbor_feature_count = 8
neighbor_mode_none = "none"
neighbor_mode_means = "neighbor_means"
neighbor_mode_psf_weighted = "psf_weighted_neighbor_means"
# Only used when the PSF width cannot be fitted from the training images.
default_neighbor_sigma_px = 2.0

tiny_conv_channels = (8, 16, 32)
tiny_hidden_units = 8

architecture_configs = {
    lightweight_cnn_site_architecture: {
        "method": lightweight_cnn_site_method,
        "conv_channels": tiny_conv_channels,
        "hidden_units": tiny_hidden_units,
        "neighbor_features": 0,
        "neighbor_feature_mode": neighbor_mode_none,
        "description": "Conv8 -> pool -> Conv16 -> pool -> Conv32 -> FC8 -> FC2",
    },
    lightweight_cnn_site_neighbor_architecture: {
        "method": lightweight_cnn_site_neighbor_method,
        "conv_channels": tiny_conv_channels,
        "hidden_units": tiny_hidden_units,
        "neighbor_features": neighbor_feature_count,
        "neighbor_feature_mode": neighbor_mode_means,
        "description": (
            "Conv8 -> pool -> Conv16 -> pool -> Conv32 -> "
            "8 neighbor means -> FC8 -> FC2"
        ),
    },
    lightweight_cnn_site_neighbor_weighted_architecture: {
        "method": lightweight_cnn_site_neighbor_weighted_method,
        "conv_channels": tiny_conv_channels,
        "hidden_units": tiny_hidden_units,
        "neighbor_features": neighbor_feature_count,
        "neighbor_feature_mode": neighbor_mode_psf_weighted,
        "description": (
            "Conv8 -> pool -> Conv16 -> pool -> Conv32 -> "
            "8 PSF-weighted neighbor means -> FC8 -> FC2"
        ),
    },
    lightweight_cnn_site_neighbor_late_architecture: {
        "method": lightweight_cnn_site_neighbor_late_method,
        "conv_channels": tiny_conv_channels,
        "hidden_units": tiny_hidden_units,
        "neighbor_features": neighbor_feature_count,
        "neighbor_feature_mode": neighbor_mode_means,
        "neighbor_injection": neighbor_injection_late,
        "description": (
            "Conv8 -> pool -> Conv16 -> pool -> Conv32 -> FC8 -> "
            "8 neighbor means -> FC2"
        ),
    },
    lightweight_cnn_site_neighbor_weighted_late_architecture: {
        "method": lightweight_cnn_site_neighbor_weighted_late_method,
        "conv_channels": tiny_conv_channels,
        "hidden_units": tiny_hidden_units,
        "neighbor_features": neighbor_feature_count,
        "neighbor_feature_mode": neighbor_mode_psf_weighted,
        "neighbor_injection": neighbor_injection_late,
        "description": (
            "Conv8 -> pool -> Conv16 -> pool -> Conv32 -> FC8 -> "
            "8 PSF-weighted neighbor means -> FC2"
        ),
    },
}


class LightweightCnnSiteNet(nn.Module):
    def __init__(
        self,
        crop_size,
        architecture=lightweight_cnn_site_architecture,
        neighbor_sigma_px=None,
        neighbor_feature_scale=None,
    ):
        super().__init__()
        crop_size = int(crop_size)
        architecture = str(architecture)
        if architecture not in architecture_configs:
            raise KeyError(f"unknown lightweight CNN-site architecture: {architecture}")

        config = architecture_configs[architecture]
        conv1_channels, conv2_channels, conv3_channels = config["conv_channels"]
        pooled_side = crop_size // 4
        if pooled_side <= 0:
            raise ValueError(f"crop_size must be at least 4, got {crop_size}")

        self.architecture = architecture
        self.crop_size = crop_size
        self.neighbor_feature_count = int(config.get("neighbor_features", 0))
        self.neighbor_feature_mode = str(
            config.get("neighbor_feature_mode", neighbor_mode_none)
        )

        self.features = nn.Sequential(
            nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(conv2_channels, conv3_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.neighbor_injection = str(
            config.get("neighbor_injection", neighbor_injection_early)
        )
        conv_features = conv3_channels * pooled_side * pooled_side
        hidden_units = int(config["hidden_units"])

        # Early: the neighbours widen the first dense layer, so they can interact
        # nonlinearly with the conv features but share the 8 hidden units with
        # 128 of them. Late: they widen the second, giving a direct undiluted
        # path to the logits at the cost of that interaction.
        if self.neighbor_injection == neighbor_injection_late:
            first_in, second_in = conv_features, hidden_units + self.neighbor_feature_count
        else:
            first_in, second_in = (
                conv_features + self.neighbor_feature_count,
                hidden_units,
            )

        self.classifier = nn.Sequential(
            nn.Linear(first_in, hidden_units),
            nn.ReLU(),
            nn.Linear(second_in, 2),
        )

        if self.neighbor_feature_mode == neighbor_mode_psf_weighted:
            if neighbor_sigma_px is None:
                neighbor_sigma_px = default_neighbor_sigma_px
            self.register_buffer(
                "neighbor_sigma_px",
                torch.tensor(float(neighbor_sigma_px)),
            )

        if self.neighbor_feature_count:
            # Divisor that brings the neighbour features to roughly +-1 so they
            # are commensurate with the conv features they are concatenated
            # with.  1.0 means "not yet calibrated"; fit_neighbor_feature_scale
            # sets the real value before training starts.
            self.register_buffer(
                "neighbor_feature_scale",
                torch.tensor(float(neighbor_feature_scale or 1.0)),
            )

    def forward(self, x, neighbor_features=None):
        features = self.features(x)

        if not self.neighbor_feature_count:
            return self.classifier(features)

        if neighbor_features is None:
            raise ValueError(
                f"{self.architecture} requires "
                f"{self.neighbor_feature_count} neighbor features"
            )
        if neighbor_features.shape[-1] != self.neighbor_feature_count:
            raise ValueError(
                f"expected {self.neighbor_feature_count} neighbor features, "
                f"got {neighbor_features.shape[-1]}"
            )
        neighbor_features = neighbor_features.to(dtype=features.dtype)

        if self.neighbor_injection == neighbor_injection_late:
            # Indexing the Sequential rather than splitting it into two modules
            # keeps the state_dict keys classifier.0/1/2 identical for both
            # injection points, so checkpoints stay interchangeable.
            hidden = self.classifier[1](self.classifier[0](features))
            return self.classifier[2](torch.cat([hidden, neighbor_features], dim=1))

        return self.classifier(torch.cat([features, neighbor_features], dim=1))


def lightweight_method_for_architecture(architecture):
    return architecture_configs[str(architecture)]["method"]


def architecture_requires_site_coordinates(architecture):
    config = architecture_configs[str(architecture)]
    return int(config.get("neighbor_features", 0)) > 0

def architecture_uses_psf_weighting(architecture):
    config = architecture_configs[str(architecture)]
    return str(config.get("neighbor_feature_mode")) == neighbor_mode_psf_weighted

def output_model_path(model_dir, h5_path, architecture):
    method_name = lightweight_method_for_architecture(architecture)
    return model_dir / f"{Path(h5_path).stem}_{method_name}.joblib"

def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())

def neighbor_indices_and_distances(sites, n_sites, device, dtype):
    """Indices of, and distances to, the 8 nearest other sites.

    Accepts either one shared (n_sites, 2) layout or a per-image
    (batch, n_sites, 2) one, and always returns batched
    (batch, n_sites, neighbor_feature_count) tensors.

    Arrays smaller than 9 sites are padded with an infinite distance, which
    drives the corresponding weight to zero for every weighting mode.
    """
    sites = torch.as_tensor(sites, device=device, dtype=dtype)
    if sites.ndim == 2:
        sites = sites[None, :, :]
    if sites.ndim != 3:
        raise ValueError(
            "sites must have shape (n_sites, 2) or (batch, n_sites, 2), "
            f"got {tuple(sites.shape)}"
        )
    if sites.shape[1] != n_sites:
        raise ValueError(f"expected {n_sites} site coordinates, got {sites.shape[1]}")

    distances = torch.cdist(sites, sites)
    eye = torch.eye(n_sites, device=device, dtype=torch.bool)
    distances = distances.masked_fill(eye[None, :, :], float("inf"))

    k_neighbors = min(neighbor_feature_count, max(n_sites - 1, 0))
    nearest = torch.topk(distances, k=k_neighbors, largest=False, dim=2)
    indices, values = nearest.indices, nearest.values

    if k_neighbors < neighbor_feature_count:
        padding = neighbor_feature_count - k_neighbors
        indices = functional.pad(indices, (0, padding))
        values = functional.pad(values, (0, padding), value=float("inf"))

    return indices, values


def neighbor_weights(distances, mode, sigma_px=None):
    if mode == neighbor_mode_means:
        return torch.isfinite(distances).to(dtype=distances.dtype)

    if mode == neighbor_mode_psf_weighted:
        if sigma_px is None:
            raise ValueError(
                f"{neighbor_mode_psf_weighted} requires a fitted PSF sigma"
            )
        sigma_px = float(sigma_px)
        if not np.isfinite(sigma_px) or sigma_px <= 0.0:
            raise ValueError(f"PSF sigma must be positive and finite, got {sigma_px}")
        squared = distances ** 2
        nearest_squared = torch.nan_to_num(squared[..., :1], posinf=0.0)
        return torch.exp(-(squared - nearest_squared) / (2.0 * sigma_px ** 2))

    raise ValueError(f"unknown neighbor feature mode: {mode}")


def neighbor_features_from_crops(
    crops,
    mode=neighbor_mode_means,
    sites=None,
    sigma_px=None,
    feature_scale=None,
):
    """Neighbour crop means, weighted and normalised.

    `feature_scale` divides the finished features so they reach roughly +-1.
    Unnormalised they span about +-0.02 -- means of mean-subtracted crops --
    while the conv features they are concatenated with reach 4.5.  Two
    consequences follow from that 230x gap, and normalising fixes both:

      * the concatenated tensor carries one fixed-point type, so a step large
        enough for the conv half is larger than the entire neighbour range and
        the neighbour half quantizes to zero;
      * the optimizer has to learn very large weights on those eight columns to
        get any use out of them.  It only reached 1.8x the conv weights when it
        needed ~52x, so the neighbour path contributes ~0.2% of the logit --
        which is a scale handicap, not evidence the information is useless.

    Scaling is symmetric (divide by max|v|, zero stays at zero) rather than an
    affine map to [0,1]: the features genuinely go negative, and an offset
    could not be folded into a weight rescale.
    """
    if sites is None:
        raise ValueError("neighbor features require site coordinates")

    batch, n_sites = crops.shape[:2]
    site_values = crops.mean(dim=(2, 3, 4))

    indices, distances = neighbor_indices_and_distances(
        sites=sites,
        n_sites=n_sites,
        device=site_values.device,
        dtype=site_values.dtype,
    )
    if indices.shape[0] == 1 and batch > 1:
        indices = indices.expand(batch, -1, -1)
        distances = distances.expand(batch, -1, -1)

    features = torch.gather(
        site_values,
        dim=1,
        index=indices.reshape(batch, -1),
    ).reshape(batch, n_sites, neighbor_feature_count)

    features = features * neighbor_weights(distances, mode=mode, sigma_px=sigma_px)

    if feature_scale is not None:
        feature_scale = float(feature_scale)
        if np.isfinite(feature_scale) and feature_scale > 0.0:
            features = features / feature_scale

    return features


def copy_neighbor_config(target, source):
    target.neighbor_feature_count = int(getattr(source, "neighbor_feature_count", 0))
    target.neighbor_feature_mode = str(
        getattr(source, "neighbor_feature_mode", neighbor_mode_none)
    )
    sigma_px = getattr(source, "neighbor_sigma_px", None)
    if sigma_px is not None:
        target.neighbor_sigma_px = float(sigma_px)
    feature_scale = getattr(source, "neighbor_feature_scale", None)
    if feature_scale is not None:
        target.neighbor_feature_scale = float(feature_scale)
    target.neighbor_injection = str(
        getattr(source, "neighbor_injection", neighbor_injection_early)
    )
    return target


def neighbor_features_for_model(crops, model, sites):
    mode = getattr(model, "neighbor_feature_mode", neighbor_mode_means)
    sigma_px = getattr(model, "neighbor_sigma_px", None)

    if mode == neighbor_mode_psf_weighted and sigma_px is None:
        # Name the offending class: this is almost always a wrapper model that
        # re-exposed the mode but not the width.
        raise ValueError(
            f"{type(model).__name__} declares neighbor_feature_mode={mode!r} "
            "but carries no neighbor_sigma_px.\n"
            "  Trained models register it as a buffer, so it is restored by "
            "load_state_dict and also stored in the checkpoint under "
            "'neighbor_sigma_px'.\n"
            "  Wrapper models must copy it across with copy_neighbor_config("
            "wrapper, trained_model)."
        )

    return neighbor_features_from_crops(
        crops,
        mode=mode,
        sites=sites,
        sigma_px=sigma_px,
        feature_scale=getattr(model, "neighbor_feature_scale", None),
    )


@torch.no_grad()
def fit_neighbor_feature_scale(model, dataloader, device, max_batches=16):
    """Largest unnormalised neighbour feature over the training data.

    Measured with the model's own scale reset to 1, so the pass sees raw
    weighted means.  Dividing by this brings the features to +-1, matching the
    magnitude of the conv features they are concatenated with.
    """
    if not int(getattr(model, "neighbor_feature_count", 0)):
        return 1.0

    previous = float(getattr(model, "neighbor_feature_scale", 1.0))
    model.neighbor_feature_scale = torch.tensor(
        1.0, device=next(model.parameters()).device
    )

    largest = 0.0
    try:
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= max_batches:
                break
            crops, labels, sites = unpack_site_batch(batch)
            if sites is None:
                raise ValueError("neighbor models require site coordinates")
            features = neighbor_features_for_model(
                crops.to(device), model, sites.to(device)
            )
            if features.numel():
                largest = max(largest, float(features.abs().max()))
    finally:
        model.neighbor_feature_scale = torch.tensor(
            previous, device=next(model.parameters()).device
        )

    if not np.isfinite(largest) or largest <= 0.0:
        return 1.0
    return largest


def fit_neighbor_sigma_px(h5_path, train_idx, num_workers=4):
    avg_image, sites = compute_average_image(
        h5_path=h5_path,
        indices=train_idx,
        num_workers=num_workers,
    )
    sigmas = fit_sigmas_from_average_image(
        avg_image=avg_image,
        sites=sites,
        fit_size=9,
    )
    return float(np.median(sigmas))

def unpack_site_batch(batch):
    if len(batch) == 2:
        crops, labels = batch
        return crops, labels, None
    if len(batch) == 3:
        crops, labels, sites = batch
        return crops, labels, sites
    raise ValueError(f"expected batch with 2 or 3 items, got {len(batch)}")

def flatten_site_batch(crops, labels, model, sites=None):
    batch, n_sites = labels.shape
    neighbor_features = None

    if int(getattr(model, "neighbor_feature_count", 0)):
        neighbor_features = neighbor_features_for_model(crops, model, sites)
        neighbor_features = neighbor_features.reshape(batch * n_sites, -1)

    crops = crops.reshape(batch * n_sites, 1, crops.shape[-2], crops.shape[-1])
    labels = labels.reshape(batch * n_sites)
    return crops, labels, neighbor_features

def prepare_site_batch(batch, model, device):
    """Unpack, move to device and flatten one site batch."""
    crops, labels, sites = unpack_site_batch(batch)
    crops = crops.to(device)
    labels = labels.to(device)
    image_count, site_count = labels.shape

    crops, flat_labels, neighbor_features = flatten_site_batch(
        crops,
        labels,
        model,
        sites=None if sites is None else sites.to(device),
    )
    return crops, flat_labels, neighbor_features, image_count, site_count

def model_logits(model, crops, neighbor_features=None):
    if int(getattr(model, "neighbor_feature_count", 0)):
        return model(crops, neighbor_features)
    return model(crops)

def load_lightweight_cnn_site_model(best, device="cpu"):
    architecture = str(best.get("architecture", lightweight_cnn_site_architecture))
    if architecture not in architecture_configs:
        raise KeyError(
            f"checkpoint architecture {architecture!r} is not available; "
            "the bare-Linear (fc2) variants were removed"
        )
    if "crop_size" not in best:
        raise KeyError(
            "checkpoint does not record crop_size; refusing to guess "
            "(simulated models use 10 px crops, real-data models use 20 px)"
        )
    model = LightweightCnnSiteNet(
        crop_size=int(best["crop_size"]),
        architecture=architecture,
        neighbor_sigma_px=best.get("neighbor_sigma_px"),
        neighbor_feature_scale=best.get("neighbor_feature_scale"),
    ).to(device)
    model.load_state_dict(best["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def calc_lightweight_cnn_site_scores_from_site_loader(dataloader, model, device="cpu"):
    model.eval()
    all_scores = []
    all_labels = []

    for batch in dataloader:
        crops, labels, neighbor_features, image_count, site_count = (
            prepare_site_batch(batch, model, device)
        )

        probabilities = torch.softmax(
            model_logits(model, crops, neighbor_features),
            dim=1,
        )[:, 1]

        all_scores.append(probabilities.reshape(image_count, site_count).cpu())
        all_labels.append(labels.reshape(image_count, site_count).float().cpu())

    return torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


@torch.no_grad()
def calc_lightweight_cnn_site_scores(dataloader, best, device="cpu"):
    """Score a site loader from a saved checkpoint."""
    model = load_lightweight_cnn_site_model(best, device=device)
    return calc_lightweight_cnn_site_scores_from_site_loader(
        dataloader=dataloader,
        model=model,
        device=device,
    )


@torch.no_grad()
def evaluate_lightweight_loss_accuracy(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    for batch in dataloader:
        crops, labels, neighbor_features, _image_count, _site_count = (
            prepare_site_batch(batch, model, device)
        )
        logits = model_logits(model, crops, neighbor_features)
        loss = loss_fn(logits, labels)

        examples = int(labels.numel())
        total_loss += float(loss.item()) * examples
        total_examples += examples
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def tune_thresholds(scores, labels):
    """Per-site decision thresholds chosen to maximise fidelity."""
    n_sites = labels.shape[1]
    thresholds = np.zeros(n_sites, dtype=np.float32)
    fidelities = np.zeros(n_sites, dtype=np.float32)

    for site_index in range(n_sites):
        threshold, fidelity = best_threshold_by_fidelity(
            scores[:, site_index],
            labels[:, site_index],
        )
        thresholds[site_index] = threshold
        fidelities[site_index] = fidelity

    return thresholds, fidelities


def evaluate_lightweight_cnn_site_scores_labels(scores, labels, best):
    thresholds = np.asarray(best["threshold"], dtype=np.float32)
    if thresholds.ndim == 0:
        thresholds = np.full(labels.shape[1], float(thresholds), dtype=np.float32)

    fidelities = np.zeros(labels.shape[1], dtype=np.float32)
    for site_index in range(labels.shape[1]):
        fidelities[site_index] = classification_fidelity(
            scores[:, site_index],
            labels[:, site_index],
            thresholds[site_index],
        )

    predictions = scores >= thresholds[None, :]
    errors = predictions != (labels > 0.5)

    return {
        "scores": scores,
        "labels": labels,
        "fidelity": fidelities,
        "mean_fidelity": float(np.mean(fidelities)),
        "errors_per_site": errors.sum(axis=0).astype(np.int64),
        "total_errors": int(errors.sum()),
        "raw_accuracy": float(1.0 - errors.mean()),
    }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def train_lightweight_cnn_site(
    h5_path,
    train_idx,
    val_idx,
    architecture=lightweight_cnn_site_architecture,
    crop_size=crop_size,
    learning_rate=learning_rate,
    epochs=epochs,
    batch_size=batch_size,
    num_workers=num_workers,
    device="cpu",
    mean_image=None,
    scale=None,
    early_stopping_patience=early_stopping_patience,
    early_stopping_min_delta=early_stopping_min_delta,
    crop_cache_path=None,
    neighbor_sigma_px=None,
):
    architecture = str(architecture)
    config = architecture_configs[architecture]
    return_sites = architecture_requires_site_coordinates(architecture)

    loader_options = {
        "h5_path": h5_path,
        "crop_size": crop_size,
        "mean_image": mean_image,
        "scale": scale,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": str(device).startswith("cuda"),
        "crop_cache_path": crop_cache_path,
        "return_sites": return_sites,
    }
    train_loader = make_site_loader(indices=train_idx, shuffle=True, **loader_options)
    val_loader = make_site_loader(indices=val_idx, shuffle=False, **loader_options)

    model = LightweightCnnSiteNet(
        crop_size=crop_size,
        architecture=architecture,
        neighbor_sigma_px=neighbor_sigma_px,
    ).to(device)

    # Fitted before the first epoch so the network trains on features that are
    # already commensurate with the conv features, rather than learning to
    # compensate for a 230x scale gap it cannot fully close.
    neighbor_feature_scale = 1.0
    if architecture_requires_site_coordinates(architecture):
        neighbor_feature_scale = fit_neighbor_feature_scale(
            model, train_loader, device
        )
        model.neighbor_feature_scale = torch.tensor(
            float(neighbor_feature_scale), device=device
        )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    best_state = None
    best_val_loss = np.inf
    best_epoch = -1
    no_improvement_epochs = 0
    stopped_early = False

    progress = tqdm(
        range(epochs),
        desc=f"{Path(h5_path).stem} {architecture}",
        unit="epoch",
        leave=False,
    )
    for epoch in progress:
        model.train()
        train_loss_sum = 0.0
        train_examples = 0
        train_correct = 0

        for batch in train_loader:
            crops, labels, neighbor_features, _image_count, _site_count = (
                prepare_site_batch(batch, model, device)
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model_logits(model, crops, neighbor_features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            examples = int(labels.numel())
            train_loss_sum += float(loss.item()) * examples
            train_examples += examples
            train_correct += int((logits.argmax(dim=1) == labels).sum().item())

        train_loss = train_loss_sum / max(train_examples, 1)
        train_accuracy = train_correct / max(train_examples, 1)
        val_metrics = evaluate_lightweight_loss_accuracy(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
            }
        )

        if val_metrics["loss"] < best_val_loss - early_stopping_min_delta:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch + 1
            no_improvement_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            no_improvement_epochs += 1

        progress.set_postfix(
            train_loss=f"{train_loss:.4g}",
            val_loss=f"{val_metrics['loss']:.4g}",
            val_acc=f"{val_metrics['accuracy']:.4f}",
            best=best_epoch,
        )

        if (
            early_stopping_patience is not None
            and early_stopping_patience > 0
            and no_improvement_epochs >= early_stopping_patience
        ):
            stopped_early = True
            break

    if best_state is None:
        raise RuntimeError("tiny CNN-site training did not produce a best checkpoint")

    model.load_state_dict(best_state)
    val_scores, val_labels = calc_lightweight_cnn_site_scores_from_site_loader(
        dataloader=val_loader,
        model=model,
        device=device,
    )
    # Tuned on validation, never on test.  Leaving these at the 0.5 default is
    # worth up to 3 pp of fidelity on the neighbor models.
    thresholds, fidelities = tune_thresholds(val_scores, val_labels)

    return {
        "method": config["method"],
        "architecture": architecture,
        "architecture_description": config["description"],
        "neighbor_features": int(config.get("neighbor_features", 0)),
        "neighbor_feature_mode": str(config.get("neighbor_feature_mode")),
        "neighbor_sigma_px": (
            None if neighbor_sigma_px is None else float(neighbor_sigma_px)
        ),
        "neighbor_feature_scale": float(neighbor_feature_scale),
        "state_dict": best_state,
        "crop_size": int(crop_size),
        "parameters": int(count_parameters(model)),
        "threshold": thresholds,
        "threshold_source": "validation_split_tuned",
        "val_fidelity": fidelities,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "stopped_epoch": int(history[-1]["epoch"]),
        "stopped_early": bool(stopped_early),
        "epochs_trained": int(len(history)),
        "early_stopping_patience": (
            None if early_stopping_patience is None else int(early_stopping_patience)
        ),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "learning_rate": float(learning_rate),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "crop_cache_path": None if crop_cache_path is None else str(crop_cache_path),
    }


def run_one_split(
    h5_path,
    split_seed,
    crop_size,
    learning_rate,
    epochs,
    batch_size,
    eval_batch_size,
    num_workers,
    device,
    early_stopping_patience,
    early_stopping_min_delta,
    architecture=lightweight_cnn_site_architecture,
    crop_cache_path=None,
):
    architecture = str(architecture)

    with h5py.File(h5_path, "r") as h5_file:
        n_samples = h5_file["images"].shape[0]

    train_idx, val_idx, test_idx = make_random_split(
        n_samples=n_samples,
        seed=split_seed,
    )
    set_random_seed(split_seed, device=device)

    mean_image, scale = compute_cnn_preprocessing(h5_path, train_idx)

    neighbor_sigma_px = None
    if architecture_uses_psf_weighting(architecture):
        neighbor_sigma_px = fit_neighbor_sigma_px(
            h5_path=h5_path,
            train_idx=train_idx,
            num_workers=num_workers,
        )

    trained = train_lightweight_cnn_site(
        h5_path=h5_path,
        train_idx=train_idx,
        val_idx=val_idx,
        architecture=architecture,
        crop_size=crop_size,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        mean_image=mean_image,
        scale=scale,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        crop_cache_path=crop_cache_path,
        neighbor_sigma_px=neighbor_sigma_px,
    )

    test_loader = make_site_loader(
        h5_path=h5_path,
        indices=test_idx,
        crop_size=crop_size,
        mean_image=mean_image,
        scale=scale,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=str(device).startswith("cuda"),
        crop_cache_path=crop_cache_path,
        return_sites=architecture_requires_site_coordinates(architecture),
    )
    model = load_lightweight_cnn_site_model(trained, device=device)
    scores, labels = calc_lightweight_cnn_site_scores_from_site_loader(
        dataloader=test_loader,
        model=model,
        device=device,
    )
    test_metrics = evaluate_lightweight_cnn_site_scores_labels(scores, labels, trained)

    return {
        "seed": split_seed,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "mean_image": mean_image,
        "scale": scale,
        "neighbor_sigma_px": neighbor_sigma_px,
        "best_epoch": int(trained["best_epoch"]),
        "best_val_loss": float(trained["best_val_loss"]),
        "stopped_epoch": int(trained["stopped_epoch"]),
        "stopped_early": bool(trained["stopped_early"]),
        "epochs_trained": int(trained["epochs_trained"]),
        "thresholds": np.asarray(trained["threshold"]).copy(),
        "val_fidelity": np.asarray(trained["val_fidelity"]).copy(),
        "test_fidelity": np.asarray(test_metrics["fidelity"]).copy(),
        "test_mean_fidelity": float(test_metrics["mean_fidelity"]),
        "test_raw_accuracy": float(test_metrics["raw_accuracy"]),
        "test_errors_per_site": np.asarray(test_metrics["errors_per_site"]).copy(),
        "test_total_errors": int(test_metrics["total_errors"]),
        "trained_model": trained,
        "crop_cache_path": None if crop_cache_path is None else str(crop_cache_path),
    }


def main(
    data_dir,
    model_dir,
    output_dir=default_output_dir,
    selected_architectures=selected_architectures,
    crop_size=crop_size,
    learning_rate=learning_rate,
    epochs=epochs,
    batch_size=batch_size,
    eval_batch_size=eval_batch_size,
    num_workers=num_workers,
    base_seed=base_seed,
    device=device,
    early_stopping_patience=early_stopping_patience,
    early_stopping_min_delta=early_stopping_min_delta,
    save_learning_curves=True,
    use_site_crop_cache=use_site_crop_cache,
    site_crop_cache_batch_size=site_crop_cache_batch_size,
):
    model_dir.mkdir(parents=True, exist_ok=True)
    if save_learning_curves:
        learning_curve_dir = output_dir / "learning_curves"
        learning_curve_dir.mkdir(parents=True, exist_ok=True)
    if use_site_crop_cache:
        site_crop_cache_dir = output_dir / "site_crop_cache"
        site_crop_cache_dir.mkdir(parents=True, exist_ok=True)

    h5_paths = sorted(data_dir.glob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"no .h5 datasets found in {data_dir}")

    device = device or get_device()

    print("Lightweight CNN-site training")
    print(f"  datasets:             {len(h5_paths)}")
    print(f"  architectures:        {len(selected_architectures)}")
    print(f"  crop size:            {crop_size}x{crop_size}")
    print(f"  device:               {device}")
    print(f"  crop cache:           {bool(use_site_crop_cache)}")
    print("Architectures:")
    for architecture in selected_architectures:
        model = LightweightCnnSiteNet(crop_size=crop_size, architecture=architecture)
        print(
            f"  {architecture:<40} "
            f"{count_parameters(model):>6,d} params | "
            f"{architecture_configs[str(architecture)]['description']}"
        )

    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as h5_file:
            n_samples = h5_file["images"].shape[0]

        if n_samples < 5:
            print(f"Skipping {h5_path}: dataset has only {n_samples} images")
            continue

        crop_cache_path = None
        if use_site_crop_cache:
            crop_cache_path = build_site_crop_cache(
                h5_path=h5_path,
                cache_path=site_crop_cache_dir / f"{h5_path.stem}_crop{crop_size}.h5",
                crop_size=crop_size,
                batch_size=site_crop_cache_batch_size,
            )

        for architecture in selected_architectures:
            architecture = str(architecture)
            output_path = output_model_path(model_dir, h5_path, architecture)

            result = run_one_split(
                h5_path=h5_path,
                split_seed=base_seed,
                crop_size=crop_size,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                num_workers=num_workers,
                device=device,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                architecture=architecture,
                crop_cache_path=crop_cache_path,
            )
            trained = result["trained_model"]

            save_obj = {
                **trained,
                "h5_path": str(h5_path),
                "seed": base_seed,
                "train_idx": result["train_idx"],
                "val_idx": result["val_idx"],
                "test_idx": result["test_idx"],
                "mean_image": result["mean_image"],
                "scale": result["scale"],
                "eval_batch_size": int(eval_batch_size),
                "thresholds": result["thresholds"],
                "test_fidelity": result["test_fidelity"],
                "test_mean_fidelity": result["test_mean_fidelity"],
                "test_raw_accuracy": result["test_raw_accuracy"],
                "test_errors_per_site": result["test_errors_per_site"],
                "test_total_errors": result["test_total_errors"],
                "preprocessing": "subtract_train_global_mean_divide_train_pixel_range",
            }
            joblib.dump(save_obj, output_path)

            sigma_note = ""
            if result["neighbor_sigma_px"] is not None:
                sigma_note = f" | PSF sigma={result['neighbor_sigma_px']:.3f} px"
            feature_scale = trained.get("neighbor_feature_scale", 1.0)
            if feature_scale != 1.0:
                sigma_note += f" | neighbor scale={feature_scale:.5f}"
            print(
                f"Saved {output_path} | "
                f"params={trained['parameters']:,} | "
                f"mean F={result['test_mean_fidelity']:.8f} | "
                f"best epoch={result['best_epoch']} | "
                f"trained epochs={result['epochs_trained']}"
                f"{sigma_note}"
            )

            if save_learning_curves:
                from plots.plot_learning_curves import plot_learning_curves

                learning_curve_path = learning_curve_dir / (
                    f"{h5_path.stem}_{trained['method']}_learning_curve.png"
                )
                try:
                    plot_learning_curves(
                        history=trained["history"],
                        output_path=learning_curve_path,
                        title=f"{architecture} learning curve: {h5_path.stem}",
                        best_epoch=result["best_epoch"],
                    )
                    print(f"Saved learning curve: {learning_curve_path}")
                except (ImportError, OSError, RuntimeError, ValueError) as exc:
                    print(f"Skipping learning curve for {h5_path.name}: {exc}")


if __name__ == "__main__":
    main(
        data_dir=default_data_dir,
        model_dir=default_model_dir,
        output_dir=default_output_dir,
    )
