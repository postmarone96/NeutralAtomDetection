from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
from plots.plot_learning_curves import plot_learning_curves

from baselines import compute_average_image, fit_shared_sigma_from_average_image
from cnn_site import (
    build_site_crop_cache,
    compute_cnn_preprocessing,
    make_site_loader,
    set_random_seed,
)
from mf_site import classification_fidelity
from utils import exposure_s, make_random_split, path_from_root
import constants


project_root = Path(__file__).resolve().parent
default_data_dir = project_root / "dataset_3x3"
default_model_dir = project_root / "models_3x3"
default_output_dir = project_root / "output_3x3"

lightweight_cnn_site_method = constants.tinycnn_method
lightweight_cnn_site_neighbor_method = constants.tinycnn_neighbor_method
lightweight_cnn_site_neighbor_gaussian_method = constants.tinycnn_neighbor_gaussian_method

selected_methods = (
    lightweight_cnn_site_method,
    lightweight_cnn_site_neighbor_method,
    lightweight_cnn_site_neighbor_gaussian_method,
)

crop_size = 10
channels = constants.tinycnn_channels
learning_rate = constants.tinycnn_learning_rate
epochs = constants.tinycnn_epochs
early_stopping_patience = constants.tinycnn_early_stopping_patience
early_stopping_min_delta = constants.tinycnn_early_stopping_min_delta
batch_size = constants.tinycnn_batch_size
eval_batch_size = constants.tinycnn_eval_batch_size
num_workers = 10
base_seed = 42
device = "cuda:1"
site_crop_cache_batch_size = 512
resume = False
selected_exposures = None
neighbor_directions = constants.tinycnn_neighbor_directions
neighbor_feature_count = constants.tinycnn_neighbor_feature_count
neighbor_feature_methods = (
    lightweight_cnn_site_neighbor_method,
    lightweight_cnn_site_neighbor_gaussian_method,
)
min_grid_fill = 0.5



class LightweightCnnSiteNet(nn.Module):
    def __init__(
        self,
        crop_size,
        method=lightweight_cnn_site_method,
        neighbor_sigma_px=None,
        neighbor_feature_scale=None,
        neighbor_noise_floor=None,
        channels=channels,
    ):
        super().__init__()
        c1, c2, c3 = channels
        self.method = method
        self.neighbor_feature_count = (
            neighbor_feature_count if method in neighbor_feature_methods else 0
        )

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_features = c3 * (crop_size // 4) ** 2
        if self.neighbor_feature_count:
            self.classifier = nn.Sequential(
                nn.Linear(conv_features, 8),
                nn.ReLU(),
                nn.Linear(8 + self.neighbor_feature_count, 8),
                nn.ReLU(),
                nn.Linear(8, 2),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(conv_features, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 2),
            )

        if method == lightweight_cnn_site_neighbor_gaussian_method:
            if neighbor_sigma_px is None:
                raise ValueError("neighbor_sigma_px is None")
            self.register_buffer(
                "neighbor_sigma_px", torch.tensor(float(neighbor_sigma_px))
            )

        if self.neighbor_feature_count:
            if neighbor_noise_floor is None:
                raise ValueError("neighbor_noise_floor is None")
            if neighbor_feature_scale is None:
                raise ValueError("neighbor_feature_scale is None")
            self.register_buffer(
                "neighbor_noise_floor", torch.tensor(float(neighbor_noise_floor))
            )
            self.register_buffer(
                "neighbor_feature_scale", torch.tensor(float(neighbor_feature_scale))
            )

    def forward(self, x, neighbor_features=None):
        features = self.features(x)

        if not self.neighbor_feature_count:
            return self.classifier(features)

        if neighbor_features is None:
            raise ValueError(
                f"{self.method} requires "
                f"{self.neighbor_feature_count} neighbor features"
            )

        hidden = self.classifier[1](self.classifier[0](features))
        hidden = self.classifier[3](
            self.classifier[2](torch.cat([hidden, neighbor_features], dim=1))
        )
        return self.classifier[4](hidden)


def grid_positions(values):
    """Grid line of every coordinate, and the centre of every line."""
    order = np.argsort(values, kind="stable")
    steps = np.diff(values[order])
    new_line = np.concatenate([[True], steps > steps.max(initial=0.0) / 2])
    positions = np.empty(len(values), dtype=np.int64)
    positions[order] = np.cumsum(new_line) - 1
    centers = np.array(
        [values[positions == line].mean() for line in range(positions.max() + 1)]
    )
    return positions, centers


def neighbor_table(sites):
    """Grid neighbours of every site, in `neighbor_directions` order.

    `sites` holds (n_sites, 2) x/y coordinates on a rectangular lattice, which
    may have holes. Returns (n_sites, 8) neighbour indices, -1 where the grid
    position is outside the grid or empty, and the (8,) distances to those grid
    positions, which are the same for every site.
    """
    sites = np.asarray(sites, dtype=np.float64)
    columns, column_centers = grid_positions(sites[:, 0])
    rows, row_centers = grid_positions(sites[:, 1])
    n_rows, n_columns = len(row_centers), len(column_centers)

    grid = np.full((n_rows, n_columns), -1, dtype=np.int64)
    grid[rows, columns] = np.arange(len(sites))
    filled = int((grid >= 0).sum())
    if (
        min(n_rows, n_columns) < 2
        or filled != len(sites)
        or filled < min_grid_fill * grid.size
    ):
        raise ValueError(
            f"{len(sites)} sites do not lie on a {n_rows}x{n_columns} grid "
            f"({filled} of {grid.size} positions filled)"
        )

    indices = np.full((len(sites), neighbor_feature_count), -1, dtype=np.int64)
    for slot, (dx, dy) in enumerate(neighbor_directions):
        row = rows + dy
        column = columns + dx
        inside = (row >= 0) & (row < n_rows) & (column >= 0) & (column < n_columns)
        indices[inside, slot] = grid[row[inside], column[inside]]

    spacing = np.array([np.diff(column_centers).mean(), np.diff(row_centers).mean()])
    distances = np.hypot(*(np.asarray(neighbor_directions) * spacing).T)
    return indices, distances


def crop_values(crops, method, sigma_px=None):
    if method == lightweight_cnn_site_neighbor_method:
        return crops.mean(dim=(2, 3, 4))

    if method == lightweight_cnn_site_neighbor_gaussian_method:
        size = crops.shape[-1]
        axis = torch.arange(size, dtype=crops.dtype, device=crops.device) - size // 2
        kernel = torch.exp(
            -(axis[:, None] ** 2 + axis[None, :] ** 2) / (2.0 * float(sigma_px) ** 2)
        )
        return (crops * (kernel / kernel.sum())).sum(dim=(2, 3, 4))

    raise ValueError(f"{method} has no neighbour features")


def neighbor_features_from_crops(crops, sites, method, noise_floor, feature_scale, sigma_px=None):
    """Grid-neighbour crop values divided by `feature_scale`.

    Neighbours outside the grid read `noise_floor`.
    """
    batch, n_sites = crops.shape[:2]
    site_values = torch.cat(
        [
            crop_values(crops, method, sigma_px),
            torch.full(
                (batch, 1), float(noise_floor), dtype=crops.dtype, device=crops.device
            ),
        ],
        dim=1,
    )

    grids, image_grid = torch.unique(
        sites.detach().cpu().double(), dim=0, return_inverse=True
    )
    tables = [neighbor_table(grid.numpy()) for grid in grids]
    indices = torch.from_numpy(np.stack([table[0] for table in tables]))[image_grid]
    indices = torch.where(indices < 0, n_sites, indices).to(crops.device)

    features = torch.gather(
        site_values, dim=1, index=indices.reshape(batch, -1)
    ).reshape(batch, n_sites, neighbor_feature_count)
    return features / float(feature_scale)


def prepare_site_batch(batch, model, device):
    """Unpack, move to device and flatten one site batch.

    Returns flattened (batch*n_sites) crops, labels and neighbour features,
    plus the image and site counts needed to reshape scores back.
    """
    crops, labels, *rest = batch
    crops = crops.to(device)
    labels = labels.to(device)
    image_count, site_count = labels.shape

    neighbor_features = None
    if model.neighbor_feature_count:
        neighbor_features = neighbor_features_from_crops(
            crops,
            sites=rest[0],
            method=model.method,
            noise_floor=model.neighbor_noise_floor,
            feature_scale=model.neighbor_feature_scale,
            sigma_px=getattr(model, "neighbor_sigma_px", None),
        ).reshape(image_count * site_count, -1)

    crops = crops.reshape(image_count * site_count, 1, *crops.shape[-2:])
    labels = labels.reshape(image_count * site_count)
    return crops, labels, neighbor_features, image_count, site_count

@torch.no_grad()
def fit_neighbor_feature_scale(dataloader, method, noise_floor, sigma_px=None, max_batches=16):
    """Largest unscaled neighbour feature in the first `max_batches` batches."""
    largest = 0.0
    for batch_index, (crops, _labels, sites) in enumerate(dataloader):
        if batch_index >= max_batches:
            break
        features = neighbor_features_from_crops(
            crops,
            sites,
            method=method,
            noise_floor=noise_floor,
            feature_scale=1.0,
            sigma_px=sigma_px,
        )
        largest = max(largest, float(features.abs().max()))
    return largest

@torch.no_grad()
def fit_neighbor_noise_floor(dataloader, method, sigma_px=None):
    """Mean crop value of the empty sites in `dataloader`."""
    total = 0.0
    count = 0
    for crops, labels, *_ in dataloader:
        empty = labels == 0
        total += float(crop_values(crops, method, sigma_px)[empty].double().sum())
        count += int(empty.sum())
    return total / count


def fit_neighbor_sigma_px(h5_path, train_idx, num_workers=4):
    avg_image, sites = compute_average_image(
        h5_path=h5_path,
        indices=train_idx,
        num_workers=num_workers,
    )
    return fit_shared_sigma_from_average_image(avg_image, sites)


def load_lightweight_cnn_site_model(best, device="cpu"):
    model = LightweightCnnSiteNet(
        crop_size=best["crop_size"],
        method=best["method"],
        neighbor_sigma_px=best.get("neighbor_sigma_px"),
        neighbor_feature_scale=best.get("neighbor_feature_scale"),
        neighbor_noise_floor=best.get("neighbor_noise_floor"),
        channels=best.get("channels", channels),
    ).to(device)
    model.load_state_dict(best["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_lightweight_loss_accuracy(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    for batch in dataloader:
        crops, labels, neighbor_features, _images, _sites = prepare_site_batch(
            batch, model, device
        )
        logits = model(crops, neighbor_features)
        loss = loss_fn(logits, labels)

        examples = labels.numel()
        total_loss += float(loss.item()) * examples
        total_examples += examples
        total_correct += int((logits.argmax(dim=1) == labels).sum())

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


@torch.no_grad()
def calc_lightweight_cnn_site_logits_from_site_loader(dataloader, model, device="cpu"):
    model.eval()
    all_logits = []
    all_labels = []

    for batch in dataloader:
        crops, labels, neighbor_features, image_count, site_count = prepare_site_batch(
            batch, model, device
        )
        logits = model(crops, neighbor_features)

        all_logits.append(logits.reshape(image_count, site_count, 2).cpu())
        all_labels.append(labels.reshape(image_count, site_count).float().cpu())

    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def evaluate_lightweight_cnn_site_logits_labels(logits, labels):
    """Each site is bright when logit1 > logit0, the comparison the FPGA makes."""
    decisions = (logits[..., 1] > logits[..., 0]).astype(np.float32)
    fidelities = np.array(
        [
            classification_fidelity(decisions[:, site_index], labels[:, site_index], 0.5)
            for site_index in range(labels.shape[1])
        ],
        dtype=np.float32,
    )
    errors = (decisions > 0.5) != (labels > 0.5)

    return {
        "fidelity": fidelities,
        "mean_fidelity": float(np.mean(fidelities)),
        "errors_per_site": errors.sum(axis=0).astype(np.int64),
        "total_errors": int(errors.sum()),
        "raw_accuracy": float(1.0 - errors.mean()),
    }


def train_lightweight_cnn_site(
    h5_path,
    train_idx,
    val_idx,
    method=lightweight_cnn_site_method,
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
    channels=channels,
):
    uses_neighbors = method in neighbor_feature_methods

    loader_options = {
        "h5_path": h5_path,
        "crop_size": crop_size,
        "mean_image": mean_image,
        "scale": scale,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": str(device).startswith("cuda"),
        "crop_cache_path": crop_cache_path,
        "return_sites": uses_neighbors,
    }
    train_loader = make_site_loader(indices=train_idx, shuffle=True, **loader_options)
    val_loader = make_site_loader(indices=val_idx, shuffle=False, **loader_options)

    neighbor_noise_floor = None
    neighbor_feature_scale = None
    if uses_neighbors:
        fit_loader = make_site_loader(indices=train_idx, shuffle=False, **loader_options)
        neighbor_noise_floor = fit_neighbor_noise_floor(
            fit_loader, method=method, sigma_px=neighbor_sigma_px
        )
        neighbor_feature_scale = fit_neighbor_feature_scale(
            fit_loader,
            method=method,
            noise_floor=neighbor_noise_floor,
            sigma_px=neighbor_sigma_px,
        )

    model = LightweightCnnSiteNet(
        crop_size=crop_size,
        method=method,
        neighbor_sigma_px=neighbor_sigma_px,
        neighbor_feature_scale=neighbor_feature_scale,
        neighbor_noise_floor=neighbor_noise_floor,
        channels=channels,
    ).to(device)

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
        desc=f"{Path(h5_path).stem} {method}",
        unit="epoch",
        leave=False,
    )
    for epoch in progress:
        model.train()
        train_loss_sum = 0.0
        train_examples = 0
        train_correct = 0

        for batch in train_loader:
            crops, labels, neighbor_features, _images, _sites = prepare_site_batch(
                batch, model, device
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(crops, neighbor_features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            examples = labels.numel()
            train_loss_sum += float(loss.item()) * examples
            train_examples += examples
            train_correct += int((logits.argmax(dim=1) == labels).sum())

        train_loss = train_loss_sum / train_examples
        train_accuracy = train_correct / train_examples
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

        if no_improvement_epochs >= early_stopping_patience:
            stopped_early = True
            break

    model.load_state_dict(best_state)
    val_logits, val_labels = calc_lightweight_cnn_site_logits_from_site_loader(
        dataloader=val_loader,
        model=model,
        device=device,
    )
    val_decision_metrics = evaluate_lightweight_cnn_site_logits_labels(val_logits, val_labels)

    return {
        "method": method,
        "neighbor_sigma_px": neighbor_sigma_px,
        "neighbor_feature_scale": neighbor_feature_scale,
        "neighbor_noise_floor": neighbor_noise_floor,
        "state_dict": best_state,
        "crop_size": crop_size,
        "channels": tuple(channels),
        "val_fidelity": val_decision_metrics["fidelity"],
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "stopped_epoch": history[-1]["epoch"],
        "stopped_early": stopped_early,
        "epochs_trained": len(history),
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
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
    method=lightweight_cnn_site_method,
    crop_cache_path=None,
    channels=channels,
):
    with h5py.File(h5_path, "r") as h5_file:
        n_samples = h5_file["images"].shape[0]

    train_idx, val_idx, test_idx = make_random_split(
        n_samples=n_samples,
        seed=split_seed,
    )
    set_random_seed(split_seed, device=device)

    mean_image, scale = compute_cnn_preprocessing(h5_path, train_idx)

    neighbor_sigma_px = None
    if method == lightweight_cnn_site_neighbor_gaussian_method:
        neighbor_sigma_px = fit_neighbor_sigma_px(
            h5_path=h5_path,
            train_idx=train_idx,
            num_workers=num_workers,
        )

    trained = train_lightweight_cnn_site(
        h5_path=h5_path,
        train_idx=train_idx,
        val_idx=val_idx,
        method=method,
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
        channels=channels,
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
        return_sites=method in neighbor_feature_methods,
    )
    model = load_lightweight_cnn_site_model(trained, device=device)
    logits, labels = calc_lightweight_cnn_site_logits_from_site_loader(
        dataloader=test_loader,
        model=model,
        device=device,
    )
    test_metrics = evaluate_lightweight_cnn_site_logits_labels(logits, labels)

    return {
        "seed": split_seed,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "mean_image": mean_image,
        "scale": scale,
        "test_fidelity": test_metrics["fidelity"],
        "test_mean_fidelity": test_metrics["mean_fidelity"],
        "test_raw_accuracy": test_metrics["raw_accuracy"],
        "test_errors_per_site": test_metrics["errors_per_site"],
        "test_total_errors": test_metrics["total_errors"],
        "trained_model": trained,
    }


def main(
    data_dir,
    model_dir,
    output_dir=default_output_dir,
    selected_methods=selected_methods,
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
    site_crop_cache_batch_size=site_crop_cache_batch_size,
    selected_exposures=selected_exposures,
    resume=resume,
    channels=channels,
):
    model_dir.mkdir(parents=True, exist_ok=True)

    learning_curve_dir = output_dir / "learning_curves"
    learning_curve_dir.mkdir(parents=True, exist_ok=True)

    site_crop_cache_dir = output_dir / "site_crop_cache"
    site_crop_cache_dir.mkdir(parents=True, exist_ok=True)

    h5_paths = sorted(data_dir.glob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"no .h5 datasets found in {data_dir}")
    if selected_exposures is not None:
        selected_exposures = set(selected_exposures)
        h5_paths = [
            h5_path for h5_path in h5_paths
            if exposure_s(h5_path) in selected_exposures
        ]
    if not h5_paths:
        raise ValueError("selected_exposures selected no datasets")

    print("Lightweight CNN-site training")
    print(f"  datasets:             {len(h5_paths)}")
    print(f"  selected exposures:   {selected_exposures}")
    print(f"  methods:              {len(selected_methods)}")
    print(f"  crop size:            {crop_size}x{crop_size}")
    print(f"  device:               {device}")
    print("Methods:")
    for method in selected_methods:
        print(f"  {method}")

    for h5_path in h5_paths:
        pending_methods = []
        for method in selected_methods:
            output_path = model_dir / f"{h5_path.stem}_{method}.joblib"
            if resume and output_path.exists():
                print(f"Skipping {output_path.name}: already trained")
            else:
                pending_methods.append(method)
        if not pending_methods:
            continue

        crop_cache_path = build_site_crop_cache(
            h5_path=h5_path,
            cache_path=site_crop_cache_dir / f"{h5_path.stem}_crop{crop_size}.h5",
            crop_size=crop_size,
            batch_size=site_crop_cache_batch_size,
        )

        for method in pending_methods:
            output_path = model_dir / f"{h5_path.stem}_{method}.joblib"

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
                method=method,
                crop_cache_path=crop_cache_path,
                channels=channels,
            )
            trained = result.pop("trained_model")
            joblib.dump({**result, **trained, "h5_path": path_from_root(h5_path)}, output_path)

            notes = ""
            if trained["neighbor_sigma_px"] is not None:
                notes += f" | PSF sigma={trained['neighbor_sigma_px']:.3f} px"
            if trained["neighbor_noise_floor"] is not None:
                notes += (
                    f" | neighbor scale={trained['neighbor_feature_scale']:.5f}"
                    f" | noise floor={trained['neighbor_noise_floor']:.5f}"
                )
            print(
                f"Saved {output_path} | "
                f"mean F={result['test_mean_fidelity']:.8f} | "
                f"best epoch={trained['best_epoch']} | "
                f"trained epochs={trained['epochs_trained']}"
                f"{notes}"
            )

            learning_curve_path = learning_curve_dir / (
                f"{h5_path.stem}_{trained['method']}_learning_curve.png")
            plot_learning_curves(
                history=trained["history"],
                output_path=learning_curve_path,
                title=f"{method} learning curve: {h5_path.stem}",
                best_epoch=trained["best_epoch"],
            )
            print(f"Saved learning curve: {learning_curve_path}")


if __name__ == "__main__":
    main(
        data_dir=default_data_dir,
        model_dir=default_model_dir,
        output_dir=default_output_dir,
    )
