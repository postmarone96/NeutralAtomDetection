from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from mf_site import classification_fidelity
from utils import H5Dataset, extract_patches, get_device


default_data_dir = Path(
    "dataset_3x3"
)
default_model_dir = Path("models_3x3")
cnn_site_decision_threshold = 0.5
crop_size = 10
learning_rate = 1e-4
epochs = 40
batch_size = 64
eval_batch_size = 2048
num_workers = 2
base_seed = 42
device = "cuda"
site_crop_cache_version = 1


def make_random_split(
    n_samples: int,
    seed: int,
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
):
    if n_samples < 5:
        raise ValueError("At least 5 samples are required.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    n_train = int(train_fraction * n_samples)
    n_val = int(val_fraction * n_samples)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def set_random_seed(seed, device="cpu"):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def compute_cnn_preprocessing(h5_path, train_idx, image_key="images", chunk_size=64):
    """
    Paper-style CNN preprocessing:
    I' = (I - mean training pixel intensity) / (train max - train min).
    """
    if len(train_idx) == 0:
        raise ValueError(f"Cannot compute preprocessing for {h5_path}: train_idx is empty")

    train_idx = np.sort(np.asarray(train_idx, dtype=np.int64))

    with h5py.File(h5_path, "r") as f:
        images = f[image_key]
        total = 0.0
        count = 0
        min_value = np.inf
        max_value = -np.inf

        for start in range(0, len(train_idx), chunk_size):
            chunk_idx = train_idx[start:start + chunk_size]
            image_chunk = images[chunk_idx].astype(np.float32)
            total += float(image_chunk.sum(dtype=np.float64))
            count += int(image_chunk.size)
            min_value = min(min_value, float(image_chunk.min()))
            max_value = max(max_value, float(image_chunk.max()))

    mean_pixel = np.float32(total / count)
    scale = float(max_value - min_value)
    if scale == 0.0:
        scale = 1.0

    return mean_pixel, scale


def extract_site_crop(image, site, crop_size):
    height, width = image.shape
    half = crop_size // 2
    pad = half + 1

    padded = np.pad(
        image,
        pad_width=((pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0.0,
    )

    center_x = int(np.rint(site[0])) + pad
    center_y = int(np.rint(site[1])) + pad

    x0 = center_x - half
    y0 = center_y - half
    crop = padded[y0:y0 + crop_size, x0:x0 + crop_size]

    if crop.shape != (crop_size, crop_size):
        raise ValueError(
            f"expected crop shape {(crop_size, crop_size)}, got {crop.shape} "
            f"from image shape {(height, width)}"
        )

    return crop.astype(np.float32, copy=False)


def extract_site_crops(image, sites, crop_size):
    height, width = image.shape
    half = crop_size // 2
    pad = half + 1

    padded = np.pad(
        image,
        pad_width=((pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0.0,
    )

    sites = np.asarray(sites, dtype=np.float32)
    centers_x = np.rint(sites[:, 0]).astype(np.int64) + pad
    centers_y = np.rint(sites[:, 1]).astype(np.int64) + pad

    crops = np.empty((sites.shape[0], crop_size, crop_size), dtype=np.float32)
    for site_index, (center_x, center_y) in enumerate(zip(centers_x, centers_y)):
        x0 = int(center_x) - half
        y0 = int(center_y) - half
        crop = padded[y0:y0 + crop_size, x0:x0 + crop_size]

        if crop.shape != (crop_size, crop_size):
            raise ValueError(
                f"expected crop shape {(crop_size, crop_size)}, got {crop.shape} "
                f"from image shape {(height, width)}"
            )

        crops[site_index] = crop

    return crops


def _source_file_signature(h5_path):
    h5_path = Path(h5_path)
    stat = h5_path.stat()
    return {
        "source_h5_path": str(h5_path.resolve()),
        "source_h5_size": int(stat.st_size),
        "source_h5_mtime_ns": int(stat.st_mtime_ns),
    }


def _site_crop_cache_matches(
    cache_path,
    h5_path,
    crop_size,
    image_key="images",
    label_key="labels",
    site_key="sites",
):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return False

    signature = _source_file_signature(h5_path)
    try:
        with h5py.File(h5_path, "r") as source, h5py.File(cache_path, "r") as cache:
            if "crops" not in cache or "labels" not in cache or "sites" not in cache:
                return False

            expected_shape = (
                source[image_key].shape[0],
                source[site_key].shape[0],
                int(crop_size),
                int(crop_size),
            )
            if tuple(cache["crops"].shape) != expected_shape:
                return False
            if tuple(cache["labels"].shape) != tuple(source[label_key].shape):
                return False
            if tuple(cache["sites"].shape) != tuple(source[site_key].shape):
                return False

            attrs = cache.attrs
            if int(attrs.get("cache_version", -1)) != site_crop_cache_version:
                return False
            if int(attrs.get("crop_size", -1)) != int(crop_size):
                return False
            for key, value in signature.items():
                if attrs.get(key) != value:
                    return False
    except OSError:
        return False

    return True


def build_site_crop_cache(
    h5_path,
    cache_path,
    crop_size,
    image_key="images",
    label_key="labels",
    site_key="sites",
    batch_size=16,
    overwrite=False,
):
    cache_path = Path(cache_path)
    crop_size = int(crop_size)

    if (
        not overwrite
        and _site_crop_cache_matches(
            cache_path=cache_path,
            h5_path=h5_path,
            crop_size=crop_size,
            image_key=image_key,
            label_key=label_key,
            site_key=site_key,
        )
    ):
        print(f"Using site crop cache: {cache_path}")
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.unlink(missing_ok=True)

    with h5py.File(h5_path, "r") as source:
        images = source[image_key]
        labels = source[label_key]
        sites = source[site_key][:].astype(np.float32)

        n_images = int(images.shape[0])
        n_sites = int(sites.shape[0])
        if labels.shape != (n_images, n_sites):
            raise ValueError(
                f"expected labels shape {(n_images, n_sites)}, got {labels.shape}"
            )

        crop_dtype = images.dtype if np.issubdtype(images.dtype, np.integer) else np.float32

        with h5py.File(tmp_path, "w") as cache:
            crop_dataset = cache.create_dataset(
                "crops",
                shape=(n_images, n_sites, crop_size, crop_size),
                dtype=crop_dtype,
                chunks=(1, n_sites, crop_size, crop_size),
            )
            label_dataset = cache.create_dataset(
                "labels",
                shape=labels.shape,
                dtype=labels.dtype,
                chunks=(1, n_sites),
            )
            cache.create_dataset("sites", data=sites, dtype=np.float32)

            cache.attrs["cache_version"] = site_crop_cache_version
            cache.attrs["crop_size"] = crop_size
            cache.attrs["image_key"] = image_key
            cache.attrs["label_key"] = label_key
            cache.attrs["site_key"] = site_key
            cache.attrs["preprocessing"] = "raw_site_crops_before_global_mean_range_normalization"
            for key, value in _source_file_signature(h5_path).items():
                cache.attrs[key] = value

            progress = tqdm(
                range(0, n_images, int(batch_size)),
                desc=f"Building {crop_size}x{crop_size} site crop cache",
                unit="batch",
                leave=False,
            )
            for start in progress:
                stop = min(start + int(batch_size), n_images)
                image_batch = images[start:stop]
                crop_batch = np.empty(
                    (stop - start, n_sites, crop_size, crop_size),
                    dtype=crop_dtype,
                )
                for local_index, image in enumerate(image_batch):
                    crop_batch[local_index] = extract_site_crops(
                        image=image.astype(np.float32, copy=False),
                        sites=sites,
                        crop_size=crop_size,
                    ).astype(crop_dtype, copy=False)

                crop_dataset[start:stop] = crop_batch
                label_dataset[start:stop] = labels[start:stop]

    tmp_path.replace(cache_path)
    print(f"Saved site crop cache: {cache_path}")
    return cache_path


class CnnSiteDataset(Dataset):
    """
    Reads one full-array image and returns all single-site crops.

    This avoids reading the same HDF5 image once per site.
    """
    def __init__(
        self,
        h5_path,
        indices,
        crop_size=10,
        image_key="images",
        label_key="labels",
        site_key="sites",
        mean_image=None,
        scale=None,
    ):
        self.h5_path = h5_path
        self.indices = list(indices)
        self.crop_size = int(crop_size)
        self.image_key = image_key
        self.label_key = label_key
        self.site_key = site_key
        self.mean_image = None if mean_image is None else np.asarray(mean_image, dtype=np.float32)
        self.scale = None if scale is None else float(scale)
        self._h5 = None

        if self.crop_size <= 6:
            raise ValueError("CNN-site crop_size must be greater than 6 for three valid 3x3 conv layers")

        with h5py.File(h5_path, "r") as f:
            self.n_images = f[image_key].shape[0]
            self.sites = f[site_key][:].astype(np.float32)

        self.n_sites = int(self.sites.shape[0])

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx):
        file_index = int(self.indices[idx])

        f = self._file()
        image = f[self.image_key][file_index].astype(np.float32)
        if self.mean_image is not None:
            image = image - self.mean_image
        if self.scale is not None:
            image = image / self.scale

        crops = extract_site_crops(
            image=image,
            sites=self.sites,
            crop_size=self.crop_size,
        )
        labels = (f[self.label_key][file_index] > 0.5).astype(np.int64)

        return (
            torch.from_numpy(crops[:, None, :, :]).float(),
            torch.from_numpy(labels).long(),
        )

    def __getitems__(self, indices):
        file_indices = np.asarray([self.indices[int(idx)] for idx in indices], dtype=np.int64)
        unique_file_indices, inverse = np.unique(file_indices, return_inverse=True)

        f = self._file()
        images = f[self.image_key][unique_file_indices].astype(np.float32)[inverse]
        labels = (f[self.label_key][unique_file_indices] > 0.5).astype(np.int64)[inverse]

        if self.mean_image is not None:
            images = images - self.mean_image
        if self.scale is not None:
            images = images / self.scale

        return [
            (
                torch.from_numpy(
                    extract_site_crops(
                        image=image,
                        sites=self.sites,
                        crop_size=self.crop_size,
                    )[:, None, :, :]
                ).float(),
                torch.from_numpy(label).long(),
            )
            for image, label in zip(images, labels)
        ]


class CachedCnnSiteDataset(Dataset):
    def __init__(
        self,
        cache_path,
        indices,
        crop_size,
        mean_image=None,
        scale=None,
    ):
        self.cache_path = cache_path
        self.indices = list(indices)
        self.crop_size = int(crop_size)
        self.mean_image = None if mean_image is None else np.asarray(mean_image, dtype=np.float32)
        self.scale = None if scale is None else float(scale)
        self._h5 = None

        if self.mean_image is not None and self.mean_image.ndim != 0:
            raise ValueError("cached site crops only support scalar mean preprocessing")

        with h5py.File(cache_path, "r") as f:
            if int(f.attrs.get("crop_size", -1)) != self.crop_size:
                raise ValueError(
                    f"cache crop_size={f.attrs.get('crop_size')} does not match "
                    f"requested crop_size={self.crop_size}"
                )
            self.n_images = int(f["crops"].shape[0])
            self.n_sites = int(f["crops"].shape[1])

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.cache_path, "r")
        return self._h5

    def _preprocess_crops(self, crops):
        crops = crops.astype(np.float32, copy=False)
        if self.mean_image is not None:
            crops = crops - self.mean_image
        if self.scale is not None:
            crops = crops / self.scale
        return crops

    def __getitem__(self, idx):
        file_index = int(self.indices[idx])

        f = self._file()
        crops = self._preprocess_crops(f["crops"][file_index])
        labels = (f["labels"][file_index] > 0.5).astype(np.int64)

        return (
            torch.from_numpy(crops[:, None, :, :]).float(),
            torch.from_numpy(labels).long(),
        )

    def __getitems__(self, indices):
        file_indices = np.asarray([self.indices[int(idx)] for idx in indices], dtype=np.int64)
        unique_file_indices, inverse = np.unique(file_indices, return_inverse=True)

        f = self._file()
        crops = self._preprocess_crops(f["crops"][unique_file_indices])[inverse]
        labels = (f["labels"][unique_file_indices] > 0.5).astype(np.int64)[inverse]

        return [
            (
                torch.from_numpy(crop[:, None, :, :]).float(),
                torch.from_numpy(label).long(),
            )
            for crop, label in zip(crops, labels)
        ]


class CnnSiteNet(nn.Module):
    def __init__(self, crop_size=10):
        super().__init__()
        crop_size = int(crop_size)
        conv_side = crop_size - 6
        if conv_side <= 0:
            raise ValueError("crop_size must be greater than 6")

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * conv_side * conv_side, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_site_loader(
    h5_path,
    indices,
    crop_size,
    mean_image,
    scale,
    batch_size,
    num_workers,
    shuffle,
    pin_memory=False,
    crop_cache_path=None,
):
    if crop_cache_path is None:
        dataset = CnnSiteDataset(
            h5_path=h5_path,
            indices=indices,
            crop_size=crop_size,
            mean_image=mean_image,
            scale=scale,
        )
    else:
        dataset = CachedCnnSiteDataset(
            cache_path=crop_cache_path,
            indices=indices,
            crop_size=crop_size,
            mean_image=mean_image,
            scale=scale,
        )
    image_batch_size = max(1, batch_size // dataset.n_sites)

    return DataLoader(
        dataset,
        batch_size=image_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def evaluate_loss_accuracy(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    for crops, labels in dataloader:
        crops = crops.to(device)
        labels = labels.to(device)

        batch, n_sites = labels.shape
        crops = crops.reshape(batch * n_sites, 1, crops.shape[-2], crops.shape[-1])
        labels = labels.reshape(batch * n_sites)

        logits = model(crops)
        loss = loss_fn(logits, labels)

        batch = int(labels.numel())
        total_loss += float(loss.item()) * batch
        total_examples += batch
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


@torch.no_grad()
def calc_cnn_site_scores_from_site_loader(
    dataloader,
    model,
    device="cpu",
):
    model.eval()
    all_scores = []
    all_labels = []

    for crops, labels in dataloader:
        batch, n_sites = labels.shape
        crops = crops.to(device)
        crops = crops.reshape(batch * n_sites, 1, crops.shape[-2], crops.shape[-1])

        probabilities = torch.softmax(model(crops), dim=1)[:, 1]
        scores = probabilities.reshape(batch, n_sites)

        all_scores.append(scores.cpu())
        all_labels.append(labels.float())

    return torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


def load_cnn_site_model(best, device="cpu"):
    model = CnnSiteNet(crop_size=int(best.get("crop_size", 10))).to(device)
    model.load_state_dict(best["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def calc_cnn_site_scores(dataloader, best, device="cpu"):
    model = load_cnn_site_model(best, device=device)
    crop_size = int(best.get("crop_size", 10))

    all_scores = []
    all_labels = []

    for images, labels, sites in dataloader:
        images = images.to(device).float()
        labels = labels.float()
        sites = sites[0].to(device)

        batch, n_sites = labels.shape
        patches = extract_patches(images, sites, crop_size)
        patches = patches.reshape(batch, n_sites, crop_size, crop_size)
        crops = patches.reshape(batch * n_sites, 1, crop_size, crop_size)

        probabilities = torch.softmax(model(crops), dim=1)[:, 1]
        scores = probabilities.reshape(batch, n_sites)

        all_scores.append(scores.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


def train_cnn_site(
    h5_path,
    train_idx,
    val_idx,
    crop_size=10,
    learning_rate=1e-4,
    epochs=40,
    batch_size=64,
    num_workers=2,
    device="cpu",
    mean_image=None,
    scale=None,
    crop_cache_path=None,
):
    train_loader = make_site_loader(
        h5_path=h5_path,
        indices=train_idx,
        crop_size=crop_size,
        mean_image=mean_image,
        scale=scale,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=str(device).startswith("cuda"),
        crop_cache_path=crop_cache_path,
    )
    val_loader = make_site_loader(
        h5_path=h5_path,
        indices=val_idx,
        crop_size=crop_size,
        mean_image=mean_image,
        scale=scale,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=str(device).startswith("cuda"),
        crop_cache_path=crop_cache_path,
    )

    n_sites = train_loader.dataset.n_sites
    model = CnnSiteNet(crop_size=crop_size).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    best_state = None
    best_val_loss = np.inf
    best_epoch = -1

    progress = tqdm(range(epochs), desc=f"{Path(h5_path).stem} CNN-site", unit="epoch", leave=False)
    for epoch in progress:
        model.train()
        train_loss_sum = 0.0
        train_examples = 0
        train_correct = 0

        for crops, labels in train_loader:
            crops = crops.to(device)
            labels = labels.to(device)

            batch, n_sites = labels.shape
            crops = crops.reshape(batch * n_sites, 1, crops.shape[-2], crops.shape[-1])
            labels = labels.reshape(batch * n_sites)

            optimizer.zero_grad(set_to_none=True)
            logits = model(crops)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            batch = int(labels.numel())
            train_loss_sum += float(loss.item()) * batch
            train_examples += batch
            train_correct += int((logits.argmax(dim=1) == labels).sum().item())

        train_loss = train_loss_sum / max(train_examples, 1)
        train_accuracy = train_correct / max(train_examples, 1)
        val_metrics = evaluate_loss_accuracy(
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

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        progress.set_postfix(
            train_loss=f"{train_loss:.4g}",
            val_loss=f"{val_metrics['loss']:.4g}",
            val_acc=f"{val_metrics['accuracy']:.4f}",
        )

    if best_state is None:
        raise RuntimeError("CNN-site training did not produce a best checkpoint")

    model.load_state_dict(best_state)
    val_scores, val_labels = calc_cnn_site_scores_from_site_loader(
        dataloader=val_loader,
        model=model,
        device=device,
    )

    thresholds = np.full(n_sites, cnn_site_decision_threshold, dtype=np.float32)
    fidelities = np.zeros(n_sites, dtype=np.float32)
    for q in range(n_sites):
        fidelities[q] = classification_fidelity(
            val_scores[:, q],
            val_labels[:, q],
            cnn_site_decision_threshold,
        )

    return {
        "method": "cnn_site",
        "architecture": "cnn_site_3conv_128fc",
        "state_dict": best_state,
        "crop_size": int(crop_size),
        "threshold": thresholds,
        "val_fidelity": fidelities,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "learning_rate": float(learning_rate),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "crop_cache_path": None if crop_cache_path is None else str(crop_cache_path),
    }


def evaluate_cnn_site_scores_labels(scores, labels, best):
    thresholds = np.asarray(best["threshold"])

    fids = np.zeros(labels.shape[1], dtype=np.float32)
    for q in range(labels.shape[1]):
        fids[q] = classification_fidelity(
            scores[:, q],
            labels[:, q],
            thresholds[q],
        )

    preds = scores >= thresholds[None, :]
    errors = preds != (labels > 0.5)

    return {
        "scores": scores,
        "labels": labels,
        "fidelity": fids,
        "mean_fidelity": float(np.mean(fids)),
        "errors_per_site": errors.sum(axis=0).astype(np.int64),
        "total_errors": int(errors.sum()),
        "raw_accuracy": float(1.0 - errors.mean()),
    }


def evaluate_cnn_site_model(dataloader, best, device="cpu"):
    scores, labels = calc_cnn_site_scores(dataloader, best, device=device)
    return evaluate_cnn_site_scores_labels(scores, labels, best)


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
    crop_cache_path=None,
):
    with h5py.File(h5_path, "r") as f:
        n_samples = f["images"].shape[0]

    train_idx, val_idx, test_idx = make_random_split(
        n_samples=n_samples,
        seed=split_seed,
    )
    set_random_seed(split_seed, device=device)

    mean_image, scale = compute_cnn_preprocessing(h5_path, train_idx)

    trained = train_cnn_site(
        h5_path=h5_path,
        train_idx=train_idx,
        val_idx=val_idx,
        crop_size=crop_size,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        mean_image=mean_image,
        scale=scale,
        crop_cache_path=crop_cache_path,
    )

    if crop_cache_path is None:
        test_loader = DataLoader(
            H5Dataset(
                h5_path,
                indices=test_idx,
                mean_image=mean_image,
                scale=scale,
            ),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=str(device).startswith("cuda"),
            persistent_workers=num_workers > 0,
        )
        test_metrics = evaluate_cnn_site_model(
            test_loader,
            trained,
            device=device,
        )
    else:
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
        )
        model = load_cnn_site_model(trained, device=device)
        scores, labels = calc_cnn_site_scores_from_site_loader(
            dataloader=test_loader,
            model=model,
            device=device,
        )
        test_metrics = evaluate_cnn_site_scores_labels(scores, labels, trained)

    return {
        "seed": split_seed,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "mean_image": mean_image,
        "scale": scale,
        "best_epoch": int(trained["best_epoch"]),
        "best_val_loss": float(trained["best_val_loss"]),
        "thresholds": np.asarray(trained["threshold"]).copy(),
        "val_fidelity": np.asarray(trained["val_fidelity"]).copy(),
        "test_fidelity": np.asarray(test_metrics["fidelity"]).copy(),
        "test_mean_fidelity": float(test_metrics["mean_fidelity"]),
        "test_raw_accuracy": float(test_metrics["raw_accuracy"]),
        "test_errors_per_site": np.asarray(
            test_metrics["errors_per_site"]
        ).copy(),
        "test_total_errors": int(test_metrics["total_errors"]),
        "trained_model": trained,
        "crop_cache_path": None if crop_cache_path is None else str(crop_cache_path),
    }


def main(
    data_dir,
    model_dir,
    crop_size=crop_size,
    learning_rate=learning_rate,
    epochs=epochs,
    batch_size=batch_size,
    eval_batch_size=eval_batch_size,
    num_workers=num_workers,
    base_seed=base_seed,
    device=device,
):
    model_dir.mkdir(parents=True, exist_ok=True)

    h5_paths = sorted(data_dir.glob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"no .h5 datasets found in {data_dir}")

    device = device or get_device()

    for h5_path in h5_paths:
        output_path = model_dir / f"{h5_path.stem}_cnn_site.joblib"

        with h5py.File(h5_path, "r") as f:
            n_samples = f["images"].shape[0]

        if n_samples < 5:
            print(f"Skipping {h5_path}: dataset has only {n_samples} images")
            continue

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
        )
        trained = result["trained_model"]

        save_obj = {
            **trained,
            "method": "cnn_site",
            "h5_path": str(h5_path),
            "seed": base_seed,
            "train_idx": result["train_idx"],
            "val_idx": result["val_idx"],
            "test_idx": result["test_idx"],
            "mean_image": result["mean_image"],
            "scale": result["scale"],
            "crop_size": int(crop_size),
            "learning_rate": float(learning_rate),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "eval_batch_size": int(eval_batch_size),
            "thresholds": result["thresholds"],
            "test_fidelity": result["test_fidelity"],
            "test_mean_fidelity": result["test_mean_fidelity"],
            "test_raw_accuracy": result["test_raw_accuracy"],
            "test_errors_per_site": result["test_errors_per_site"],
            "test_total_errors": result["test_total_errors"],
            "preprocessing": "subtract_train_global_mean_divide_train_pixel_range",
            "architecture": "cnn_site_3conv_128fc",
        }

        joblib.dump(save_obj, output_path)

        print(
            f"Saved {output_path} | "
            f"mean F={result['test_mean_fidelity']:.8f} | "
            f"best epoch={result['best_epoch']}"
        )


if __name__ == "__main__":
    main(
        data_dir=default_data_dir,
        model_dir=default_model_dir,
    )
