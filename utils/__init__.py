import os
import re
from pathlib import Path

import torch
import h5py
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Dataset and model filenames
#
# Every dataset is  qubit_dataset_sim_<exposure seconds>.h5  and every model is
# qubit_dataset_sim_<exposure seconds>_<method>.joblib.  The exposure lives in
# the filename in SECONDS; almost everything downstream wants MILLISECONDS.
# ---------------------------------------------------------------------------

project_root = Path(__file__).resolve().parents[1]


def path_from_root(path):
    return os.path.relpath(Path(path).resolve(), project_root)


dataset_prefix = "qubit_dataset_sim"
exposure_re = re.compile(rf"{dataset_prefix}_([0-9]+(?:\.[0-9]+)?)")
h5_name_re = re.compile(rf"{dataset_prefix}_([0-9]+(?:\.[0-9]+)?)\.h5$")
model_name_re = re.compile(rf"{dataset_prefix}_([0-9]+(?:\.[0-9]+)?)_(.+)\.joblib$")


def dataset_stem(exposure):
    """0.026 -> 'qubit_dataset_sim_0.026'. Takes seconds, not milliseconds."""
    return f"{dataset_prefix}_{exposure:g}"


def exposure_s(path):
    """Exposure in seconds, from a dataset or model filename."""
    match = exposure_re.match(Path(path).name)
    if match is None:
        raise ValueError(f"no exposure in filename: {path}")
    return float(match.group(1))


def exposure_ms(path):
    """Exposure in milliseconds, from a dataset or model filename."""
    return 1000.0 * exposure_s(path)


def model_method(path):
    """The method suffix of a model filename, e.g. 'lightweight_cnn_site'."""
    match = model_name_re.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"not a model filename: {path}")
    return match.group(2)


def parse_model_name(path):
    """(exposure_s, exposure_ms, method), or None if the name is not a model.

    Returns None rather than raising because callers glob whole directories and
    use this to filter out the files that are not exposure-tagged models.
    """
    match = model_name_re.fullmatch(Path(path).name)
    if match is None:
        return None
    seconds = float(match.group(1))
    return seconds, 1000.0 * seconds, match.group(2)


def exposure_matches(exposure, selected_exposures):
    """An empty selection means "no filter", so everything matches."""
    if not selected_exposures:
        return True
    return any(np.isclose(exposure, selected) for selected in selected_exposures)


def method_matches(method, selected_methods):
    if not selected_methods:
        return True
    return method in set(selected_methods)


def find_h5_for_exposure(data_dir, exposure):
    """The dataset for an exposure in seconds, matched numerically.

    The exact name is tried first; the scan after it is what catches datasets
    written with a different float repr (0.010 vs 0.01).
    """
    data_dir = Path(data_dir)

    expected = data_dir / f"{dataset_stem(exposure)}.h5"
    if expected.exists():
        return expected

    for path in sorted(data_dir.glob(f"{dataset_prefix}_*.h5")):
        match = h5_name_re.fullmatch(path.name)
        if match is not None and np.isclose(float(match.group(1)), exposure):
            return path

    raise FileNotFoundError(
        f"could not find dataset for exposure {exposure:g} s in {data_dir}"
    )


def resolve_h5_path(model, data_dir, exposure):
    """The dataset a model was trained on, else the one in data_dir with its exposure in seconds."""
    h5_path = model.get("h5_path")
    if h5_path is not None and Path(h5_path).exists():
        return Path(h5_path)
    return find_h5_for_exposure(data_dir, exposure)


splits_method_families = (
    "cnn_array",
    "cnn_site",
    "lightweight_cnn_site",
    "mf_site",
    "mf_array_gaussian",
    "mf_array",
)


def fold_splits_method(method):
    """'mf_site_10_splits' -> 'mf_site'; every other method name comes back unchanged."""
    if method.endswith("_splits"):
        for family in splits_method_families:
            if method.startswith(f"{family}_"):
                return family
    return method


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


def fill_rate(path):
    """Fill percentage of a 'fill_<n>' directory, or of a dataset inside one."""
    path = Path(path)
    directory = path if path.name.startswith("fill_") else path.parent
    return int(directory.name.split("_")[1])

def sync_device(device):
    device = str(device)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def extract_patches(images, sites, s):
    """
    images: (B, H, W)
    sites:  (Q, 2) or (B, Q, 2), where x is column and y is row

    returns:
        X: (B, Q, s*s)

    No interpolation: sites are rounded to nearest integer pixel.
    """
    batch = images.shape[0]
    device = images.device
    sites = sites.to(device)

    half = s // 2
    pad = half + 1

    padded = F.pad(images, pad=(pad, pad, pad, pad), mode="constant", value=0.0)
    offsets = torch.arange(s, device=device, dtype=torch.long) - half

    if sites.ndim == 2:
        n_qubits = sites.shape[0]
        center_x = torch.round(sites[:, 0]).long() + pad
        center_y = torch.round(sites[:, 1]).long() + pad

        patch_x = center_x[:, None] + offsets[None, :]
        patch_y = center_y[:, None] + offsets[None, :]

        patches = padded[:, patch_y[:, :, None], patch_x[:, None, :]]
        return patches.reshape(batch, n_qubits, s * s)

    if sites.ndim == 3:
        if sites.shape[0] != batch:
            raise ValueError(
                f"per-image sites batch size {sites.shape[0]} does not match images batch size {batch}"
            )
        n_qubits = sites.shape[1]
        center_x = torch.round(sites[:, :, 0]).long() + pad
        center_y = torch.round(sites[:, :, 1]).long() + pad

        patch_x = center_x[:, :, None] + offsets[None, None, :]
        patch_y = center_y[:, :, None] + offsets[None, None, :]
        batch_index = torch.arange(batch, device=device)[:, None, None, None]

        patches = padded[batch_index, patch_y[:, :, :, None], patch_x[:, :, None, :]]
        return patches.reshape(batch, n_qubits, s * s)

    raise ValueError(f"sites must have shape (Q, 2) or (B, Q, 2), got {tuple(sites.shape)}")


mf_array_neighbor_offsets = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def grid_side_from_n_qubits(n_qubits):
    side = int(np.sqrt(n_qubits))
    if side * side != n_qubits:
        raise ValueError(f"MF-array local neighbors require a square grid, got {n_qubits} sites")
    return side


def local_neighbor_means(patch_means, q, grid_side):
    row = q // grid_side
    col = q % grid_side
    zero = patch_means.new_zeros((patch_means.shape[0], 1))
    values = []

    for drow, dcol in mf_array_neighbor_offsets:
        neighbor_row = row + drow
        neighbor_col = col + dcol

        if (
            0 <= neighbor_row < grid_side
            and 0 <= neighbor_col < grid_side
        ):
            neighbor_q = neighbor_row * grid_side + neighbor_col
            values.append(patch_means[:, neighbor_q:neighbor_q + 1])
        else:
            values.append(zero)

    return torch.cat(values, dim=1)


def gaussian_weighted_patch_values(X, s, sigmas, normalize=False):
    sigmas = torch.as_tensor(sigmas, dtype=X.dtype, device=X.device)
    if sigmas.ndim != 1 or sigmas.shape[0] != X.shape[1]:
        raise ValueError(
            f"expected one Gaussian sigma per site, got shape {tuple(sigmas.shape)} "
            f"for {X.shape[1]} sites"
        )

    sigmas = torch.clamp(sigmas, min=1e-6)
    offsets = torch.arange(s, dtype=X.dtype, device=X.device) - (s - 1) / 2
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    radius_squared = (xx ** 2 + yy ** 2).reshape(1, s * s)
    kernels = torch.exp(-radius_squared / (2 * sigmas[:, None] ** 2))

    if normalize:
        kernels = kernels / torch.clamp(kernels.sum(dim=1, keepdim=True), min=1e-12)

    return torch.einsum("bqd,qd->bq", X, kernels)


def mf_array_site_context_values(X, s, mf_array_gaussian=False, mf_array_sigmas=None, mf_array_gaussian_normalize=False):
    if mf_array_gaussian:
        if mf_array_sigmas is None:
            raise ValueError("mf_array_gaussian=True requires mf_array_sigmas")
        return gaussian_weighted_patch_values(
            X,
            s=s,
            sigmas=mf_array_sigmas,
            normalize=mf_array_gaussian_normalize,
        )

    return X.mean(dim=2)


def local_neighbor_values(site_values, q, grid_side):
    row = q // grid_side
    col = q % grid_side
    zero = site_values.new_zeros((site_values.shape[0], 1))
    values = []

    for drow, dcol in mf_array_neighbor_offsets:
        neighbor_row = row + drow
        neighbor_col = col + dcol

        if (
            0 <= neighbor_row < grid_side
            and 0 <= neighbor_col < grid_side
        ):
            neighbor_q = neighbor_row * grid_side + neighbor_col
            values.append(site_values[:, neighbor_q:neighbor_q + 1])
        else:
            values.append(zero)

    return torch.cat(values, dim=1)


def indexed_neighbor_values(site_values, q, neighbor_indices):
    indices = neighbor_indices[q]
    valid = indices >= 0
    clipped = torch.clamp(indices, min=0)
    values = site_values.index_select(dim=1, index=clipped)

    if not bool(torch.all(valid).item()):
        values = values * valid.to(dtype=values.dtype)[None, :]

    return values


def learn_W(dataloader,s, mf_array=False, alpha=0.0, device="cpu", mf_array_gaussian=False, mf_array_sigmas=None,
            mf_array_gaussian_normalize=False, mf_array_neighbor_indices=None):
    sites = dataloader.dataset.sites
    n_qubits = sites.shape[0]
    patch_dim = s * s
    feature_dim = patch_dim + 1
    neighbor_indices = None
    if mf_array:
        if mf_array_neighbor_indices is not None:
            neighbor_indices = torch.as_tensor(
                mf_array_neighbor_indices,
                dtype=torch.long,
                device=device,
            )
            if neighbor_indices.shape[0] != n_qubits:
                raise ValueError(
                    f"expected neighbor indices for {n_qubits} sites, "
                    f"got shape {tuple(neighbor_indices.shape)}"
                )
            use_global_mf_array = False
            feature_dim = patch_dim + int(neighbor_indices.shape[1]) + 1
        else:
            use_global_mf_array = n_qubits == 9
            if use_global_mf_array:
                feature_dim = patch_dim + (n_qubits - 1) + 1
            else:
                grid_side = grid_side_from_n_qubits(n_qubits)
                feature_dim = patch_dim + len(mf_array_neighbor_offsets) + 1

        if use_global_mf_array:
            feature_dim = patch_dim + (n_qubits - 1) + 1

    XtX = torch.zeros((n_qubits, feature_dim, feature_dim), device=device)
    Xty = torch.zeros((n_qubits, feature_dim, 1), device=device)

    for images, labels, sites in dataloader:
        images = images.to(device).float()
        labels = labels.to(device).float()
        sites = sites[0].to(device)

        X = extract_patches(images, sites, s)
        if mf_array:
            site_context_values = mf_array_site_context_values(
                X,
                s=s,
                mf_array_gaussian=mf_array_gaussian,
                mf_array_sigmas=mf_array_sigmas,
                mf_array_gaussian_normalize=mf_array_gaussian_normalize,
            )
            batch = X.shape[0]
            for q in range(n_qubits):
                if neighbor_indices is not None:
                    neighbor_values = indexed_neighbor_values(
                        site_context_values,
                        q,
                        neighbor_indices,
                    )
                elif use_global_mf_array:
                    neighbor_values = torch.cat(
                        [
                            site_context_values[:, :q],
                            site_context_values[:, q + 1:],
                        ],
                        dim=1,
                    )
                else:
                    neighbor_values = local_neighbor_values(
                        site_context_values,
                        q,
                        grid_side,
                    )
                ones = torch.ones((batch, 1), device=device)
                Xq = torch.cat([X[:, q, :], neighbor_values, ones], dim=1)
                yq = labels[:, q]

                XtX[q] += Xq.T @ Xq
                Xty[q] += (Xq.T @ yq[:, None])
        else:
            ones = torch.ones((X.shape[0], n_qubits, 1), device=device)
            X_aug = torch.cat([X, ones], dim=2)

            XtX += torch.einsum("bqi,bqj->qij", X_aug, X_aug)
            Xty += torch.einsum("bqi,bq->qi", X_aug, labels).unsqueeze(-1)

    I = torch.eye(feature_dim, device=device).unsqueeze(0)
    if mf_array:
        I[:, -1, -1] = 0.0
    W = torch.linalg.solve(XtX + alpha * I, Xty)
    return W


@torch.no_grad()
def calc_score(dataloader, s, W, mf_array=False, device="cpu", mf_array_gaussian=False, mf_array_sigmas=None,
               mf_array_gaussian_normalize=False, mf_array_neighbor_indices=None):
    all_scores = []
    all_labels = []

    W = W.to(device)
    n_qubits = W.shape[0]
    neighbor_indices = None
    if mf_array:
        patch_dim = s * s
        global_feature_dim = patch_dim + (n_qubits - 1) + 1
        local_feature_dim = patch_dim + len(mf_array_neighbor_offsets) + 1
        feature_dim = int(W.shape[1])

        if mf_array_neighbor_indices is not None:
            neighbor_indices = torch.as_tensor(
                mf_array_neighbor_indices,
                dtype=torch.long,
                device=device,
            )
            if neighbor_indices.shape[0] != n_qubits:
                raise ValueError(
                    f"expected neighbor indices for {n_qubits} sites, "
                    f"got shape {tuple(neighbor_indices.shape)}"
                )
            expected_feature_dim = patch_dim + int(neighbor_indices.shape[1]) + 1
            if feature_dim != expected_feature_dim:
                raise ValueError(
                    f"MF-array W feature dimension {feature_dim} does not match "
                    f"{expected_feature_dim} from custom neighbors and s={s}"
                )
            use_global_mf_array = False
            use_local_mf_array = False
        else:
            use_global_mf_array = n_qubits == 9 or feature_dim == global_feature_dim
            use_local_mf_array = not use_global_mf_array and feature_dim == local_feature_dim

        if neighbor_indices is None and use_local_mf_array:
            grid_side = grid_side_from_n_qubits(n_qubits)
        elif neighbor_indices is None and not use_global_mf_array:
            raise ValueError(
                f"Cannot match MF-array W feature dimension {feature_dim} "
                f"for {n_qubits} sites and s={s}"
            )

    for images, labels, sites in dataloader:
        images = images.to(device).float()
        labels = labels.float()
        sites = sites[0].to(device)

        X = extract_patches(images, sites, s)
        batch = X.shape[0]

        if mf_array:
            site_context_values = mf_array_site_context_values(
                X,
                s=s,
                mf_array_gaussian=mf_array_gaussian,
                mf_array_sigmas=mf_array_sigmas,
                mf_array_gaussian_normalize=mf_array_gaussian_normalize,
            )
            scores = torch.zeros((batch, n_qubits), device=device)

            for q in range(n_qubits):
                if neighbor_indices is not None:
                    neighbor_values = indexed_neighbor_values(
                        site_context_values,
                        q,
                        neighbor_indices,
                    )
                elif use_global_mf_array:
                    neighbor_values = torch.cat(
                        [
                            site_context_values[:, :q],
                            site_context_values[:, q + 1:],
                        ],
                        dim=1,
                    )
                else:
                    neighbor_values = local_neighbor_values(
                        site_context_values,
                        q,
                        grid_side,
                    )
                ones = torch.ones((batch, 1), device=device)
                Xq = torch.cat([X[:, q, :], neighbor_values, ones], dim=1)
                scores[:, q] = (Xq @ W[q]).squeeze(-1)
        else:
            ones = torch.ones((batch, n_qubits, 1), device=device)
            X_aug = torch.cat([X, ones], dim=2)
            scores = torch.einsum("bqd,qdk->bqk", X_aug, W).squeeze(-1)

        all_scores.append(scores.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


def compute_preprocessing(h5_path, train_idx, image_key="images", chunk_size=64):
    if len(train_idx) == 0:
        raise ValueError(f"Cannot compute preprocessing for {h5_path}: train_idx is empty")

    train_idx = np.sort(np.asarray(train_idx, dtype=np.int64))

    with h5py.File(h5_path, "r") as f:
        images = f[image_key]
        sum_image = np.zeros(images.shape[1:], dtype=np.float64)

        for start in range(0, len(train_idx), chunk_size):
            chunk_idx = train_idx[start:start + chunk_size]
            image_chunk = images[chunk_idx].astype(np.float32)
            sum_image += image_chunk.sum(axis=0, dtype=np.float64)

        mean_image = (sum_image / len(train_idx)).astype(np.float32)

        scale = 0.0
        for start in range(0, len(train_idx), chunk_size):
            chunk_idx = train_idx[start:start + chunk_size]
            image_chunk = images[chunk_idx].astype(np.float32)
            scale = max(scale, float(np.max(np.abs(image_chunk - mean_image))))

    if scale == 0.0:
        scale = 1.0

    return mean_image, scale

class H5Dataset(Dataset):
    """
    Returns:
        images: (H, W)
        labels: (n_sites,)
        sites:  (n_sites, 2)
    """
    def __init__(
        self,
        h5_path,
        indices=None,
        image_key="images",
        label_key="labels",
        site_key="sites",
        mean_image=None,
        scale=None,
    ):
        self.h5_path = h5_path
        self.image_key = image_key
        self.label_key = label_key
        self.site_key = site_key
        self.mean_image = None if mean_image is None else np.asarray(mean_image, dtype=np.float32)
        self.scale = None if scale is None else float(scale)
        self._h5 = None

        with h5py.File(h5_path, "r") as f:
            self.n = f[image_key].shape[0]
            site_shape = tuple(f[site_key].shape)
            if len(site_shape) == 2 and site_shape[1] == 2:
                self.sites_are_per_image = False
                self.n_sites = int(site_shape[0])
                self.sites = torch.from_numpy(f[site_key][:]).float()
            elif len(site_shape) == 3 and site_shape[0] == self.n and site_shape[2] == 2:
                self.sites_are_per_image = True
                self.n_sites = int(site_shape[1])
                self.sites = torch.from_numpy(f[site_key][0]).float()
            else:
                raise ValueError(
                    "sites must have shape (n_sites, 2) or (n_images, n_sites, 2), "
                    f"got {site_shape}"
                )

        if indices is None:
            self.indices = list(range(self.n))
        else:
            self.indices = list(indices)

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
        f = self._file()
        i = self.indices[idx]

        image_np = f[self.image_key][i].astype(np.float32)
        if self.mean_image is not None:
            image_np = image_np - self.mean_image
        if self.scale is not None:
            image_np = image_np / self.scale

        image = torch.from_numpy(image_np).float()
        label = torch.from_numpy(f[self.label_key][i]).float()
        if self.sites_are_per_image:
            sites = torch.from_numpy(f[self.site_key][i].astype(np.float32)).float()
        else:
            sites = self.sites

        return image, label, sites
