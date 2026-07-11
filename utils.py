import torch
import h5py
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

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

# def extract_patches(images, sites, s):
#     """
#     images: (B, H, W)
#     sites:  (Q, 2), where sites[:, 0] = x, sites[:, 1] = y in image pixel coordinates
#
#     returns:
#         X: (B, Q, s*s)
#     """
#     Batch, H, W = images.shape
#     Qubits = sites.shape[0]
#     device = images.device
#
#     x = sites[:, 0].to(device)
#     y = sites[:, 1].to(device)
#
#     # Use symmetric offsets around the subpixel site center.
#     offsets = torch.arange(s, device=device).float() - (s - 1) / 2
#     dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
#
#     sample_x = x[:, None, None] + dx[None, :, :]  # (Q, s, s)
#     sample_y = y[:, None, None] + dy[None, :, :]  # (Q, s, s)
#
#     # Convert pixel coordinates to grid_sample coordinates in [-1, 1].
#     grid_x = 2.0 * sample_x / (W - 1) - 1.0
#     grid_y = 2.0 * sample_y / (H - 1) - 1.0
#
#     grid = torch.stack([grid_x, grid_y], dim=-1)  # (Q, s, s, 2)
#
#     # Repeat the same site grid for every image in the batch.
#     grid = grid[None].expand(Batch, Qubits, s, s, 2).reshape(Batch, Qubits * s, s, 2)
#
#     img = images[:, None, :, :]  # (B, 1, H, W)
#
#     patches = F.grid_sample(
#         img,
#         grid,
#         mode="bilinear",
#         padding_mode="zeros",
#         align_corners=True,
#     )
#
#     patches = patches[:, 0]
#     patches = patches.reshape(Batch, Qubits, s, s)
#     return patches.reshape(Batch, Qubits, s * s)

def extract_patches(images, sites, s):
    """
    images: (B, H, W)
    sites:  (Q, 2), where sites[:, 0] = x, sites[:, 1] = y

    returns:
        X: (B, Q, s*s)

    No interpolation: sites are rounded to nearest integer pixel.
    """
    Batch, H, W = images.shape
    Qubits = sites.shape[0]
    device = images.device

    half = s // 2
    pad = half + 1

    padded = F.pad(images, pad=(pad, pad, pad, pad), mode="constant", value=0.0)

    center_x = torch.round(sites[:, 0].to(device)).long() + pad
    center_y = torch.round(sites[:, 1].to(device)).long() + pad

    offsets = torch.arange(s, device=device, dtype=torch.long) - half

    patch_x = center_x[:, None] + offsets[None, :]
    patch_y = center_y[:, None] + offsets[None, :]

    patches = padded[:, patch_y[:, :, None], patch_x[:, None, :]]
    return patches.reshape(Batch, Qubits, s * s)


def compute_preprocessing(h5_path, train_idx, image_key="images"):
    if len(train_idx) == 0:
        raise ValueError(f"Cannot compute preprocessing for {h5_path}: train_idx is empty")

    with h5py.File(h5_path, "r") as f:
        images = f[image_key]
        sum_image = np.zeros(images.shape[1:], dtype=np.float64)

        for i in train_idx:
            sum_image += images[int(i)]

        mean_image = (sum_image / len(train_idx)).astype(np.float32)

        scale = 0.0
        for i in train_idx:
            image = images[int(i)].astype(np.float32) - mean_image
            scale = max(scale, float(np.max(np.abs(image))))

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
            self.sites = torch.from_numpy(f[site_key][:]).float()

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

        return image, label, self.sites
