from cnn_site import make_site_loader
import constants
from lightweight_cnn_site import (
    LightweightCnnSiteNet,
    neighbor_table,
    prepare_site_batch,
)

channels = constants.tinycnn_channels
lightweight_cnn_site_neighbor_method = constants.tinycnn_neighbor_method
neighbor_feature_count = constants.tinycnn_neighbor_feature_count


def load_lightweight_cnn_site_model(best, device="cpu"):
    """What the FPGA runs, and what the profile is taken from."""
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
