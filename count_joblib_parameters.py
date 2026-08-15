from pathlib import Path

import joblib
import torch


model_dir = Path("models_3x3")
patterns = (
    "*cnn_site*.joblib",
)


def state_dict_from_joblib(obj):
    if "state_dict" in obj:
        return obj["state_dict"]
    if "trained_model" in obj and "state_dict" in obj["trained_model"]:
        return obj["trained_model"]["state_dict"]
    return None


def count_state_dict_parameters(state_dict):
    total = 0
    by_tensor = []
    for name, value in state_dict.items():
        if torch.is_tensor(value):
            count = int(value.numel())
        else:
            count = int(torch.as_tensor(value).numel())
        total += count
        by_tensor.append((name, count, tuple(value.shape)))
    return total, by_tensor


def unique_paths(model_dir, patterns):
    paths = []
    seen = set()
    for pattern in patterns:
        for path in sorted(model_dir.glob(pattern)):
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def main(model_dir=model_dir, patterns=patterns, show_tensors=False):
    paths = unique_paths(model_dir, patterns)
    if not paths:
        raise FileNotFoundError(f"no joblib files matched {patterns} in {model_dir}")

    print(f"{'parameters':>12}  {'method':<42}  {'crop':>4}  {'model'}")
    print("-" * 115)

    for path in paths:
        obj = joblib.load(path)
        state_dict = state_dict_from_joblib(obj)
        if state_dict is None:
            print(f"{'—':>12}  {'no state_dict':<42}  {'—':>4}  {path.name}")
            continue

        total, by_tensor = count_state_dict_parameters(state_dict)
        method = str(obj.get("method", "unknown"))
        architecture = str(obj.get("architecture", "unknown"))
        crop_size = obj.get("crop_size", "—")

        print(f"{total:>12,d}  {method:<42}  {crop_size:>4}  {path.name}")
        if architecture != "unknown":
            print(f"{'':>12}  architecture: {architecture}")

        if show_tensors:
            for name, count, shape in by_tensor:
                print(f"{'':>12}    {count:>10,d}  {name:<45} {shape}")


if __name__ == "__main__":
    main()
