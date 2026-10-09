# FPGA-accelerated neutral-atom detection

Train TinyCNN, choose its fixed-point precisions, generate the VHDL data, and
run inference on an AMD Alveo U55C. The validated design uses one CNN to classify
nine sites in a 3×3 grid at 400 MHz. Each site has a 10×10 crop and eight neighbor
features; the convolutional layers have 4, 8, and 16 channels.

The instructions below use `complete_workflow/rtl`. The separate parallel
TinyCNN design without neighbor features is in `complete_workflow/rtl_parallel`.

## 1. Prepare the environment and dataset

Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch matplotlib tqdm
python -m pip install -r complete_workflow/rtl/requirements.txt
```

Training requires the HDF5 dataset, supplied separately, at
`dataset_3x3/qubit_dataset_sim_0.01.h5` for the 10 ms exposure used by this build.
Use a PyTorch installation compatible with your GPU, or train on the CPU.
RTL simulation also requires GHDL.

FPGA implementation requires Linux, Vivado and Vitis 2024.2, and the U55C platform
`xilinx_u55c_gen3x16_xdma_3_202210_1`. Hardware execution requires an Alveo U55C
with the matching platform and XRT installed, including its `pyxrt` bindings.

The generated VHDL, ROM files, and test vectors are already included. To build
those sources without retraining, start at step 5.

## 2. Train TinyCNN

In `lightweight_cnn_site.py`, select the model and exposure used by the single-CNN
hardware:

```python
selected_methods = (lightweight_cnn_site_neighbor_method,)
selected_exposures = (0.01,)
device = "cuda:0"  # Use "cpu" if no GPU is available.
```

Keep `crop_size = 10` and the channels `(4, 8, 16)` in `constants.py`. Then run:

```bash
python lightweight_cnn_site.py
```

Training saves the weights and preprocessing parameters in
`models_3x3/qubit_dataset_sim_0.01_lightweight_cnn_site_neighbor.joblib`.
Learning curves and the crop cache are saved under `output_3x3/`.

## 3. Profile and choose fixed-point precisions

The pipeline is configured for the same neighbor model and 10 ms exposure.
Run its stages in order:

```bash
python complete_workflow/run_pipeline.py --only profile
python complete_workflow/run_pipeline.py --only sensitivity
python complete_workflow/run_pipeline.py --only optimize
```

Profiling measures the ranges of weights and activations. Sensitivity evaluates
the effect of reducing precision. Optimization selects widths using the integer
reference model and the configured accuracy tolerance.

Results are saved in `complete_workflow/results/inj/`, including
`precisions.csv`, `sensitivity.csv`, and `best_0.01s.json`.
To run all three stages with one command:

```bash
python complete_workflow/run_pipeline.py
```

## 4. Generate and verify the RTL data

```bash
python complete_workflow/rtl/gen/generate_all.py
python complete_workflow/rtl/xrt/sim/test_full_runner.py
python -m unittest discover -s complete_workflow/rtl/xrt/host -p 'test_*.py'
```

The generator reads the trained model and selected precisions, then writes the
packages, weight and bias ROM files, preprocessing tables, and reference vectors.
It replaces the existing generated files. The full simulation compares eight
frames, containing 72 site predictions, against the fixed-point reference and
checks the hardware counters.

## 5. Build the Alveo XCLBIN

On the FPGA build machine:

```bash
cd complete_workflow/rtl
bash impl/build_alveo_400.sh \
  /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
```

The script packages the RTL kernel, runs synthesis and implementation with the
Alveo shell, and verifies the 400 MHz runtime clock and routed timing. The output
is `impl/alveo_400_single_cnn/neutral_atom_detector.xclbin`, with reports in the
same build directory. For another build, pass a fresh output directory as the
second argument; the script prevents overwriting an existing build.

## 6. Run predictions and read the cycle counts

From `complete_workflow/rtl`, on the machine with the Alveo card:

```bash
source /opt/xilinx/xrt/setup.sh
python3 xrt/host/run_detector.py \
  impl/alveo_400_single_cnn/neutral_atom_detector.xclbin \
  tb/vectors/raw_frame.txt \
  --frame-index 0 --device 0 --freq-mhz 400
```

The host prints both logits and the occupied/empty decision for each site,
followed by the FPGA cycle counts and their latency at 400 MHz. It also reports
the transfer-to-result interval measured by native XRT profiling.

The submitted implementation was verified on hardware with all 72 predictions
matching the reference:

| FPGA measurement | Cycles | Latency at 400 MHz |
|---|---:|---:|
| Preprocessing, all nine sites | 802 | 2.005 µs |
| CNN inference, one site | 858 | 2.145 µs |
| First preprocessing pixel to the last prediction | 8559 | 21.3975 µs |

The wrapper fetches the frame from HBM before feeding preprocessing at one pixel
per clock. These hardware counters exclude that initial fetch, host transfers,
kernel launch, and result writeback. The separate XRT measurement includes the
input transfer to HBM through result readback. Convert cycles to microseconds
with `cycles / 400`.

To validate all eight supplied frames and save the measurements:

```bash
python3 xrt/host/validate_hardware.py \
  impl/alveo_400_single_cnn/neutral_atom_detector.xclbin \
  --output impl/alveo_400_single_cnn/hardware
```

Final reports and hardware measurements for the submitted run are in
`report/data/fpga_runs/2026-10-08/single_cnn_400/`. Build products, datasets,
simulation caches, and private Markdown notes are excluded from Git.
