# MSTVI

This is the official implementation of paper "MSTVI: Multi-Scale Time-Variable Interaction for Multivariate Time Series Forecasting".

## Main Experiment
![image](image/image.png)

## Start

### 1. Create the environment

Python 3.10 is recommended. Create and activate an isolated virtual environment:

```bash
cd MSTVI
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch separately so that its CUDA build matches your hardware. For an NVIDIA GPU with CUDA 12.8 support (including Blackwell GPUs), run:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

For a CPU-only environment, use:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### 2. Prepare the data

Download the pre-processed forecasting datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing), extract them, and place the dataset directories under `./dataset`.

For example, ETTh1 should be located at:

```text
dataset/ETT-small/ETTh1.csv
```

### 3. Train and evaluate

The experiment scripts are provided under `./scripts/long_term_forecast/MSTVI/`. Each dataset has an independent script containing its complete configuration. Run a script from the repository root, for example:

```bash
bash scripts/long_term_forecast/MSTVI/traffic.sh
```

## Contact
If you have any questions or suggestions, feel free to contact our maintenance team:
- Quangao Liu (liuquangao@exeter.ac.uk)
- Ruiqi Li (liruiqi1@sia.cn)
- Maowei Jiang (jiangmaowei@sia.cn)

Or describe it in Issues.


## Citation

If you find this repo useful, please cite our paper
```
@article{liu2025mstvi,
  title={Mstvi: Multi-scale Time-Variable Interaction for multivariate time series forecasting},
  author={Liu, Quangao and Li, Ruiqi and Jiang, Maowei and Yang, Wei and Liang, Chen and Pang, Longlong and Zou, Zhuozhang},
  journal={Knowledge-Based Systems},
  pages={113551},
  year={2025},
  publisher={Elsevier}
}
```
## Acknowledgement

Our code is based on Time Series Library (TSLib)：https://github.com/thuml/Time-Series-Library
