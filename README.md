# Bounding-Box Inference for Error-Aware Model-Based Reinforcement Learning

This repository is my attempt to reproduce the method and results in the paper:

> **Talvitie et al. (2024). Bounding-Box Inference for Error-Aware Model-Based Reinforcement Learning.**
> *Reinforcement Learning Journal, vol. 5, 2024, pp. 2440–2460.*

The code implements a selective planning method that uses bounding‐box inference to mitigate catastrophic planning in model‐based reinforcement learning (MBRL). In the experiments, we evaluate the approach on the [GoRight environment](https://github.com/cruz-lucas/goright) (see below), comparing variants such as Q-learning, Perfect, Expectation, Sampling, and different bounding-box inference (BBI) learned models (linear, tree, and neural).


## Overview

This repository replicates the experimental results from the paper by integrating a bounding-box inference procedure into a tabular Q-learning framework. In each training episode, the agent:
- Collects a real transition from the GoRight environment.
- Uses a predictive model to simulate additional rollout steps up to a specified horizon.
- Combines the multiple (real and simulated) TD targets via a softmin weighting over their uncertainties (computed via bounding-box inference) to update the Q–values.

The repository includes:
- **`src/train.py`** – The main training script, configurable via [gin-config](https://github.com/google/gin-config).
- **`src/bbi/agents/agent.py`** – The implementation of the learning agent with bounding-box inference.
- **`src/bbi/models/`** – Different model implementations (Expectation, Sampling, Perfect, and a base Model).

## Installation

### Prerequisites

- **Python 3.10+**
- **Docker** (optional, for containerized execution)

### 1. Clone the Repository

```bash
git clone https://github.com/cruz-lucas/bbi.git
cd bbi
```

### 2. Create a Virtual Environment & Install Dependencies

#### 1. Using **pip**:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. Or using **uv**:

```bash
uv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
uv pip install .
```

## Usage

### Running Training Experiments

The primary training script is `train.py`. You can run it directly via:

```bash
python src/main.py --config_file="goright_bbi"
```

The following command-line arguments are available:
- `--config_file`: Path (without extension) to the gin configuration file (located in `bbi/config/`).

## The GoRight Environment

The experiments in this repository use the GoRight environment – a simple RL environment that poses a challenging exploration problem. The environment (implemented separately) is available at:
[https://github.com/cruz-lucas/goright](https://github.com/cruz-lucas/goright)

For full details on the GoRight environment, please refer to its repository.

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{talvitie2024bounding,
    title={Bounding-Box Inference for Error-Aware Model-Based Reinforcement Learning},
    author={Talvitie, Erin J and Shao, Zilei and Li, Huiying and Hu, Jinghan and Boerma, Jacob and Zhao, Rory and Wang, Xintong},
    journal={Reinforcement Learning Journal},
    volume={5},
    pages={2440--2460},
    year={2024}
}
```


## Contributing

Contributions, improvements, and suggestions are welcome! Feel free to open an issue or submit a pull request.


## License

This project is licensed under the [MIT License](LICENSE).
