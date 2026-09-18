# ASIC vs. GPU Data Center Energy & Performance Simulator

[![Purdue Expo Award](https://img.shields.io/badge/Award-Presentation%20with%20Distinction-gold.svg)](#research-context)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

An automated Python-based benchmarking tool designed to model, simulate, and compare Multiply-Accumulate (MAC) energy outputs, execution latency, and chip area utilization across domain-specific ASICs vs. enterprise GPUs running machine learning workloads.

> **Research Context:** This project was developed as part of an undergraduate research initiative at Purdue University studying data center hardware sustainability. The work earned a **Presentation with Distinction** at the Purdue Fall Undergraduate Research Expo.

---

## Key Features

- **Workload Tracing:** Extracts computational graphs and operator-level MAC counts from PyTorch models.
- **Hardware Architecture Parameterization:** Custom configurations for memory bandwidth, clock frequency, systolic array dimensions, and chip area.
- **Energy Estimation Model:** Estimates Joules-per-inference based on standard node energy scaling (e.g., SRAM read/write, 32-bit floating-point vs. INT8 MAC operations).
- **Automated Visualization:** Generates side-by-side efficiency curves and energy trade-off graphs using Matplotlib and Seaborn.

---

## Architecture Overview

```
               ┌──────────────────────────────┐
               │     Target PyTorch Model     │
               └──────────────┬───────────────┘
                              │
                      [ Model Parser ]
                              │
               ┌──────────────▼──────────────┐
               │    MAC Execution Graph      │
               └──────────────┬───────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
┌───────▼─────────────┐                   ┌─────────▼───────────┐
│ GPU Hardware Model  │                   │ ASIC Hardware Model │
│ (Parallel/Generic)  │                   │  (Systolic Array)   │
└───────┬─────────────┘                   └─────────┬───────────┘
        │                                           │
        └─────────────────────┬─────────────────────┘
                              │
                  [ Simulation Engine ]
                              │
               ┌──────────────▼──────────────┐
               │   Energy & Latency Metrics  │
               └─────────────────────────────┘
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- `pip` or `conda` package manager

### Standard Setup

```bash
# Clone the repository
git clone https://github.com/your-username/asic-vs-gpu-compiler.git
cd asic-vs-gpu-compiler

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Dependencies
```text
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
```

---

## Quickstart & Usage

To run a comparative energy simulation using a default Convolutional / Multi-Layer Perceptron workload:

```bash
python run_simulation.py --model dummy_ml --asic-config configs/asic_systolic_16x16.json --gpu-config configs/a100_spec.json
```

### Example Python API Usage

```python
from simulator import HardwareSimulator, LoadWorkload
from simulator.models import ASICConfig, GPUConfig

# Load custom hardware target definitions
asic_target = ASICConfig.from_json("configs/asic_systolic_16x16.json")
gpu_target = GPUConfig.from_json("configs/a100_spec.json")

# Trace PyTorch Model
workload = LoadWorkload.from_pytorch_model(my_torch_model)

# Instantiate and Run Simulator
sim = HardwareSimulator(workload=workload)
asic_results = sim.evaluate_target(asic_target)
gpu_results = sim.evaluate_target(gpu_target)

print(f"ASIC Energy per Inference: {asic_results.energy_mJ:.3f} mJ")
print(f"GPU Energy per Inference:  {gpu_results.energy_mJ:.3f} mJ")
print(f"Efficiency Gain:           {gpu_results.energy_mJ / asic_results.energy_mJ:.2f}x")
```

---

## Research Results Summary

Our simulated benchmarks across scaled deep learning workloads demonstrated that application-specific architectures (ASICs) optimized for tailored dataflows achieved up to a **>40% reduction in MAC energy consumption** over enterprise GPUs. This highlights the vital role of domain-specific hardware in reducing data center environmental impacts.

---

## Citation

If you use this simulator or build upon this research, please cite:

```bibtex
@misc{velmurugan2025asic,
  author = {Velmurugan, Sarathisamy},
  title = {ASICs vs GPUs for Data Center Sustainability: Energy Modeling Simulator},
  year = {2025},
  publisher = {Purdue University Undergraduate Research Expo}
}
