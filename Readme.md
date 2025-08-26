# Knowledge-data fusion dominated vehicle platoon dynamics modeling and analysis: A physics-encoded deep learning approach (PeMTFLN)

[![Journal](https://img.shields.io/badge/Information%20Fusion-Accepted-blue.svg)](https://www.sciencedirect.com/journal/information-fusion)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation for the paper: **"Knowledge-data fusion dominated vehicle platoon dynamics modeling and analysis: A physics-encoded deep learning approach"**, accepted by the journal *Information Fusion*.

[**[Paper Link]**](https://www.sciencedirect.com/science/article/pii/S1566253525006943?via%3Dihub#fig1) 

---

## Abstract

Nonlinear platoon dynamics modeling is essential for predicting and optimizing vehicle interactions. However, existing approaches often struggle to capture platoon-scale interaction features while maintaining both high accuracy and physical analyzability. To address these challenges, this paper introduces a novel **Physics-encoded Deep Learning Network (PeMTFLN)** for modeling nonlinear vehicle platoon dynamics. The framework features an **Analyzable Parameters Encoded Computational Graph (APeCG)** to ensure physically consistent and stable platoon responses, and a **Multi-scale Trajectory Feature Learning Network (MTFLN)** to learn driving patterns from trajectory data. Trained on the HIGHSIM dataset, PeMTFLN outperforms baseline models in prediction accuracy and successfully replicates real-world platoon stability and safety characteristics, demonstrating a superior balance of predictive accuracy and physical interpretability.

---

## Framework

The overall architecture of our proposed **Parameters Encoder Multi-scale Trajectory Feature Learning Network (PeMTFLN)** integrates a multi-scale feature learning network (MTFLN) to learn driving patterns from data and an analyzable parameters encoded computational graph (APeCG) to ensure physical consistency and stability.

![Framework Diagram](https://github.com/HaoLyu666/PeMTFLN/blob/main/figures/framework.png)
> **Figure**: The architecture of PeMTFLN under the PeDL Framework, consisting of (a) Vehicle-level Feature Learning, (b) Platoon-level Feature Learning, (c) Non-Autoregressive Parameters Decoder, and (d) Analyzable Parameters Encoded Computational Graph.

---

## Main Contributions

* **A novel vehicle platoon dynamics modeling framework (PeDL) with both interpretability and high accuracy is proposed.** It focuses on directly learning and encoding the physical parameters of a generalized platoon model and can be scaled to model platoons with varying numbers of vehicles.
* **A multi-scale trajectory feature learning network (MTFLN) is designed** to facilitate the end-to-end learning of the parameters required by PeDL, capturing features at both vehicle and platoon levels.
* **The model's effectiveness is validated on the real-world HIGH-SIM dataset.** PeMTFLN accurately reproduces platoon following behavior, including stability and safety evolution, and shows robust generalization on the NGSIM dataset.

---

## Environment Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/HaoLyu666/PeMTFLN.git](https://github.com/HaoLyu666/PeMTFLN.git)
cd PeMTFLN
```

**2. Create a Conda Environment (Recommended):**
```bash
conda create -n pemtfln python=3.8
conda activate pemtfln
```

**3. Install Dependencies:**
This project relies on several Python libraries. You can install them via pip.
```bash
pip install torch numpy tqdm pandas matplotlib
# Install PyTorch according to your CUDA version
# e.g., for CUDA 11.3:
# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)
```
* **OS**: Ubuntu 20.04
* **CUDA Version**: 11.4
* **PyTorch Version**: 1.12.1+

---

## Data Preparation

* **Dataset**: The primary dataset is extracted from **HIGH-SIM**, containing platoon trajectories of seven vehicles. Generalization experiments are conducted on the **NGSIM** dataset.
* **Data Format**: The preprocessed data is stored in a `.npz` file (`platoons_data_split.npz`), which contains training, validation, and test sets. The data loading logic can be found in `loader2.py`.
* **Data Structure**: Each data sample consists of `hist` (historical trajectory features like gap, speed, etc.), `fut` (future ground truth gap and speed), and `nextv` (the lead vehicle's future velocity). This structure is defined in `loader2.py`.

---

## Usage

### Configuration

All hyperparameters for the model and training process are defined in `config.py`. You can modify this file to adjust settings like `batch_size`, `learning_rate`, `in_length`, `out_length`, etc.

### Model Training

Run the following script to start training the model. Model checkpoints will be saved in the `checkponint/` directory, and results will be logged in the `result/` directory. The training process is detailed in `train_highsim.py`.
```bash
python train_highsim.py
```

### Model Evaluation

After training, use the `evaluate_highsim.py` script to evaluate the model's performance on the test set. You need to specify the model epoch to load by modifying the `names` variable within the script.
```bash
# In evaluate_highsim.py, modify the 'names' variable
# names = '20' # Loads the model from the 20th epoch

python evaluate_highsim.py
```

---

## Results

### Quantitative Results

PeMTFLN was compared with several baseline models on the HIGH-SIM dataset. As shown in the table below, our model achieves state-of-the-art performance in both velocity and gap prediction across various metrics.

| Model | Velocity RMSE (m/s) Avg | Gap RMSE (m) Avg | Velocity MAPE (%) Avg | Gap MAPE (%) Avg |
|:--- |:---:|:---:|:---:|:---:|
| PerIDM | 1.705 | 2.034 | 9.95 | 8.61 |
| PerACC | 1.561 | 1.398 | 10.63 | 5.24 |
| KoopmanNet | 0.684 | 0.983 | 4.91 | 3.36 |
| Seq2Seq | 0.472 | 0.922 | 3.42 | 3.21 |
| Transformer | 0.476 | 0.797 | 3.26 | 2.81 |
| PeLSTM | 0.492 | 0.689 | 3.48 | 2.12 |
| PeTransformer | 0.479 | 0.666 | 3.25 | 1.99 |
| **PeMTFLN (Ours)** | **0.469** | **0.643** | **3.09** | **1.91** |

### Qualitative Results

**Trajectory Reproduction**: PeMTFLN accurately reproduces platoon trajectories under various driving scenarios, including continuous acceleration, oscillation, and deceleration.
![Trajectory Reproduction](https://github.com/HaoLyu666/PeMTFLN/blob/main/figures/accuracylocalvis.png)

**Stability Analysis**: The model successfully replicates the stability evolution of real-world platoons, which validates its capability for physical analysis.
![Stability Analysis](https://github.com/HaoLyu666/PeMTFLN/blob/main/figures/stability_analysis.png)

**Safety Analysis**: The model's predictions align closely with ground-truth data in terms of surrogate safety measure distributions, such as Post-Encroachment Time (PET) and Safe Stopping Distance Difference (SSDD).
![Safety Analysis](https://github.com/HaoLyu666/PeMTFLN/blob/main/figures/safety_analysis.png)

---

## Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{lyu2025knowledge,
title = {Knowledge-data fusion dominated vehicle platoon dynamics modeling and analysis: A physics-encoded deep learning approach},
journal = {Information Fusion},
pages = {103622},
year = {2025},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2025.103622},
}
```

---

## Contact

If you have any questions, please feel free to email the authors:

- **Hao Lyu**: `lyu_hao@seu.edu.cn`
- **Yanyong Guo** (Corresponding Author): `guoyanyong@seu.edu.cn`

---

## License

This project is released under the [MIT License](LICENSE).





 




