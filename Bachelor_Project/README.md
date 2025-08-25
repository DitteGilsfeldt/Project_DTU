# BSc project title (15 ECTS):

Federated learning for tackling data heterogeneity in healthcare applications

### Main Focus
Investigating PFL improvements of model robustness and stability from FL for cross-silo systems under highly heterogeneous medical environments.

___

# About the repository

This repository contains the notebooks used in the BSc project:

"Federated learning for tackling data heterogeneity in healthcare applications".

The project researches the use of Federated Learning (FL) and Personalized Federated Learning (PFL) to improve model robustness and stability when dealing with highly heterogeneous medical data set up as two distinct clients: each a hospital with its own patient data.

The notebooks here contain:

- FL pipeline on healthcare data (2 clients)

- PFL pipeline on healthcare data (2 clients)

- FL pipeline on MNIST (control)

- PFL pipeline on MNIST (control)

- Centralized MLP model for healthcare data

- Centralized MLP model for MNIST

- Centralized LSTM + MLP model for healthcare data

See detailed structure of repository below.

___

# Repository structure

```
├── Codes_and_notebooks/
│   ├── FL_pipeline.ipynb
│   ├── MNIST_FL.ipynb
│   ├── MNIST_PFL.ipynb
│   ├── PFL_pipeline.ipynb
│   ├── stat_tests.ipynb
│   ├── preprocessed_data_showcase.ipynb
│
│   ├── Centralized_models/
│   │   ├── mnist_centralized.ipynb
│   │   ├── lstm_mimic_eicu_training.ipynb
│   │   └── mlp_mimic_eicu_training.ipynb
│
│   ├── MIMIC_data_handler/
│   │   ├── dataloader_mimic.ipynb
│   │   └── preprocessor_mimic_icustay.ipynb
│
│   ├── eICU_data_handler/
│   │   ├── dataloader_eicu.ipynb
│   │   └── preprocessing_eicu.ipynb
│
│
├── .gitignore
├── LICENSE
├── README.md

```

---

# Platform notes

This project was developed on macOS using Apple Silicon (M1 and M2 chips). All PyTorch operations default to the **MPS** backend.

If running the notebooks on a different platform (e.g., Linux or Windows), adjust the device configuration in the code (e.g. `'mps'`, `'cuda'`, `'cpu'`) accordingly.

---

# Data usage and access restrictions

This project uses data from:

- **MIMIC-III**: https://physionet.org/content/mimiciii/1.4/
- **eICU Collaborative Research Database**: https://physionet.org/content/eicu-crd/2.0/

These datasets are protected under data use agreements. Due to licensing restrictions and dataset size, **no raw or preprocessed data is included** in this repository.

To use the code or reproduce results, you must:

1. Create a PhysioNet account.
2. Complete the required human subjects research training (CITI program).
3. Agree to the data use agreements (DUAs) for both MIMIC-III and eICU.

---

## Data setup instructions

If you have access to the datasets, follow these steps:

1. In the root directory of the repository (where `README.md` and `Codes_and_notebooks` are), create a folder named `Datasets` with two subfolders:

```
Datasets/
├── mimic_iii_data/ # place raw MIMIC-III CSV files here
└── eicu_data/      # place raw eICU CSV files here
```

2. Run the following notebooks to generate aggregated raw CSVs:

- `dataloader_MIMIC.ipynb`
- `dataloader_eICU.ipynb`

These will output processed CSVs into:

Codes_and_notebooks/MIMIC_data_handler/

Codes_and_notebooks/eICU_data_handler/

___


3. Then run the preprocessing notebooks:

- `preprocessor_mimic_icustay.ipynb`
- `preprocessing_eicu.ipynb`

These convert data into final NumPy arrays for model input. Once this step is done, all experiment notebooks can be run directly.

**Note**: All large data files are automatically ignored via `.gitignore`.

---

# Privacy and ethics

All data handling follows the ethical guidelines and constraints of the PhysioNet DUAs. No identifiable patient data is ever exposed or transferred across silos. All ID and sensitive columns are excluded from processing.

---

# Requirements

See `requirements.txt` for a full list of required Python packages.

Main dependencies include:

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
statsmodels
jupyterlab
ipykernel
ipywidgets
torch==2.5.1
torchaudio
torchvision
lightgbm
```

# Installation

To clone the repository, use:

```bash
git clone https://github.com/DitteGilsfeldt/Federated_Learning_for_tackling_data_heterogeneity_in_healthcare_applications
```

To set up the environment, run:

```bash
pip install -r requirements.txt
```

