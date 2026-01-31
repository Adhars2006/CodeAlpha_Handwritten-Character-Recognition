# CodeAlpha_Handwritten-Character-Recognition

A PyTorch-based example for handwritten character recognition using the EMNIST balanced dataset. The repository contains a single-file pipeline (model.py) that downloads the dataset (via Kaggle), prepares the EMNIST images, defines a small CNN, trains the model, and reports evaluation metrics (accuracy, precision, recall, F1, ROC-AUC). The script also includes optional visualization of predictions.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration / Parameters](#configuration--parameters)
- [Training & Evaluation](#training--evaluation)
- [Extending the project](#extending-the-project)
- [Contributing](#contributing)
- [License](#license)

## Features

- Uses the EMNIST balanced split (47 classes: digits, uppercase, some lowercase).
- Simple CNN (2 conv layers + FC) implemented with PyTorch.
- Data preprocessing includes normalization, reshaping & rotation fix for EMNIST.
- Data augmentation (rotation, affine/shear) available in the pipeline.
- Prints common classification metrics and optional ROC-AUC for multi-class.

## Requirements

- Python 3.8+
- Packages (examples):
  - torch, torchvision
  - pandas, numpy
  - scikit-learn
  - matplotlib
  - kaggle (for dataset download)
  - (optional) jupyter, tqdm

You can create a requirements.txt with these packages or install directly:

pip install torch torchvision pandas numpy scikit-learn matplotlib kaggle

## Dataset

The script uses the EMNIST dataset from Kaggle. model.py attempts to download it with the Kaggle CLI:

kaggle datasets download -d crawford/emnist -p ./dataset --unzip

You must configure the Kaggle API credentials (kaggle.json) or download the dataset manually and place:
- ./dataset/emnist-balanced-train.csv
- ./dataset/emnist-balanced-test.csv

## Installation

1. Clone the repo:

git clone https://github.com/Adhars2006/CodeAlpha_Handwritten-Character-Recognition.git
cd CodeAlpha_Handwritten-Character-Recognition

2. Install dependencies:

pip install -r requirements.txt
# or
pip install torch torchvision pandas numpy scikit-learn matplotlib kaggle

3. Configure Kaggle:
- Place kaggle.json in ~/.kaggle/kaggle.json, or set KAGGLE_USERNAME & KAGGLE_KEY.

## Usage

To train the model:
```
python model.py
```

To run the UI:
```
streamlit run ui.py
```

Then open the provided URL in your browser to upload images and get predictions.

Run the training/evaluation pipeline:

python model.py

The script performs:
- Dataset download (via Kaggle) and loading
- Preprocessing and DataLoader creation
- Model definition, training for `num_epochs`, and evaluation
- Prints metrics and optionally visualizes some predictions

## Configuration / Parameters

Open model.py and edit variables such as:
- num_epochs (training epochs)
- batch_size (DataLoader)
- learning rate (inside optimizer)
- num_classes (should be 47 for EMNIST balanced)
- transform augmentation settings

## Training & Evaluation

- The script trains a small CNN and prints per-epoch loss plus final metrics:
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC (multi-class via one-vs-rest)
- Example visualization: `imshow` prints a few predicted vs actual samples.

## Extending the project

- For full-word/sentence handwriting recognition, combine CNN feature extraction with RNN + CTC (CRNN) and use IAM Handwriting datasets.
- Add checkpoint saving (torch.save) and model loading for inference.
- Add unit tests, a CLI wrapper, or a notebook for step-by-step demonstration.

## Contributing

Contributions welcome. Please open issues or PRs with clear descriptions and tests/examples where possible.

## License

Add an appropriate LICENSE file (e.g., MIT) if you want this repository to be openly licensed.