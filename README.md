# MarbliX

**MarbliX** is a framework designed to integrate diverse biomedical data modalities, such as histopathology images and genomic data, into compact binary representations called _monograms_. This framework is adaptable to other modalities, enabling scalable and interpretable multimodal search, classification, and patient similarity analysis for a variety of applications in biomedical research.

---

## Project Structure

```
MarbliX/
├── data/
│   ├── embeddings/
│   │   ├── images/                  # Directory containing image embedding .npy files
│   │   └── sequences/              # Directory containing sequence embedding .npy files
│   └── metadata/
│       └── samples.csv             # CSV file listing sample metadata (file names + labels)
├── src/
│   └── marblix.py                  # Main script containing model training and feature extraction
├── requirements.txt
└── README.md
```

---

## Prerequisites

Ensure Python 3.8+ is installed. Then install the required packages:

```bash
pip install -r requirements.txt
```

---

## Data Format

Place your data under the `data/` folder in the following format:

### 1. `data/metadata/samples.csv`

This CSV file should have at least the following columns:

| file_name  | label   |
| ----------- | ------- |
| sample_001 | low-risk  |
| sample_002 | high-risk |
| ...         | ...     |

Each `file_name` should correspond to a sample, and match the `.npy` files in the embeddings folders.

---

### 2. `data/embeddings/images/`

Contains NumPy `.npy` files for **image embeddings**. Each file should be named:

```
sample_001-image-features.npy
sample_002-image-features.npy
...
```

### 3. `data/embeddings/sequences/`

Contains NumPy `.npy` files for **sequence embeddings**. Each file should be named:

```
sample_001-seq-features.npy
sample_002-seq-features.npy
...
```

---

## Running MarbliX

After placing your data and updating the paths in the code, you can run the training pipeline with:

```bash
python src/marblix.py
```

This will:

* Load and normalize image and sequence embeddings
* Train:

  * Image -> Sequence Hybrid Autoencoder
  * Sequence -> Image Hybrid Autoencoder
  * Multimodal Triplet Model
* Extract real-valued and binary multimodal features
* Optionally visualize binary feature matrices

---

## Output

* Console output includes training metrics and model summaries.
* Final multimodal (`features`) and binary (`binary_features`) features are computed for test samples.
* Visualization of one binary matrix (`8x8`) is displayed using matplotlib.

You can extend the code to save outputs by adding:

```python
np.save('multimodal_features.npy', multimodal_features)
np.save('multimodal_binary_features.npy', multimodal_binary_features)
```

