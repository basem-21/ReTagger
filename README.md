# ReTagger

**ReTagger** is a lightweight training utility for fine-tuning ONNX-based image tagging models with new tags.  
It allows you to continuously evolve your models by injecting new classes from tagged image folders, while preserving existing knowledge.

---

## ✨ Features
- **Continuous Learning**  
  - Retains previously trained classes while adding new tags.  
  - Supports incremental growth of ONNX-based classifiers.  

- **Feature Extraction**  
  - Converts your model temporarily into a feature extractor.  
  - Pre-computes embeddings for fast training.  

- **Cumulative Training**  
  - Transfers old weights into the new classifier head.  
  - Fine-tunes only the new head for efficiency.  

- **Weight Transplant**  
  - Injects the newly trained weights back into the ONNX model.  
  - Produces a fully compatible `model.onnx` and updated `model.csv`.  

- **ComfyUI Compatibility**  
  - Automatically generates a 4-column `model.csv` with `tag_id`, `name`, `category`, and `count`.  

---

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/ReTagger.git
cd ReTagger
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🖥️ Usage

1. Place your **base ONNX model** and its **CSV tag file** in a folder (e.g. `model_folder`).  
2. Prepare a **training folder** containing images (`.png` or `.jpg`) with corresponding `.txt` files listing comma-separated tags.  
3. Run the trainer:
```bash
python retagger.py
```
4. Edit the config (`csv trainer config.json`) when prompted (paths, learning rate, epochs, etc.).  

After training, you will receive:
- `model.onnx` → the updated model with new tags.  
- `model.csv` → updated tags file (ComfyUI format).  

---

## 📜 License

This project is released under the AGPL-3.0 license for personal and non-commercial use.  
For commercial use, please contact the author for licensing terms.  

---

## 🙏 Credits
- [ONNX](https://onnx.ai/) for model format.  
- [PyTorch](https://pytorch.org/) for training utilities.  
- [Pandas](https://pandas.pydata.org/) for CSV handling.  
- [TorchVision](https://pytorch.org/vision/stable/index.html) for preprocessing.
