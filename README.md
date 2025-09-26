# Sentiment Classification with DistilBERT (LoRA Fine-Tuning)

This project demonstrates **fine-tuning a DistilBERT model** using **Low-Rank Adaptation (LoRA)** for sentiment analysis on the **IMDB movie reviews dataset**.  
The model classifies reviews as either **positive** or **negative**, showcasing how lightweight transformers and parameter-efficient tuning can achieve strong results in text classification.

---

## 📂 Project Structure

- **`local_dataset_utilities.py`**  
  Utilities for:
  - Downloading and extracting the IMDB dataset  
  - Loading reviews into a Pandas DataFrame with sentiment labels (`pos` = 1, `neg` = 0):contentReference[oaicite:0]{index=0}  
  - Partitioning into **train**, **validation**, and **test** CSV files  
  - Custom `IMDBDataset` class for PyTorch integration  

- **`lora_distilbert_finetuning_merged.ipynb`**  
  Jupyter notebook that:
  - Loads and tokenizes the IMDB dataset  
  - Fine-tunes DistilBERT with **LoRA adapters**  
  - Evaluates model performance on validation/test sets  
  - Demonstrates inference on new text samples  

- **`requirements.txt`**:contentReference[oaicite:1]{index=1}  
  Required dependencies:
  ```txt
  transformers
  datasets
  lightning
  watermark
  scikit-learn==1.5.2

