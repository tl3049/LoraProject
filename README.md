# Sentiment Classification with DistilBERT (LoRA Fine-Tuning)

This project demonstrates **fine-tuning a DistilBERT model** using **Low-Rank Adaptation (LoRA)** for sentiment analysis on the **IMDB movie reviews dataset**.  
The model classifies reviews as either **positive** or **negative**, showcasing how lightweight transformers and parameter-efficient tuning can achieve strong results in text classification.

---

## Project Structure

- **`local_dataset_utilities.py`**  
  Utilities for:
  - Downloading and extracting the IMDB dataset  
  - Loading reviews into a Pandas DataFrame with sentiment labels (`pos` = 1, `neg` = 0)  
  - Partitioning into **train**, **validation**, and **test** CSV files  
  - Custom `IMDBDataset` class for PyTorch integration  

- **`lora_distilbert_finetuning_merged.ipynb`**  
  Jupyter notebook that:
  - Loads and tokenizes the IMDB dataset  
  - Fine-tunes DistilBERT with **LoRA adapters**  
  - Evaluates model performance on validation/test sets  
  - Deploy the complete model in Huggingface
  - Demonstrates inference on new text samples  

- **`lora_distilbert_finetuning_adapter.ipynb`**  
  Jupyter notebook that:
  - Performs the same three steps as before  
  - Deploys the adapter model to Hugging Face (significantly smaller than the merged model)  
  - Demonstrates inference by loading models directly from Hugging Face, with additional steps required to match layer keys


- **`requirements.txt`**:
  Required dependencies:
  ```txt
  transformers
  datasets
  lightning
  watermark
  scikit-learn==1.5.2

## Installation

Clone the repo and install dependencies:

```python
git clone https://github.com/tl3049/LoraProject.git
pip install -r requirements.txt
```

## Usage
Open and run the Jupyter notebook:
```python
jupyter notebook lora_distilbert_finetuning_merged.ipynb
```


## Dataset

- Source: IMDB Large Movie Review Dataset (http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
- Size: 50,000 labeled reviews (balanced positive/negative)





