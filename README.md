# Sentiment Analysis with BERT

A deep learning project for **sentiment classification of tweets** using **BERT (Bidirectional Encoder Representations from Transformers)**. The project includes data preprocessing, vocabulary/tokenizer setup, model training, evaluation, and visualization of results such as confusion matrix, ROC curves, and word clouds.  

---

## Features
- Preprocess tweets with cleaning, tokenization, and splitting into train/val/test sets.  
- Fine-tune `bert-base-uncased` on sentiment labels (`negative`, `neutral`, `positive`).  
- Track and visualize **training & validation loss**.  
- Generate **classification reports, confusion matrices, ROC curves**.  
- Create **word clouds** for positive and negative predictions.  
- Modular codebase with reproducible pipelines for preprocessing, training, and evaluation.  

---

## Project Structure
```
sentiment-analysis-bert/
├─ data/
│  ├─ train.csv
│  ├─ val.csv
│  └─ test.csv
├─ outputs/
│  ├─ best_model.pt
│  ├─ confusion_matrix.png
│  ├─ training_curves.png
│  ├─ classification_report.txt
│  ├─ wordcloud_negative.png
│  └─ roc_curve.png
├─ src/
│  ├─ preprocess.py
│  ├─ train_lstm.py
│  ├─ train_bert.py
│  ├─ evaluate.py
│  └─ utils.py
└─ README.md
```
---

## Setup
```bash
# Create environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

## Preprocess Data
```bash
python src/preprocess.py --input data/tweets_sample.csv --outdir data --val-size 0.15 --test-size 0.15
```

## Train BERT
```bash
python src/train_bert.py --train data/train.csv --val data/val.csv  --outdir outputs --epochs 3 --batch-size 16 --lr 2e-5  --model bert-base-uncased --max-len 128
```

## Evaluate
```bash
python src/evaluate.py --test data/test.csv --checkpoint outputs  --outdir outputs --wordclouds
```
---

## Results
- **Classification Report:** `outputs/classification_report.txt`
---

- **Confusion Matrix:**  

  <img width="800" height="640" alt="confusion_matrix" src="https://github.com/user-attachments/assets/f7c6d7c0-bb1d-467c-819a-2999a6d32e4f" />
---

- **Training Curves:**  

  <img width="960" height="640" alt="training_curves" src="https://github.com/user-attachments/assets/219af181-15fd-4f1e-aabe-c3234a816af7" />
---

- **Negative Wordcloud:**  

  <img width="1280" height="640" alt="wordcloud_negative" src="https://github.com/user-attachments/assets/1f68aed5-ad15-4388-b132-026fc7e5b65b" />
---

## Requirements
- Python 3.8+  
- PyTorch  
- Transformers (HuggingFace)  
- Scikit-learn, Pandas, Matplotlib, Seaborn  
- WordCloud  

---

## Next Steps
- Expand dataset for more robust evaluation.  
- Try advanced transformer models (RoBERTa, DistilBERT).  
- Apply hyperparameter tuning and cross-validation.  
- Deploy model with FastAPI or Streamlit for interactive demo.  
