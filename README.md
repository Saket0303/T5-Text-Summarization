# T5-Text-Summarization

# Abstractive Text Summarization using T5 Transformers

This project implements an **abstractive text summarization system** using the **T5 (Text-to-Text Transfer Transformer)** model.  
The model is fine-tuned on the **CNN/DailyMail dataset** and evaluated using **ROUGE metrics**.  
The entire pipeline is optimized to run on limited GPU environments such as **Kaggle**.

---

## 🚀 Features
- Transformer-based encoder–decoder model (T5)
- Fine-tuning on real-world news articles
- ROUGE-1, ROUGE-2, ROUGE-L evaluation
- Memory-optimized training for constrained GPUs
- Modular structure: training, evaluation, inference
- Ready for deployment (API integration possible)

---

## 📂 Project Structure
t5-text-summarization/
│
├── train.py            # Fine-tuning script for T5
├── evaluate.py         # ROUGE evaluation script
├── inference.py        # Run summarization on new input text
├── requirements.txt    # Project dependencies
├── README.md
│
├── notebooks/
│   └── experiments.ipynb   # Kaggle experimentation notebook
│
└── models/
    └── t5_finetuned/       # Saved fine-tuned model and tokenizer



---

## 🧠 Model Used
- **Model:** `t5-small` (can be upgraded to `t5-base`)
- **Architecture:** Encoder–Decoder Transformer
- **Tokenizer:** SentencePiece
- **Framework:** Hugging Face Transformers

---

## 📊 Dataset
- **CNN/DailyMail Dataset**
  - Articles as input
  - Human-written highlights as summaries
- Loaded directly via Hugging Face:
```python
load_dataset("cnn_dailymail", "3.0.0")
```


## **Installation**
```
- pip install -r requirements.txt
```
or
```
- pip install transformers datasets evaluate rouge_score sentencepiece torch accelerate
```
## Training
- Load and preprocess the dataset
- Fine-tune the T5 model
- Save the trained model locally
```python
python train.py
```
## Evaluation (ROUGE Metrics)
- ROUGE-1 → Unigram overlap
- ROUGE-2 → Bigram overlap
- ROUGE-L → Longest common subsequence
```python
python evaluate.py
```
## Memory Optimizations Used
- Dynamic padding using DataCollatorForSeq2Seq
- Small batch size with gradient accumulation
- Mixed precision training (fp16)
- Reduced generation length during evaluation
- Small validation split for ROUGE computation
- GPU memory cleanup before evaluation

These optimizations allow the model to be trained and evaluated on Colab GPUs without memory crashes.

## 🌱 Future Improvements

- Upgrade to t5-base for higher quality summaries
- Train on larger dataset splits
- Deploy as a REST API using FastAPI
- Add long-text chunking for very large documents
- Compare with BART and PEGASUS models

✨ Author
## Saket Jasuja
Software Engineer | ML & GenAI Enthusiast
