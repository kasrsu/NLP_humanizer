# NLP_humanizer


# 🤖🧠 AI Text Detector & Converter — Humanize Your AI Text

A full-stack NLP system that classifies whether a sentence is AI-written or human-written — and if it's AI-written, rewrites it to sound human. Built using TF-IDF, RoBERTa, SentenceBERT, and local LLMs.

🔥 This project tackles a real problem: AI-generated content feels robotic. This system detects that style and rewrites it to be natural, semantic, and human.

---

## 🚀 Project Goals

- 🕵️‍♂️ Detect if a given sentence or paragraph was written by AI
- 🧠 Automatically rewrite AI-style text to sound like a human
- 🧼 Handle slang, robotic phrases, and awkward structure in AI outputs
- 🧪 Evaluate classification accuracy and rewriting quality

---

## 🛠️ Tech Stack

| Module         | Tools / Models Used                              |
|----------------|--------------------------------------------------|
| Preprocessing  | spaCy, custom slang filter, tokenizer            |
| Classifier     | TF-IDF + POS Features → Logistic Regression, RF  |
| Rewriter       | RoBERTa + SentenceBERT + FastText for context    |
| Embedding Logic| Cosine similarity for word replacement           |
| Deployment     | Flask (optional), local testing with Ollama      |

---

## 📊 Model Performance

- **Classifier Accuracy**: `88.43%`
- **AUC-ROC**: `0.948`
- **Tested On**: Pre-2020 Human Articles vs. LLM Outputs (GPT, Mistral)

---

## 💡 Key Features

### 🧠 Text Classification
Detects if input text is AI-written using:
- **TF-IDF features** for token weighting  
- **POS tag normalization** to catch robotic sentence structures  
- Trained using **Logistic Regression + Random Forest**, ensemble voting  

### 🔁 AI-to-Human Rewriting
If AI text is detected:
- **RoBERTa + SentenceBERT** used to rank words/phrases for replacement  
- **FastText & custom slang filters** applied to humanize tone  
- Uses **cosine similarity** to choose contextually accurate replacements  

---


