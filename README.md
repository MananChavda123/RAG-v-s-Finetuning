# RAG vs Fine-tuning

This project compares two approaches for adapting language models to specific tasks: **Retrieval-Augmented Generation (RAG)** and **Fine-Tuning**.

---

## 🛠️ Setup Instructions

Follow the steps below to set up and run the project.

### 1. Create a Virtual Environment

```bash
python -m venv env
```

Activate it:

- On **Windows**:
  ```bash
  .\env\Scripts\activate
  ```
- On **macOS/Linux**:
  ```bash
  source env/bin/activate
  ```

---

### 2. Install Dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Fine-Tuning Pipeline

1. Navigate to the `Finetune` folder:
   ```bash
   cd Finetune
   ```

2. Run the fine-tuning script:
   ```bash
   python Finetune.py
   ```

3. Start the chatbot:
   ```bash
   python chat.py
   ```

---

### RAG Pipeline

1. Navigate to the `RAG` folder:
   ```bash
   cd RAG
   ```

2. Run the RAG script:
   ```bash
   python RAG.py
   ```

---

## 📁 Project Structure

```
├── Finetune/
│   ├── Finetune.py
│   └── chat.py
├── RAG/
│   └── RAG.py
├── data/
│   └── train1.jsonl
├── requirements.txt
└── README.md
```

---

## 📌 Notes

- First always run the finetuning files and then the RAG files.
- Ensure the virtual environment is activated before running any Python scripts.
- Models used during training are ignored in version control. Check `.gitignore` for details.

---

## 🧠 About

This project demonstrates the practical differences between fine-tuning a model on specific data and using retrieval-augmented generation to enhance performance without retraining the base model.
