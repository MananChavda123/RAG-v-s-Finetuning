from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import torch
import numpy as np

MODEL_DIR = "../models/distilgpt2"
DATA_FILE = "../data/train1.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL = "all-MiniLM-L6-v2"  # Compact + Fast

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding is set

dataset = load_dataset("json", data_files=DATA_FILE, split="train")
documents = dataset["text"]

embedder = SentenceTransformer(EMBED_MODEL)
doc_embeddings = embedder.encode(documents, convert_to_tensor=False)

dim = len(doc_embeddings[0])
index = faiss.IndexFlatL2(dim)
doc_embeddings = np.array(doc_embeddings).astype("float32")
index.add(doc_embeddings)


def retrieve_context(query, k=3):
    query_embedding = embedder.encode([query])[0]
    query_embedding = np.array([query_embedding]).astype("float32")  # ✅ Convert to correct shape & dtype
    D, I = index.search(query_embedding, k)
    return [documents[i] for i in I[0]]


# ==== GENERATE RESPONSE ====
def rag_generate(query):
    context_docs = retrieve_context(query, k=3)
    context = "\n---\n".join(context_docs)  # clear separation between docs
    prompt = f"Context:\n{context}\n\nUser: {query}\nAssistant:"


    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    output = model.generate(
        **inputs,
        max_length=256,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("Assistant:")[-1].strip()

if __name__ == "__main__":
    print("RAG Chatbot Ready! Type 'exit' to quit.")
    while True:
        query = input("User: ")
        if query.lower() == "exit":
            break
        answer = rag_generate(query)
        print(f"Assistant: {answer}")
