from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "../models/distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

while True:
    user_input = input("User: ")
    prompt = f"User: {user_input}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True,top_k=50,top_p =0.95,temperature=0.8,pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()

    print("Assistant:", response)
