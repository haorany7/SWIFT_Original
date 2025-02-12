from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "microsoft/Phi-3.5-mini-instruct"  # Replace with your model name
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare dummy input data
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")

# Run a forward pass
try:
    with torch.no_grad():
        outputs = model(**inputs)
    print("Forward pass successful!")
    print("Output logits:", outputs.logits)
except Exception as e:
    print(f"Error during forward pass: {e}")