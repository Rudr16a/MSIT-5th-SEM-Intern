import torch

# Load pre-trained SPLADE model and tokenizer
model_name = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

def generate_sparse_vector(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Generate SPLADE representation
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the SPLADE sparse vector
    logits = outputs.logits[0]
    sparse_vector = torch.max(torch.log1p(torch.relu(logits)) * inputs['attention_mask'][0].unsqueeze(-1), dim=0)[0]
    
    # Convert to dictionary format (term_id: weight)
    sparse_dict = {idx: weight.item() for idx, weight in enumerate(sparse_vector) if weight > 0}
    
    return sparse_dict

# Example usage
text = "This is an example sentence for SPLADE sparse vector generation."
sparse_vector = generate_sparse_vector(text)

print("Sparse vector:")
print(sparse_vector)

# Optionally, you can map the term IDs back to tokens
id2token = {v: k for k, v in tokenizer.get_vocab().items()}
sparse_terms = {id2token[idx]: weight for idx, weight in sparse_vector.items()}

print("\nSparse terms:")
print(sparse_terms)
