from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Load pre-trained SPLADE model and tokenizer
splade_model_name = "naver/splade-cocondenser-ensembledistil"
splade_tokenizer = AutoTokenizer.from_pretrained(splade_model_name)
splade_model = AutoModelForMaskedLM.from_pretrained(splade_model_name)

# Load pre-trained Sentence Transformer model
sbert_model_name = "all-MiniLM-L6-v2"
sbert_model = SentenceTransformer(sbert_model_name)

def generate_sparse_vector(text):
    inputs = splade_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = splade_model(**inputs)
    
    logits = outputs.logits[0]
    sparse_vector = torch.max(torch.log1p(torch.relu(logits)) * inputs['attention_mask'][0].unsqueeze(-1), dim=0)[0]
    
    sparse_dict = {idx: weight.item() for idx, weight in enumerate(sparse_vector) if weight > 0}
    return sparse_dict

def generate_dense_vector(text):
    return sbert_model.encode(text)

def combine_vectors(sparse_vector, dense_vector, alpha=0.5):
    # Convert sparse vector to dense format
    max_dim = max(sparse_vector.keys()) + 1
    sparse_dense = np.zeros(max_dim)
    for idx, weight in sparse_vector.items():
        sparse_dense[idx] = weight
    
    # Normalize vectors
    sparse_dense = sparse_dense / np.linalg.norm(sparse_dense)
    dense_vector = dense_vector / np.linalg.norm(dense_vector)
    
    # Pad dense vector if necessary
    if len(dense_vector) < max_dim:
        dense_vector = np.pad(dense_vector, (0, max_dim - len(dense_vector)))
    else:
        dense_vector = dense_vector[:max_dim]
    
    # Combine vectors
    combined = alpha * sparse_dense + (1 - alpha) * dense_vector
    
    return combined

# Example usage
text = "This is an example sentence for combining sparse and dense vectors."
sparse_vector = generate_sparse_vector(text)
dense_vector = generate_dense_vector(text)
combined_vector = combine_vectors(sparse_vector, dense_vector)

print("Sparse vector (top 5 terms):")
print(dict(sorted(sparse_vector.items(), key=lambda x: x[1], reverse=True)[:5]))

print("\nDense vector (first 5 dimensions):")
print(dense_vector[:5])

print("\nCombined vector (first 5 dimensions):")
print(combined_vector[:5])

# Optionally, map sparse terms to tokens
id2token = {v: k for k, v in splade_tokenizer.get_vocab().items()}
sparse_terms = {id2token[idx]: weight for idx, weight in sparse_vector.items()}

print("\nTop 5 sparse terms:")
print(dict(sorted(sparse_terms.items(), key=lambda x: x[1], reverse=True)[:5]))

