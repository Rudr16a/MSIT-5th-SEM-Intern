# Indexing dense vectors in Qdrant
client.create_collection('medical_dense_vectors', vector_size=768)

# Indexing the dense vectors
client.upload_collection(
    collection_name='medical_dense_vectors',
    vectors=df['dense_vectors'].tolist(),
    payload=df['text'].tolist()
)

# Querying dense vectors
query_vector = generate_dense_vector('treatment for high blood sugar')
results = client.search(
    collection_name='medical_dense_vectors',
    query_vector=query_vector,
    top=5
)

