import main
import os
import numpy as np
from typing import List

# Temporarily set the environment variable for the embedding model name
os.environ["EMBEDDING_MODEL_NAME"] = "text-embedding-gecko@001"

class FakeEmbeddingResponse:
    def __init__(self, values: List[float]):
        self.values = values

class FakeTextEmbeddingModel:
    @classmethod
    def from_pretrained(cls, model_name: str):
        # Simulate the actual model's behavior for testing
        instance = cls() # Create an instance of FakeTextEmbeddingModel
        if model_name == "text-embedding-gecko@001":
            instance.expected_dim = 768
        else:
            instance.expected_dim = 16 # Default for other fake models
        return instance

    def get_embeddings(self, texts: List[str]) -> List[FakeEmbeddingResponse]:
        # Return fake embeddings with the expected dimension
        return [FakeEmbeddingResponse([0.1] * self.expected_dim) for _ in texts]

# Monkeypatch the TextEmbeddingModel to use our fake version
original_TextEmbeddingModel = main.TextEmbeddingModel
main.TextEmbeddingModel = FakeTextEmbeddingModel

try:
    # Test _get_embedding_model
    embedding_model = main._get_embedding_model()
    
    # Test the embedding generation logic from process_document_pipeline
    sample_text = "This is a test sentence for embedding."
    text_splitter = main.RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(sample_text)
    
    embeddings = None
    if chunks:
        all_embeddings = []
        for i in range(0, len(chunks), 5):
            batch_chunks = chunks[i:i+5]
            response = embedding_model.get_embeddings(batch_chunks)
            all_embeddings.extend([embedding.values for embedding in response])
        embeddings = np.array(all_embeddings)

    print('EMBED_RETURN_TYPE', type(embeddings).__name__)
    if embeddings is not None:
        print('EMBED_SHAPE', embeddings.shape)
        expected_dim = 768 # For text-embedding-gecko@001
        assert embeddings.shape[1] == expected_dim, f'Unexpected embedding dimension: Expected {expected_dim}, got {embeddings.shape[1]}'
    else:
        raise SystemExit('Embeddings returned None in smoke test')

    print('SMOKE_OK')

finally:
    # Restore the original TextEmbeddingModel
    main.TextEmbeddingModel = original_TextEmbeddingModel
    # Clean up the environment variable
    del os.environ["EMBEDDING_MODEL_NAME"]
