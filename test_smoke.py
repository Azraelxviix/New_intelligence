import main

class FakeCursor:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def execute(self, *args, **kwargs):
        pass
    def fetchone(self):
        return None
    def fetchall(self):
        return []

class FakeDB:
    def cursor(self):
        return FakeCursor()
    def commit(self):
        pass
    def rollback(self):
        pass

class FakeEmbedding:
    def __init__(self, dim=16):
        self.values = [0.1]*dim

class FakeModel:
    def get_embeddings(self, batch_texts):
        return [FakeEmbedding() for _ in batch_texts]

# Monkeypatch aiplatform model loader
orig_text_embedding_model = getattr(main.aiplatform, 'TextEmbeddingModel', None)
class _StubLoader:
    @staticmethod
    def from_pretrained(name):
        return FakeModel()

setattr(main.aiplatform, 'TextEmbeddingModel', _StubLoader)

# Run embedding generation
sample_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 200  # long enough for multiple chunks
arr = main._generate_and_store_embeddings(1, sample_text, FakeDB(), 'textembedding-gecko@003')

print('EMBED_RETURN_TYPE', type(arr).__name__)
if arr is not None:
    print('EMBED_SHAPE', arr.shape)
    assert arr.shape[1] == 16, 'Unexpected embedding dimension'
else:
    raise SystemExit('Embeddings returned None in smoke test')

# Restore original if needed
if orig_text_embedding_model is not None:
    setattr(main.aiplatform, 'TextEmbeddingModel', orig_text_embedding_model)

print('SMOKE_OK')
