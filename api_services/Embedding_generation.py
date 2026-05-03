from typing import List
from sentence_transformers import SentenceTransformer
import torch

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def embed_text(self, text: List[str]) -> torch.Tensor:
        """Generate embeddings for a list of texts"""

        return self.model.encode(text, convert_to_tensor=True, device=self.device, show_progress_bar=False)


    def embed_query(self, query:str)->torch.Tensor:
        """Generate embeddings for a query"""
        return self.embed_text([query])[0]

    def cosine_similarity(self, vec1:torch.Tensor, vec2:torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between two vectors"""
        return torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))

def __main__():
    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.embed_text(["hello world", "hello universe", "hello my country", "hello my parents"])
    print("Cosine Similarity", embedding_generator.cosine_similarity(embeddings[0], embeddings[1]))
    print("Cosine Similarity", embedding_generator.cosine_similarity(embeddings[0], embeddings[2]))
    print("Cosine Similarity", embedding_generator.cosine_similarity(embeddings[0], embeddings[3]))

if __name__ == "__main__":
    __main__()

    
