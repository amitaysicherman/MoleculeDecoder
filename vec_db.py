import numpy as np
from typing import List, Tuple, Dict, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class VectorDatabase:
    def __init__(self, distance_metric: str = "cosine"):
        self.vectors = []
        self.documents = []
        self.distance_metric = distance_metric

        if distance_metric not in ["cosine", "l2"]:
            raise ValueError("Distance metric must be either 'cosine' or 'l2'")

    def add_vectors(self, vectors: List[np.ndarray], documents: List[str]):
        if len(vectors) != len(documents):
            raise ValueError("Number of vectors must match number of documents")

        self.vectors.extend(vectors)
        self.documents.extend(documents)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, str]]:
        if not self.vectors:
            return []
        scores = []
        if self.distance_metric == "cosine":
            query_norm = np.linalg.norm(query_vector)
            if query_norm == 0:
                raise ValueError("Query vector has zero norm")
            normalized_query = query_vector / query_norm

            for idx, vector in enumerate(self.vectors):
                vec_norm = np.linalg.norm(vector)
                if vec_norm == 0:
                    scores.append((idx, 0))
                    continue

                normalized_vec = vector / vec_norm
                similarity = np.dot(normalized_query, normalized_vec)
                scores.append((idx, similarity))

        elif self.distance_metric == "l2":
            # Calculate L2 distance (Euclidean)
            for idx, vector in enumerate(self.vectors):
                distance = np.linalg.norm(query_vector - vector)
                # Convert distance to similarity (smaller distance = higher similarity)
                similarity = 1 / (1 + distance)
                scores.append((idx, similarity))

        # Sort by similarity score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        results = []
        for idx, score in scores[:top_k]:
            results.append((idx, score, self.documents[idx]))

        return results

    def __len__(self):
        return len(self.vectors)


def create_vector_database(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        documents: List[str],
        batch_size: int = 8,
        max_length: int = 512,
        device: Optional[str] = None,
        distance_metric: str = "cosine",
        metadata: Optional[List[Dict]] = None
) -> VectorDatabase:
    """
    Create a vector database from a list of documents using the given model and tokenizer.

    Args:
        model (PreTrainedModel): The model to use for embedding
        tokenizer (PreTrainedTokenizer): The tokenizer to use
        documents (List[str]): The list of documents to embed
        batch_size (int): Batch size for processing
        max_length (int): Maximum token length for the model
        device (Optional[str]): Device to run model on ('cuda', 'cpu', etc.)
        distance_metric (str): Distance metric for similarity search
        metadata (Optional[List[Dict]]): Optional metadata for each document

    Returns:
        VectorDatabase: A vector database containing the embedded documents
    """
    # Set the device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    vectors = []

    # Process documents in batches
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]

        # Tokenize the batch
        encoded_input = tokenizer(
            batch_docs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        # Generate embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            embeddings = model_output.pooler_output
            batch_vectors = embeddings.cpu().numpy()
            vectors.extend(batch_vectors)

    # Create and return the vector database
    db = VectorDatabase(distance_metric=distance_metric)
    db.add_vectors(vectors, documents, metadata)
    return db



# Example usage:
"""
from transformers import AutoModel, AutoTokenizer

# Load model and tokenizer (e.g., BERT or any other embedding model)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning models can be used for text embeddings",
    "Vector databases enable efficient similarity search",
    "Python is a popular programming language for AI development",
    "Natural language processing helps computers understand human language"
]

# Optional metadata
metadata = [
    {"id": 1, "category": "proverb"},
    {"id": 2, "category": "machine learning"},
    {"id": 3, "category": "databases"},
    {"id": 4, "category": "programming"},
    {"id": 5, "category": "NLP"}
]

# Create vector database
db = create_vector_database(
    model=model,
    tokenizer=tokenizer,
    documents=documents,
    metadata=metadata,
    distance_metric="cosine"
)

# Example search query
query = "How do computers understand language?"
query_tokens = tokenizer(query, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    query_output = model(**query_tokens)
    if hasattr(query_output, "last_hidden_state"):
        query_embedding = mean_pooling(
            query_output.last_hidden_state, 
            query_tokens.get("attention_mask", None)
        )
    else:
        query_embedding = query_output.pooler_output

    query_vector = query_embedding.cpu().numpy()[0]

# Search for similar documents
results = db.search(query_vector, top_k=3)

# Print results
for idx, score, document, meta in results:
    print(f"Score: {score:.4f}, Category: {meta.get('category', 'N/A')}")
    print(f"Document: {document}")
    print()
"""