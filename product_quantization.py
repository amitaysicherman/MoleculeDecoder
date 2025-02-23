import numpy as np
import os
import json
from sklearn.cluster import MiniBatchKMeans
from typing import List, Tuple, Optional, Dict


class ProductQuantization:
    def __init__(
            self,
            n_subspaces: int,
            n_clusters: int,
            batch_size: int = 1024,
            max_iter: int = 10,
            n_init: str = "auto",
            random_state: Optional[int] = None
    ):
        """
        Initialize Product Quantization with multiple MiniBatchKMeans models.

        Args:
            n_subspaces: Number of subspaces to split the input vectors
            n_clusters: Number of clusters for each subspace
            batch_size: Size of mini batches
            max_iter: Maximum number of iterations over the full dataset
            n_init: Number of initializations, "auto" for automatic selection
            random_state: Random state for reproducibility
        """
        self.n_subspaces = n_subspaces
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        # Initialize a MiniBatchKMeans model for each subspace
        self.subspace_models = [
            MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=batch_size,
                max_iter=max_iter,
                n_init=n_init,
                random_state=random_state + i if random_state else None
            )
            for i in range(n_subspaces)
        ]

        self.vector_dim = None
        self.subvector_dim = None

    def _split_vector(self, X: np.ndarray) -> List[np.ndarray]:
        """Split input vectors into subvectors."""
        return np.array_split(X, self.n_subspaces, axis=1)

    def _get_batches(self, X: np.ndarray) -> List[np.ndarray]:
        """Split data into batches."""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            yield X[indices[start_idx:end_idx]]

    def partial_fit(self, X: np.ndarray) -> 'ProductQuantization':
        """
        Partially fit Product Quantization model to a batch of data.

        Args:
            X: Training data batch of shape (n_samples, n_features)

        Returns:
            self: Partially fitted model
        """
        # Initialize dimensions if not set
        if self.vector_dim is None:
            self.vector_dim = X.shape[1]
            self.subvector_dim = self.vector_dim // self.n_subspaces

        # Check dimensions
        if X.shape[1] != self.vector_dim:
            raise ValueError(f"Expected input dimension {self.vector_dim}, got {X.shape[1]}")

        # Split vectors into subspaces
        subvectors = self._split_vector(X)

        # Partially fit each subspace model
        for subvector, model in zip(subvectors, self.subspace_models):
            model.partial_fit(subvector)

        return self

    def fit(self, X: np.ndarray) -> 'ProductQuantization':
        """
        Fit Product Quantization model to the entire dataset using mini-batches.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self: Fitted model
        """
        # Initialize dimensions
        self.vector_dim = X.shape[1]
        self.subvector_dim = self.vector_dim // self.n_subspaces

        # Train for multiple epochs
        for epoch in range(self.max_iter):
            print(f"Epoch {epoch + 1}/{self.max_iter}")

            # Process data in batches
            for batch in self._get_batches(X):
                self.partial_fit(batch)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data into quantized representations."""
        if self.vector_dim != X.shape[1]:
            raise ValueError(f"Expected input dimension {self.vector_dim}, got {X.shape[1]}")

        # Split vectors and get cluster assignments
        subvectors = self._split_vector(X)
        quantized = np.column_stack([
            np.argmin(
                np.sum(
                    (subvector[:, np.newaxis, :] - model.cluster_centers_[np.newaxis, :, :]) ** 2,
                    axis=2
                ),
                axis=1
            )
            for subvector, model in zip(subvectors, self.subspace_models)
        ])

        return quantized

    def inverse_transform(self, quantized: np.ndarray) -> np.ndarray:
        """Reconstruct original vectors from quantized representations."""
        reconstructed_subvectors = []

        for subspace_idx, model in enumerate(self.subspace_models):
            subspace_clusters = quantized[:, subspace_idx]
            reconstructed_subvectors.append(model.cluster_centers_[subspace_clusters])

        return np.hstack(reconstructed_subvectors)

    def save(self, directory: str):
        """
        Save the Product Quantization model to a directory.

        Args:
            directory: Path to the directory where the model will be saved
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save model parameters
        params = {
            'n_subspaces': self.n_subspaces,
            'n_clusters': self.n_clusters,
            'batch_size': self.batch_size,
            'max_iter': self.max_iter,
            'n_init': self.n_init,
            'random_state': self.random_state,
            'vector_dim': self.vector_dim,
            'subvector_dim': self.subvector_dim
        }

        with open(os.path.join(directory, 'params.json'), 'w') as f:
            json.dump(params, f)

        # Save cluster centers for each subspace model
        for i, model in enumerate(self.subspace_models):
            np.save(
                os.path.join(directory, f'cluster_centers_{i}.npy'),
                model.cluster_centers_
            )

    def load(self, directory: str) -> 'ProductQuantization':
        """
        Load a saved Product Quantization model from a directory.

        Args:
            directory: Path to the directory containing the saved model

        Returns:
            self: Loaded model
        """
        # Load model parameters
        with open(os.path.join(directory, 'params.json'), 'r') as f:
            params = json.load(f)

        # Initialize model with loaded parameters
        self.n_subspaces = params['n_subspaces']
        self.n_clusters = params['n_clusters']
        self.batch_size = params['batch_size']
        self.max_iter = params['max_iter']
        self.n_init = params['n_init']
        self.random_state = params['random_state']
        self.vector_dim = params['vector_dim']
        self.subvector_dim = params['subvector_dim']

        # Initialize subspace models
        self.subspace_models = [
            MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=self.batch_size,
                max_iter=self.max_iter,
                n_init=self.n_init,
                random_state=self.random_state + i if self.random_state else None
            )
            for i in range(self.n_subspaces)
        ]

        # Load cluster centers for each subspace model
        for i, model in enumerate(self.subspace_models):
            centers_path = os.path.join(directory, f'cluster_centers_{i}.npy')
            model.cluster_centers_ = np.load(centers_path)
            # Set flag to indicate the model has been fitted
            model._fitted = True

        return self

    def compute_distances(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Compute approximate distances between query vectors and database vectors."""
        n_queries = query.shape[0]
        n_samples = codes.shape[0]
        distances = np.zeros((n_queries, n_samples))

        query_subvectors = self._split_vector(query)

        for subspace_idx, (subvector, model) in enumerate(zip(query_subvectors, self.subspace_models)):
            centroid_distances = np.square(
                subvector[:, np.newaxis, :] - model.cluster_centers_[np.newaxis, :, :]
            ).sum(axis=2)

            distances += centroid_distances[
                np.arange(n_queries)[:, np.newaxis],
                codes[:, subspace_idx][np.newaxis, :]
            ]

        return np.sqrt(distances)


# Example usage demonstrating batch processing
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 128
    X = np.random.randn(n_samples, n_features)

    # Initialize PQ model
    pq = ProductQuantization(
        n_subspaces=8,
        n_clusters=10,
        batch_size=100,  # Process 100 samples at a time
        max_iter=5,  # 5 epochs over the full dataset
        random_state=42
    )

    # Method 1: Manual batch processing
    print("Manual batch processing:")
    for i in range(0, n_samples, 100):
        batch = X[i:i + 100]
        pq.partial_fit(batch)
        print(f"Processed batch {i // 100 + 1}")

    # Test the model
    codes = pq.transform(X[:10])
    print("\nFirst 10 codes shape:", codes.shape)

    # Method 2: Automatic batch processing
    print("\nAutomatic batch processing:")
    pq = ProductQuantization(
        n_subspaces=8,
        n_clusters=10,
        batch_size=100,
        max_iter=5,
        random_state=42
    )
    pq.fit(X)  # This will handle batching internally

    # Test reconstruction
    codes = pq.transform(X)
    X_reconstructed = pq.inverse_transform(codes)
    reconstruction_error = np.mean(np.square(X - X_reconstructed))
    print(f"\nMean squared reconstruction error: {reconstruction_error:.6f}")

    # Test save and load functionality
    print("\nTesting save and load functionality:")

    # Save the model
    save_dir = "pq_model"
    pq.save(save_dir)
    print(f"Model saved to {save_dir}")

    # Load the model
    new_pq = ProductQuantization(
        n_subspaces=8,
        n_clusters=256,
        batch_size=100,
        max_iter=5,
        random_state=42
    )
    new_pq.load(save_dir)
    print("Model loaded successfully")

    # Verify the loaded model performs the same
    new_codes = new_pq.transform(X)
    print(new_codes)
    new_X_reconstructed = new_pq.inverse_transform(new_codes)
    new_reconstruction_error = np.mean(np.square(X - new_X_reconstructed))
    print(f"Reconstruction error after loading: {new_reconstruction_error:.6f}")

    # Clean up
    import shutil

    shutil.rmtree(save_dir)