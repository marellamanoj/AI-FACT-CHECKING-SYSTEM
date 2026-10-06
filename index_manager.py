"""
Index persistence manager for FactCheckLIAR.

Provides functions to save and load FAISS and BM25 indexes to disk,
avoiding the ~30-60s rebuild time on every startup.
"""

import hashlib
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


DEFAULT_CACHE_DIR = "cache"


def compute_dataset_hash(data_path: str) -> str:
    """
    Compute MD5 hash of a dataset file for cache invalidation.

    Args:
        data_path: Path to the dataset file

    Returns:
        MD5 hash string of the file contents
    """
    hasher = hashlib.md5()
    with open(data_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_faiss_index(index: faiss.Index, path: str) -> None:
    """
    Save a FAISS index to disk.

    Args:
        index: FAISS index to save
        path: File path to save to
    """
    faiss.write_index(index, path)


def load_faiss_index(path: str) -> faiss.Index:
    """
    Load a FAISS index from disk.

    Args:
        path: File path to load from

    Returns:
        Loaded FAISS index
    """
    return faiss.read_index(path)


def save_bm25_index(bm25: BM25Okapi, path: str) -> None:
    """
    Save a BM25 index to disk using pickle.

    Args:
        bm25: BM25 index to save
        path: File path to save to
    """
    with open(path, "wb") as f:
        pickle.dump(bm25, f)


def load_bm25_index(path: str) -> BM25Okapi:
    """
    Load a BM25 index from disk.

    Args:
        path: File path to load from

    Returns:
        Loaded BM25 index
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def is_cache_valid(cache_dir: str, data_path: str) -> bool:
    """
    Check if cached indexes exist and match the current dataset.

    Args:
        cache_dir: Directory containing cached indexes
        data_path: Path to the dataset file

    Returns:
        True if cache is valid, False otherwise
    """
    cache_path = Path(cache_dir)
    faiss_path = cache_path / "faiss_index.bin"
    bm25_path = cache_path / "bm25_index.pkl"
    hash_path = cache_path / "dataset_hash.txt"

    # Check all required files exist
    if not all(p.exists() for p in [faiss_path, bm25_path, hash_path]):
        return False

    # Check hash matches current dataset
    current_hash = compute_dataset_hash(data_path)
    with open(hash_path, "r") as f:
        cached_hash = f.read().strip()

    return current_hash == cached_hash


def load_or_build_indexes(
    data_path: str,
    statements: list[str],
    cache_dir: str = DEFAULT_CACHE_DIR,
    verbose: bool = False
) -> tuple[BM25Okapi, faiss.Index, SentenceTransformer]:
    """
    Load indexes from cache if valid, otherwise build and cache them.

    Args:
        data_path: Path to the dataset file (for hash computation)
        statements: List of claim statements to index
        cache_dir: Directory to store/load cached indexes
        verbose: Whether to print status messages

    Returns:
        Tuple of (bm25_index, faiss_index, dense_model)
    """
    cache_path = Path(cache_dir)
    faiss_path = cache_path / "faiss_index.bin"
    bm25_path = cache_path / "bm25_index.pkl"
    hash_path = cache_path / "dataset_hash.txt"

    # Always load the dense model (needed for query encoding)
    dense_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Check if we can use cached indexes
    if is_cache_valid(cache_dir, data_path):
        if verbose:
            print(f"Loading indexes from cache ({cache_dir})...")

        bm25 = load_bm25_index(str(bm25_path))
        faiss_index = load_faiss_index(str(faiss_path))

        if verbose:
            print("Indexes loaded from cache.")

        return bm25, faiss_index, dense_model

    # Build indexes from scratch
    if verbose:
        print("Building indexes (this may take 30-60 seconds)...")

    # Build BM25 index
    tokenized_docs = [doc.lower().split() for doc in statements]
    bm25 = BM25Okapi(tokenized_docs)
    if verbose:
        print("BM25 index built.")

    # Compute dense embeddings and build FAISS index
    if verbose:
        print("Computing dense embeddings for FAISS index...")
    embeddings = dense_model.encode(statements, batch_size=32, show_progress_bar=verbose)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    d = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(embeddings)
    if verbose:
        print("FAISS index built.")

    # Save to cache
    cache_path.mkdir(parents=True, exist_ok=True)

    save_bm25_index(bm25, str(bm25_path))
    save_faiss_index(faiss_index, str(faiss_path))

    current_hash = compute_dataset_hash(data_path)
    with open(hash_path, "w") as f:
        f.write(current_hash)

    if verbose:
        print(f"Indexes cached to {cache_dir}/")

    return bm25, faiss_index, dense_model
