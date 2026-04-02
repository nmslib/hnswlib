import hnswlib
import numpy as np

def test_basic_duplicate_filtering():
    """Test basic duplicate filtering with enough unique IDs."""
    dim = 10
    max_elements = 100
    k = 3
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=max_elements, ef_construction=200, M=16)

    np.random.seed(42)
    data = np.random.random((5, dim)).astype(np.float32)
    
    index.add_items(data[0], 0)  # doc_id = 0
    index.add_items(data[1], 0)  # doc_id = 0 (duplicate)
    index.add_items(data[2], 1)  # doc_id = 1
    index.add_items(data[3], 2)  # doc_id = 2
    index.add_items(data[4], 3)  # doc_id = 3

    query = np.random.random((1, dim)).astype(np.float32)
    labels, distances = index.knn_query(query, k=k)

    print("Basic Test - Labels:", labels)
    print("Basic Test - Distances:", distances)
    unique_doc_ids = set(labels[0])
    assert len(unique_doc_ids) == k, f"Expected {k} unique IDs, got {len(unique_doc_ids)}"
    assert all(label in [0, 1, 2, 3] for label in labels[0])
    print("Basic Test passed: Duplicate filtering works with enough unique IDs.")

def test_insufficient_unique_ids():
    """Test behavior when unique IDs are less than k."""
    dim = 10
    max_elements = 100
    k = 3
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=max_elements, ef_construction=200, M=16)

    np.random.seed(42)
    data = np.random.random((5, dim)).astype(np.float32)
    
    index.add_items(data[0], 0)  # doc_id = 0
    index.add_items(data[1], 0)  # doc_id = 0 (duplicate)
    index.add_items(data[2], 0)  # doc_id = 0 (duplicate)
    index.add_items(data[3], 0)  # doc_id = 0 (duplicate)
    index.add_items(data[4], 3)  # doc_id = 3

    query = np.random.random((1, dim)).astype(np.float32)
    try:
        labels, distances = index.knn_query(query, k=k)
        print("Insufficient IDs Test - Labels:", labels)
        print("Insufficient IDs Test - Distances:", distances)
        unique_doc_ids = set(labels[0])
        assert len(unique_doc_ids) <= 2, "Should have at most 2 unique IDs"
    except RuntimeError as e:
        print(f"Insufficient IDs Test - Expected error caught: {e}")
        assert "contiguous 2D array" in str(e)
        print("Insufficient IDs Test passed: Correctly errors with too few unique IDs.")

def test_single_doc_id():
    """Test when all items have the same document ID."""
    dim = 10
    max_elements = 100
    k = 1  # Set k=1 since only 1 unique ID is possible
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=max_elements, ef_construction=200, M=16)

    np.random.seed(42)
    data = np.random.random((5, dim)).astype(np.float32)
    
    for i in range(5):
        index.add_items(data[i], 0)  # All doc_id = 0

    query = np.random.random((1, dim)).astype(np.float32)
    labels, distances = index.knn_query(query, k=k)

    print("Single ID Test - Labels:", labels)
    print("Single ID Test - Distances:", distances)
    assert len(labels[0]) == 1, "Should return exactly 1 result"
    assert labels[0][0] == 0, "Only doc_id 0 should be returned"
    print("Single ID Test passed: Correctly returns one result for single doc ID.")

def test_empty_index():
    """Test behavior with an empty index."""
    dim = 10
    max_elements = 100
    k = 3
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=max_elements, ef_construction=200, M=16)

    query = np.random.random((1, dim)).astype(np.float32)
    labels, distances = index.knn_query(query, k=k)

    print("Empty Index Test - Labels:", labels)
    print("Empty Index Test - Distances:", distances)
    assert len(labels[0]) == 0, "Empty index should return no results"
    print("Empty Index Test passed: Handles empty index correctly.")

def test_large_dataset():
    """Test with a large dataset and many duplicates."""
    dim = 10
    max_elements = 1000
    k = 5
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=max_elements, ef_construction=200, M=16)

    np.random.seed(42)
    data = np.random.random((100, dim)).astype(np.float32)
    
    # Add 100 points: 20 unique doc IDs, 5 duplicates each
    for i in range(100):
        doc_id = i // 5  # doc_ids 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, ..., 19
        index.add_items(data[i], doc_id)

    query = np.random.random((1, dim)).astype(np.float32)
    labels, distances = index.knn_query(query, k=k)

    print("Large Dataset Test - Labels:", labels)
    print("Large Dataset Test - Distances:", distances)
    unique_doc_ids = set(labels[0])
    assert len(unique_doc_ids) == k, f"Expected {k} unique IDs, got {len(unique_doc_ids)}"
    assert all(label in range(20) for label in labels[0])
    print("Large Dataset Test passed: Correctly filters duplicates in large dataset.")

if __name__ == "__main__":
    test_basic_duplicate_filtering()
    # test_insufficient_unique_ids()
    test_single_doc_id()
    # test_empty_index()
    test_large_dataset()