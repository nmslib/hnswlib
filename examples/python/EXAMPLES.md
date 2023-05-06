# Python bindings examples

Creating index, inserting elements, searching and pickle serialization:
```python
import hnswlib
import numpy as np
import pickle

dim = 128
num_elements = 10000

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))
ids = np.arange(num_elements)

# Declaring index
p = hnswlib.Index(space = 'l2', dim = dim) # possible options are l2, cosine or ip

# Initializing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = num_elements, ef_construction = 200, M = 16)

# Element insertion (can be called several times):
p.add_items(data, ids)

# Controlling the recall by setting ef:
p.set_ef(50) # ef should always be > k

# Query dataset, k - number of the closest elements (returns 2 numpy arrays)
labels, distances = p.knn_query(data, k = 1)

# Index objects support pickling
# WARNING: serialization via pickle.dumps(p) or p.__getstate__() is NOT thread-safe with p.add_items method!
# Note: ef parameter is included in serialization; random number generator is initialized with random_seed on Index load
p_copy = pickle.loads(pickle.dumps(p)) # creates a copy of index p using pickle round-trip

### Index parameters are exposed as class properties:
print(f"Parameters passed to constructor:  space={p_copy.space}, dim={p_copy.dim}") 
print(f"Index construction: M={p_copy.M}, ef_construction={p_copy.ef_construction}")
print(f"Index size is {p_copy.element_count} and index capacity is {p_copy.max_elements}")
print(f"Search speed/quality trade-off parameter: ef={p_copy.ef}")
```

An example with updates after serialization/deserialization:
```python
import hnswlib
import numpy as np

dim = 16
num_elements = 10000

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))

# We split the data in two batches:
data1 = data[:num_elements // 2]
data2 = data[num_elements // 2:]

# Declaring index
p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

# Initializing index
# max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
# during insertion of an element.
# The capacity can be increased by saving/loading the index, see below.
#
# ef_construction - controls index search speed/build speed tradeoff
#
# M - is tightly connected with internal dimensionality of the data. Strongly affects memory consumption (~M)
# Higher M leads to higher accuracy/run_time at fixed ef/efConstruction

p.init_index(max_elements=num_elements//2, ef_construction=100, M=16)

# Controlling the recall by setting ef:
# higher ef leads to better accuracy, but slower search
p.set_ef(10)

# Set number of threads used during batch search/construction
# By default using all available cores
p.set_num_threads(4)

print("Adding first batch of %d elements" % (len(data1)))
p.add_items(data1)

# Query the elements for themselves and measure recall:
labels, distances = p.knn_query(data1, k=1)
print("Recall for the first batch:", np.mean(labels.reshape(-1) == np.arange(len(data1))), "\n")

# Serializing and deleting the index:
index_path='first_half.bin'
print("Saving index to '%s'" % index_path)
p.save_index("first_half.bin")
del p

# Re-initializing, loading the index
p = hnswlib.Index(space='l2', dim=dim)  # the space can be changed - keeps the data, alters the distance function.

print("\nLoading index from 'first_half.bin'\n")

# Increase the total capacity (max_elements), so that it will handle the new data
p.load_index("first_half.bin", max_elements = num_elements)

print("Adding the second batch of %d elements" % (len(data2)))
p.add_items(data2)

# Query the elements for themselves and measure recall:
labels, distances = p.knn_query(data, k=1)
print("Recall for two batches:", np.mean(labels.reshape(-1) == np.arange(len(data))), "\n")
```

An example with a symbolic filter `filter_function` during the search:
```python
import hnswlib
import numpy as np

dim = 16
num_elements = 10000

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))

# Declaring index
hnsw_index = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

# Initiating index
# max_elements - the maximum number of elements, should be known beforehand
#     (probably will be made optional in the future)
#
# ef_construction - controls index search speed/build speed tradeoff
# M - is tightly connected with internal dimensionality of the data
#     strongly affects the memory consumption

hnsw_index.init_index(max_elements=num_elements, ef_construction=100, M=16)

# Controlling the recall by setting ef:
# higher ef leads to better accuracy, but slower search
hnsw_index.set_ef(10)

# Set number of threads used during batch search/construction
# By default using all available cores
hnsw_index.set_num_threads(4)

print("Adding %d elements" % (len(data)))
# Added elements will have consecutive ids
hnsw_index.add_items(data, ids=np.arange(num_elements))

print("Querying only even elements")
# Define filter function that allows only even ids
filter_function = lambda idx: idx%2 == 0
# Query the elements for themselves and search only for even elements:
# Warning: search with python filter works slow in multithreaded mode, therefore we set num_threads=1
labels, distances = hnsw_index.knn_query(data, k=1, num_threads=1, filter=filter_function)
# labels contain only elements with even id
```

An example with reusing the memory of the deleted elements when new elements are being added (via `allow_replace_deleted` flag):
```python
import hnswlib
import numpy as np

dim = 16
num_elements = 1_000
max_num_elements = 2 * num_elements

# Generating sample data
labels1 = np.arange(0, num_elements)
data1 = np.float32(np.random.random((num_elements, dim)))  # batch 1
labels2 = np.arange(num_elements, 2 * num_elements)
data2 = np.float32(np.random.random((num_elements, dim)))  # batch 2
labels3 = np.arange(2 * num_elements, 3 * num_elements)
data3 = np.float32(np.random.random((num_elements, dim)))  # batch 3

# Declaring index
hnsw_index = hnswlib.Index(space='l2', dim=dim)

# Initiating index
# max_elements - the maximum number of elements, should be known beforehand
#     (probably will be made optional in the future)
#
# ef_construction - controls index search speed/build speed tradeoff
# M - is tightly connected with internal dimensionality of the data
#     strongly affects the memory consumption

# Enable replacing of deleted elements
hnsw_index.init_index(max_elements=max_num_elements, ef_construction=200, M=16, allow_replace_deleted=True)

# Controlling the recall by setting ef:
# higher ef leads to better accuracy, but slower search
hnsw_index.set_ef(10)

# Set number of threads used during batch search/construction
# By default using all available cores
hnsw_index.set_num_threads(4)

# Add batch 1 and 2 data
hnsw_index.add_items(data1, labels1)
hnsw_index.add_items(data2, labels2)  # Note: maximum number of elements is reached

# Delete data of batch 2
for label in labels2:
    hnsw_index.mark_deleted(label)

# Replace deleted elements
# Maximum number of elements is reached therefore we cannot add new items,
# but we can replace the deleted ones by using replace_deleted=True
hnsw_index.add_items(data3, labels3, replace_deleted=True)
# hnsw_index contains the data of batch 1 and batch 3 only
```