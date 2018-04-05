# HNSW - Approximate nearest neighbor search
Header-only C++ HNSW implementation with python bindings. Paper code for the HNSW 200M SIFT experiment

NEW: Added support for cosine similarity and inner product distances


Part of the nmslib project https://github.com/nmslib/nmslib

### Python bindings


#### Supported distances:
1) Squared L2 ('l2', d = sum((Ai-Bi)^2)
2) Inner product ('ip', d = 1.0 - sum(Ai\*Bi))
3) Cosine similarity ('cosine', d = 1.0 - sum(Ai\*Bi)/sqrt(sum(Ai\*Ai)*sum(Bi\*Bi))))

Note that inner product is not a metric. An element can be closer to some other element than to itself.

For other spaces use the main library https://github.com/nmslib/nmslib 


#### Python bindings example
```python
import hnswlib
import numpy as np

dim = 128
num_elements = 10000

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))
data_labels = np.arange(num_elements)

# Declaring index
p = hnswlib.Index(space = 'l2', dim = dim) # possible options are l2, cosine or ip

# Initing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = num_elements, ef_construction = 200, M = 16)

# Element insertion (can be called several times):
int_labels = p.add_items(data, data_labels)

# Controlling the recall by setting ef:
p.set_ef(50) # ef should always be > k

# Query dataset, k - number of closest elements (returns 2 numpy arrays)
labels, distances = p.knn_query(data, k = 1)

```
An example with updates after serialization/deserialization:
```python
import hnswlib
import numpy as np

dim = 16
num_elements = 10000

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))

# Declaring index
p = hnswlib.Index(space = 'l2', dim = dim) # possible options are l2, cosine or ip

# Initing index - the maximum number of elements should be known beforehand
# Probably this will become an optional parameter in the future
p.init_index(max_elements = num_elements, ef_construction = 100, M = 16)

# Controlling the recall by setting ef:
p.set_ef(10)
p.set_num_threads(4) # by default using all available cores


# We split the data in two batches:
data1 = data[:num_elements//2]
data2 = data[num_elements//2:]

# We first insert the element from the first batch:
print("Adding first batch of %d elements" % (len(data1)))
p.add_items(data1)

# Query the elements for themselves and measure recall:
labels, distances = p.knn_query(data1, k = 1)
print("Recall for the first batch:",np.mean(labels.reshape(-1)==np.arange(len(data1))),"\n")

# Serialize index and delete it:
print("Saving to first_half.bin")
p.save_index("first_half.bin")
del p

# Reinit the index, deserialize it (note that the data is loaded automitically)
p = hnswlib.Index(space = 'l2', dim = dim) # you can change the sa

print("Loading from first_half.bin\n")
p.load_index("first_half.bin")


print("Adding the second batch of %d elements" % (len(data2)))
p.add_items(data2)
labels, distances = p.knn_query(data, k = 1)
labels=labels.reshape(-1)

# Query the elements for themselves and measure recall:
print("Recall for two batches:",np.mean(labels==np.arange(len(data))),"\n")
```

#### Bindings installation
```bash
pip3 install pybind11 numpy setuptools
cd python_bindings
python3 setup.py install
```

### 200M SIFT test reproduction 
To download and extract the bigann dataset:
```bash
python3 download_bigann.py
```
To compile:
```bash
cmake .
make all
```

To run the test on 200M SIFT subset:
```bash
./main
```

The size of the bigann subset (in millions) is controlled by the variable **subset_size_milllions** hardcoded in **sift_1b.cpp**.



References:
Malkov, Yu A., and D. A. Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." arXiv preprint arXiv:1603.09320 (2016). https://arxiv.org/abs/1603.09320
