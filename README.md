# Hnswlib - fast approximate nearest neighbor search
Header-only C++ HNSW implementation with python bindings. Paper code for the HNSW 200M SIFT experiment

Highlights:
1) Lightweight, header-only, no dependences other than C++ 11.
2) Interfaces for C++, python and R (https://github.com/jlmelville/rcpphnsw).
3) Has full support for incremental index construction.
4) Can work with custom user distances (C++). 
5) Significantly less memory footprint and faster build time compared to current nmslib's implementation, although is slightly slower in terms of the search speed.

Description of the algroithm's parameters can be found in [ALGO_PARAMS.md](ALGO_PARAMS.md).


### Python bindings

#### Supported distances:

| Distance         | parameter       | Equation                |
| -------------    |:---------------:| -----------------------:|
|Squared L2        |'l2'             | d = sum((Ai-Bi)^2)      |
|Inner product     |'ip'             | d = 1.0 - sum(Ai\*Bi))  |
|Cosine similarity |'cosine'         | d = 1.0 - sum(Ai\*Bi) / sqrt(sum(Ai\*Ai) * sum(Bi\*Bi))|

Note that inner product is not a metric. An element can be closer to some other element than to itself.

For other spaces use the nmslib library https://github.com/nmslib/nmslib. 

#### short API description
* ```hnswlib.Index(space, dim)``` creates a non-initalized index an HNSW in space ```space``` with integer dimension ```dim```.

Index methods:
* ```init_index(max_elements, ef_construction = 200, M = 16, random_seed = 100)``` initalizes the index from with no elements. 
    * ```max_elements``` defines the maximum number of elements that can be stored in the structure(can be increased/shrunk via saving/loading).
    * ```ef_construction``` defines a construcion time/accuracy tradeoff (see [ALGO_PARAMS.md](ALGO_PARAMS.md)).
    * ```M``` defines tha maximum number of outgoing connections in the graph ([ALGO_PARAMS.md](ALGO_PARAMS.md)).
* ```add_items(data, data_labels, num_threads = -1)``` - inserts the ```data```(numpy array of vectors, shape:```N*dim```) into the structure. 
    * ```labels``` is an optional N-size numpy array of integer labels for all elements in ```data```.
    * ```num_threads``` sets the number of cpu threads to use (-1 means use default).
* ```set_ef(ef)``` - sets the query time accuracy/speed tradeoff, defined by the ```ef``` parameter (
[ALGO_PARAMS.md](ALGO_PARAMS.md)).
* ```knn_query(data, k = 1, num_threads = -1)``` make a batch query for ```k``` closests elements for each element of the 
    * ```data``` (shape:```N*dim```). Returns a numpy array of (shape:```N*k```).
    * ```num_threads``` sets the number of cpu threads to use (-1 means use default).
* ```load_index(path_to_index, max_elements = 0)``` loads the index from persistence to the unintialized index.
    * ```max_elements```(optional) resets the maximum number of elements in the structure.  
* ```save_index(path_to_index)``` saves the index from persistence.
* ```set_num_threads(num_threads)``` set the defualt number of cpu threads used during data insertion/querying.  
* ```get_items(ids)``` - returns a numpy array (shape:```N*dim```) of vectors that have integer identifiers specified in ```ids``` numpy vector (shape:```N```).  
* ```get_ids_list()```  - returns a list of all element ids.

   
        
        
  
#### Python bindings examples
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
p.add_items(data, data_labels)

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

# We split the data in two batches:
data1 = data[:num_elements // 2]
data2 = data[num_elements // 2:]

# Declaring index
p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

# Initing index
# max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
# during insertion of an element.
# The capacity can be increased by saving/loading the index, see below.
#
# ef_construction - controls index search speed/build speed tradeoff
#
# M - is tightly connected with internal dimensionality of the data. Strongly affects the memory consumption (~M)
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

# Reiniting, loading the index
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

### Bindings installation
```bash
apt-get install -y python-setuptools python-pip
pip3 install pybind11 numpy setuptools
cd python_bindings
python3 setup.py install
```

### Other implementations
* Non-metric space library (nmslib) - main library(python, C++), supports exotic distances: https://github.com/nmslib/nmslib
* Faiss libary by facebook, uses own HNSW  implemenation for coarse quatization (python, C++):
https://github.com/facebookresearch/faiss
* Code for the paper 
["Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors"](https://arxiv.org/abs/1802.02422) 
(current state-of-the-art in compressed indexes, C++):
https://github.com/dbaranchuk/ivf-hnsw
* TOROS N2 (python, C++): https://github.com/kakao/n2 
* Online HNSW (C++): https://github.com/andrusha97/online-hnsw) 
* Go implementation: https://github.com/Bithack/go-hnsw
* Python implementation (as a part of the clustring code by by Matteo Dell'Amico): https://github.com/matteodellamico/flexible-clustering 


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

### HNSW example demos

- Visual search engine for 1M amazon products (MXNet + HNSW): [website](https://thomasdelteil.github.io/VisualSearch_MXNet/), [code](https://github.com/ThomasDelteil/VisualSearch_MXNet), demo by [@ThomasDelteil](https://github.com/ThomasDelteil)

### References

Malkov, Yu A., and D. A. Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." arXiv preprint arXiv:1603.09320 (2016). https://arxiv.org/abs/1603.09320
