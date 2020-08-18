


# HNSWLIB - Fast Approximate Nearest Neighbor Search

Hnswlib is a C++ library with Python bindings for highly performant implementation of [HNSW](https://arxiv.org/abs/1603.09320) *(Hierarchical Navigable Small World Graphs)* algorithm to perform fast and efficient vector similarity search in high dimensional spaces . It achieves state-of-the-art performance on diverse datasets and one of the top-most leaders in ANN performance benchmarks as show in *[ann-benchmarks.com](http://ann-benchmarks.com)*. 
HNSW algorithm is being leveraged globally for performing fast and efficient similarity search. Some public examples for the usage in the industry are  *[Facebook](https://github.com/facebookresearch/faiss)*,  *[Twitter](link)*,  *[Pinterest](https://arxiv.org/pdf/2007.03634.pdf)*,  *[Amazon](https://docs.aws.amazon.com/sagemaker/latest/dg/k-nearest-neighbors.html)*,  *[Microsoft](https://github.com/microsoft/HNSW.Net)* and *[Open Distro](https://opendistro.github.io/for-elasticsearch/blog/odfe-updates/2020/04/Building-k-Nearest-Neighbor-(k-NN)-Similarity-Search-Engine-with-Elasticsearch)* . 


### News

* *Thanks to Apoorv Sharma [@apoorv-sharma](https://github.com/apoorv-sharma), hnswlib now supports true element updates i.e feature vectors can be updated incrementally for elements without rebuilding the index from scratch (Interface remains the same as element insertion)*

* *Thanks to Dmitry [@2ooom](https://github.com/2ooom), hnswlib got a boost in performance for vector dimensions that are not multiple of 4.* 

* *Thanks to Louis Abraham ([@louisabraham](https://github.com/louisabraham)) hnswlib can now be installed via pip!*

### Highlights

1) Very Lightweight, header-only and no dependencies other than C++ 11.
2) Works well for both low and high dimensional datasets.
3) It belongs to unrestricted memory ANN that allow the vectors to be stored in memory. No bound on RAM allows the best performance in terms of speed and resulting accuracy. 
4) Interfaces and bindings for C++, Python. External bindings for [R](https://github.com/jlmelville/rcpphnsw) and [Java](https://github.com/stepstone-tech/hnswlib-jna) contributed by community.
5) Other external implementation of the algorithm available in diverse languages like .Net, Go, Java, Python, Rust, Julia etc. Refer this [section](#other-implementations) for more details. 
6) Significantly less memory footprint and faster build time compared to current NMSLIB's implementation.


### Supported Operations

1) Supports batch(offline) and realtime(online) index.
2) Supports multi-threaded incremental query, insert, update and deletion of vectors.
3) Highly performant and efficient locking implementation to support multi-threaded reads and writes in parallel i.e multi-threaded query/insert/update/delete in parallel (Currently exposed in C++ version only).
4) Supports efficient serialization and deserialization of index to/from disk.
5) Can support user defined arbitrary and exotic similarity metrics like Hyperbolic distances (Poincare/Lorentzian), Jaccard distance, Manhattan distance etc. (In C++ version)

*Note: Currently deletions of elements from the index does not free the associated memory of the  vectors to be deleted and performance of read/writes scales with number of cpu cores in machine.*

### Installation
It can be installed from sources:
```bash
apt-get install -y python-setuptools python-pip
pip3 install pybind11 numpy setuptools
cd python_bindings
python3 setup.py install
```

or it can be installed via pip:
`pip install hnswlib`

### Python code example
***Example*** : Perform in-memory query, inserts, updates, deletes and serialization/deserialization of the index. For algorithm construction and runtime parameter details please refer [Params](ALGO_PARAMS.md).

```python
import hnswlib
import numpy as np

################## Declaring and Initializing index ##################
dim = 128
num_elements = 10000
index = hnswlib.Index(space = 'l2', dim = dim) # possible space options: [l2/cosine/ip]
# Set number of threads used during batch search/construction
# By default using all available cores
index.set_num_threads(4)
# Initing index - the maximum number of elements should be known beforehand
index.init_index(max_elements = num_elements, ef_construction = 200, M = 16)

################## Perform elements insertion ##################
data = np.float32(np.random.random((num_elements, dim))) # Generating sample data
data_labels = np.arange(num_elements)
index.add_items(data, data_labels) # Element insertion (can be called several times)

################## Perform element feature vector updates ##################
element_labels_to_update = data_labels[0] # Update feature vector for first element
element_updated_vectors = np.float32(np.random.random((1, dim))) # Corresponding updated vector
index.add_items(element_labels_to_update, element_updated_vectors) # Perform update

################## Perform element deletion ##################
element_labels_to_delete = data_label[0:2] # Delete first two elements
index.mark_deleted(element_labels_to_delete)

################## Perform nearest neighbor querying ##################
index.set_ef(50) # Controlling the recall by setting ef, it should always be > k
# Query dataset, k - number of closest elements (returns 2 numpy arrays)
labels, distances = index.knn_query(data, k = 1)

################## Serializing the index to disk ##################
index_path='first_half.bin'
index.save_index("first_half.bin")
del index

################## Loading the index from disk and increase the capacity of the index ##################
index = hnswlib.Index(space='l2', dim=dim)  # the space can be changed - keeps the data, alters the distance function.
# If required, total capacity of the index can be increased while loading the index using `max_elements`, so that it will be able to handle insertion of new data
index.load_index(index_path, max_elements =  2 * num_elements) # Increase capacity of the index by 2x
new_data = np.float32(np.random.random((num_elements, dim))) # Generate new sample data
index.add_items(new_data) # Add new data to index

```

#### Supported distances:

| Distance         | parameter       | Equation                |
| -------------    |:---------------:| -----------------------:|
|Squared L2        |'l2'             | d = sum((Ai-Bi)^2)      |
|Inner product     |'ip'             | d = 1.0 - sum(Ai\*Bi)   |
|Cosine similarity |'cosine'         | d = 1.0 - sum(Ai\*Bi) / sqrt(sum(Ai\*Ai) * sum(Bi\*Bi))|

*Note: Inner product is not an actual distance metric. An element can be closer to some other element than to itself. That allows some speedup if you remove all elements that are not the closest to themselves from the index.*

For other spaces [nmslib](https://github.com/nmslib/nmslib) library can be used or user can define their own distance/similarity metric as part of C++ version of Hnswlib library.

#### Short Python API description
* `hnswlib.Index(space, dim)` creates a non-initialized HNSW index in space `space` with integer dimension `dim`.
* `init_index(max_elements, ef_construction = 200, M = 16, random_seed = 100)` initializes the index from with no elements. 
    * `max_elements` defines the maximum number of elements that can be stored in the structure(can be increased/shrunk). Will throw an exception if exceeded during insertion of an element. The capacity can be increased by saving/loading the index as specified in the above python example.
    * `ef_construction` defines a construction time/accuracy trade-off (See [Params](ALGO_PARAMS.md)).
    * `M` defines the maximum number of outgoing connections in the graph (See [Params](ALGO_PARAMS.md)).
    
* `add_items(data, data_labels, num_threads = -1)` - **inserts/updates** the `data`(numpy array of vectors, shape:`N*dim`) into the structure. 
    * `labels` is an optional N-size numpy array of integer labels for all elements in `data`.
    * `num_threads` sets the number of cpu threads to use (-1 means use default).
    * `data_labels` specifies the labels for the data. If index already has the elements with the same labels, their feature vectors will be updated. Note that update procedure is slower than insertion of a new element, but more memory and query-efficient.
    * Thread-safe with other `add_items` calls, but not with `knn_query`.
    
* `mark_deleted(data_label)`  - marks the element as **deleted**, so it will be omitted from search results.

* `resize_index(new_size)` - changes the maximum capacity of the index. Not thread safe with `add_items` and `knn_query`.

* `set_ef(ef)` - sets the runtime query accuracy/speed trade-off, defined by the `ef` parameter (See [Params](ALGO_PARAMS.md)). Note that the parameter is currently not saved along with the index, so you need to set it manually after loading (The value does not need to be the same. User can control it according to their performance needs).

* `knn_query(data, k = 1, num_threads = -1)` a batch query for `k` nearest neighbor for each element of the 
    * `data` (shape:`N*dim`). Returns a numpy array of (shape:`N*k`).
    * `num_threads` sets the number of cpu threads to use (-1 means use default).
    * Thread-safe with other `knn_query` calls, but not with `add_items`.
    
* `load_index(path_to_index, max_elements = 0)` loads the index from persistence to the uninitialized index.
    * `max_elements`(optional) resets the maximum number of elements in the structure.
      
* `save_index(path_to_index)` saves the index from persistence.

* `set_num_threads(num_threads)` set the default number of cpu threads used during data insertion/querying.
  
* `get_items(ids)` - returns a numpy array (shape:`N*dim`) of vectors that have integer identifiers specified in `ids` numpy vector (shape:`N`). Note that for cosine similarity it currently returns **normalized** vectors.
  
* `get_ids_list()`  - returns a list of all elements' ids.

* `get_max_elements()` - returns the current capacity of the index

* `get_current_count()` - returns the current number of element stored in the index

### Tests
To reproduce performance benchmark results on 200M SIFT dataset as described in HNSW paper or to run tests for feature vector updates please refer [Tests](TESTS.md).

### Authors and Contributors

- [Yury Malkov](https://github.com/yurymalkov) is the lead author and developer of the HNSW algorithm and Hnswlib library.
- [Apoorv Sharma](https://github.com/apoorv-sharma)  co-authored an algorithm for performing dynamic updates of feature vectors in HNSW with [Yury Malkov](https://github.com/yurymalkov) and implemented it in HnswLib.
- [User2](https://github.com/user) User2 implemented delete...
- [User3](https://github.com/user) User3....


### Contributing to the repository and HNSW Community
Contributions are highly welcome! 
Please make pull requests against the `develop` branch. 
Please feel free to ask questions, report bugs and raise new feature requests at [issues page](https://github.com/nmslib/hnswlib/issues) of the repository.

### Other implementations
* Non-metric space library (nmslib) - main library(python, C++), supports exotic distances: https://github.com/nmslib/nmslib
* Faiss library by facebook, uses own HNSW  implementation for coarse quantization (python, C++):
https://github.com/facebookresearch/faiss
* Code for the paper 
["Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors"](https://arxiv.org/abs/1802.02422) 
(current state-of-the-art in compressed indexes, C++):
https://github.com/dbaranchuk/ivf-hnsw
* TOROS N2 (python, C++): https://github.com/kakao/n2 
* Online HNSW (C++): https://github.com/andrusha97/online-hnsw) 
* Go implementation: https://github.com/Bithack/go-hnsw
* Python implementation (as a part of the clustering code by by Matteo Dell'Amico): https://github.com/matteodellamico/flexible-clustering
* Java implementation: https://github.com/jelmerk/hnswlib
* Java bindings using Java Native Access: https://github.com/stepstone-tech/hnswlib-jna
* .Net implementation:  https://github.com/microsoft/HNSW.Net
* Rust implementation: https://github.com/rust-cv/hnsw
* Julia implementation: https://juliapackages.com/p/hnsw

### HNSW example demos

- Visual search engine for 1M amazon products (MXNet + HNSW): [website](https://thomasdelteil.github.io/VisualSearch_MXNet/), [code](https://github.com/ThomasDelteil/VisualSearch_MXNet), demo by [@ThomasDelteil](https://github.com/ThomasDelteil)

### References

Reference to cite when you use HNSW or Hnswlib in a research paper:
```
@article{DBLP:journals/corr/MalkovY16,
  author    = {Yury A. Malkov and
               D. A. Yashunin},
  title     = {Efficient and robust approximate nearest neighbor search using Hierarchical
               Navigable Small World graphs},
  journal   = {CoRR},
  volume    = {abs/1603.09320},
  year      = {2016},
  url       = {http://arxiv.org/abs/1603.09320},
  archivePrefix = {arXiv},
  eprint    = {1603.09320},
  timestamp = {Mon, 13 Aug 2018 16:46:53 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/MalkovY16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
