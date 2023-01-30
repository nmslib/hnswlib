import hnswlib
import numpy as np


"""
Example of filtering elements when searching
"""

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
# Warning: search with a filter works slow in python in multithreaded mode, therefore we set num_threads=1
labels, distances = hnsw_index.knn_query(data, k=1, num_threads=1, filter=filter_function)
# labels contain only elements with even id
