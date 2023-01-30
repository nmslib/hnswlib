import hnswlib
import numpy as np


"""
Example of replacing deleted elements with new ones
"""

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
