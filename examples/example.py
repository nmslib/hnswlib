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





