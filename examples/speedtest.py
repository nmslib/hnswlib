import hnswlib
import numpy as np
import os.path
import time
import argparse

# Use nargs to specify how many arguments an option should take.
ap = argparse.ArgumentParser()
ap.add_argument('-d')
ap.add_argument('-n')
ap.add_argument('-t')
args = ap.parse_args()
dim = int(args.d)
name = args.n
threads=int(args.t)
num_elements = 1000000 * 4//dim

# Generating sample data
np.random.seed(1)
data = np.float32(np.random.random((num_elements, dim)))


index_path=f'speed_index{dim}.bin'
# Declaring index
p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

if not os.path.isfile(index_path) :

    p.init_index(max_elements=num_elements, ef_construction=100, M=16)

    # Controlling the recall by setting ef:
    # higher ef leads to better accuracy, but slower search
    p.set_ef(10)

    # Set number of threads used during batch search/construction
    # By default using all available cores
    p.set_num_threads(12)

    p.add_items(data)

    # Serializing and deleting the index:

    print("Saving index to '%s'" % index_path)
    p.save_index(index_path)
p.set_num_threads(threads)
times=[]
time.sleep(10)
p.set_ef(100)
for _ in range(3):
    p.load_index(index_path)
    for _ in range(10):
        t0=time.time()
        labels, distances = p.knn_query(data, k=1)
        tt=time.time()-t0
        times.append(tt)
        print(f"{tt} seconds")    
str_out=f"mean time:{np.mean(times)}, median time:{np.median(times)}, std time {np.std(times)} {name}"
print(str_out)
with open (f"log_{dim}_t{threads}.txt","a") as f:
    f.write(str_out+"\n")
    f.flush()

