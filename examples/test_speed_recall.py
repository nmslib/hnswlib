import hnswlib
import numpy as np
import time
dim = 24
# M=96
num_elements = 1000_000

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))
data = data / np.linalg.norm(data,axis=1,keepdims=True)


data_q = np.float32(np.random.random((20000, dim)))
data_q = data_q / np.linalg.norm(data_q,axis=1,keepdims=True)


t0=time.time()
gts=[]
batch=4000
batch_idx=0
while batch_idx<len(data_q):
    dist=np.matmul(data_q[batch_idx:batch_idx+batch],data.T)        
    gts.append(np.argmax(dist, axis=1).reshape(-1))
    batch_idx+=batch
gt=np.concatenate(gts,axis=0)
print("brute force time:",1000*(time.time()-t0)/len(data_q))
with open(f"logd{dim}_time.txt","w") as f:
 with open(f"logd{dim}_hops.txt","w") as f_h:
  with open(f"logd{dim}_dists.txt","w") as f_d:
   for M in [8,16,24,32,48,96]:
    for alpha in [0.9, 0.95, 1, 1.05,1.1, 1.2, 1.3]:
        # Declaring index
        p = hnswlib.Index(space='ip', dim=dim)  # possible options are l2, cosine or ip
        
        # Initing index
        # max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
        # during insertion of an element.
        # The capacity can be increased by saving/loading the index, see below.
        #
        # ef_construction - controls index search speed/build speed tradeoff
        #
        # M - is tightly connected with internal dimensionality of the data. Strongly affects the memory consumption (~M)
        # Higher M leads to higher accuracy/run_time at fixed ef/efConstruction

        p.init_index(max_elements=num_elements, ef_construction=1000, M=M)

        p.set_alpha(alpha)

        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search


        # Set number of threads used during batch search/construction
        # By default using all available cores
        

        p.set_num_threads(24)
        # print("Adding first batch of %d elements" % (len(data)))
        p.add_items(data)

        # Query the elements for themselves and measure recall:

        


        p.set_num_threads(24)
        times=[]
        recalls=[]
        hops=[]
        dist_comps=[]
        target_recalls=[0.5,0.9, 0.95, 0.99]
        print(f" ---- alpha={alpha} ---- ")
        for ef in [1,2,3,4,5,6,7,8,9,10,12,15,17,20,22,25,27,30,35,40,45,50,55,60,65,70,75,80,85, 90,100,110,120,130,140,150,160,170,180,190, 200,
        250,260,270,280,290,300,320,350,370,400,500,600,700,800,900,1000,1100,1200,1400,1600,1800,2000, 2200,2400]:
            p.set_ef(ef)
            p.reset_metrics_computations()            
            t0=time.time()
            labels, distances = p.knn_query(data_q, k=1)    
            time1=1000*(time.time()-t0)/len(data_q)

            recall=np.mean(labels.reshape(-1) == gt.reshape(-1))

            dist_comp=p.get_distance_computations()/len(data_q)           
            hop=p.get_hops()/len(data_q)

            # print(ef, recall, time1, dist_comp, hop)
            recalls.append(recall)
            times.append(time1)
            hops.append(hop)
            dist_comps.append(dist_comp)

        
            if recall>np.max(target_recalls):
                break
        # print(target_recalls, times)
        times_at_recall = np.interp (target_recalls, recalls,times, right=100000)
        dist_comp_at_recall = np.interp (target_recalls, recalls, dist_comps, right=100000)
        hops_at_recall = np.interp (target_recalls, recalls,hops, right=100000)

        def write_to_log(f, data, metric):
            f.write(f"{alpha} {M} ")
            for idx in range(len(target_recalls)):
                print(f"recall: {target_recalls[idx]}, {data[idx]:2.3f} {metric}")
                f.write(f"{data[idx]:1.3f} ")
            f.write("\n")
            f.flush()

        write_to_log(f,times_at_recall, "ms")
        print("Dist")
        write_to_log(f_d,dist_comp_at_recall, "distance computations")
        print("Hops:")
        write_to_log(f_h,hops_at_recall, "hops")
        

        

        print()
