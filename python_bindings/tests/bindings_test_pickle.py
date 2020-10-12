import unittest

import numpy as np


def get_dist(metric, pt1, pt2):
    if metric == 'l2':
        return np.sum((pt1-pt2)**2)
    elif metric == 'ip':
        return 1. - np.sum(np.multiply(pt1,pt2))
    elif metric == 'cosine':
        return 1. - np.sum(np.multiply(pt1,pt2)) / (np.sum(pt1**2) * np.sum(pt2**2))**.5

def brute_force_distances(metric, items, query_items, k):
    dists=np.zeros((query_items.shape[0], items.shape[0]))
    for ii in range(items.shape[0]):
        for jj in range(query_items.shape[0]):
            dists[jj,ii]=get_dist(metric, items[ii, :], query_items[jj, :])

    labels = np.argsort(dists, axis=1)
    dists = np.sort(dists, axis=1)


    return labels[:,:k], dists[:,:k]


class PickleSelfTestCase(unittest.TestCase):

    def check_ann_results(self, metric, items, query_items, k, ann_l, ann_d, err_thresh=0, total_thresh=0, dists_thresh=0):
        brute_l, brute_d = brute_force_distances(metric, items, query_items, k)
        err_total = 0
        for jj in range(query_items.shape[0]):
            err = np.sum(np.isin(brute_l[jj, :], ann_l[jj, :], invert=True))
            if err > 0:
                print(f"Warning: {err} labels are missing from ann results (k={k}, err_thresh={err_thresh})")

            if err > err_thresh:
                err_total += 1
                
        self.assertLessEqual( err_total, total_thresh, f"Error: knn_query returned incorrect labels for {err_total} items (k={k})")

        wrong_dists=np.sum(((brute_d- ann_d)**2.)>1e-3)
        if wrong_dists > 0:
            dists_count=brute_d.shape[0]*brute_d.shape[1]
            print(f"Warning: {wrong_dists} ann distance values are different from brute-force values (total # of values={dists_count}, dists_thresh={dists_thresh})")
            
        self.assertLessEqual( wrong_dists, dists_thresh, msg=f"Error: {wrong_dists} ann distance values are different from brute-force values")

    def testPickle(self):
        import hnswlib
        import pickle

        ef_construction = 725
        M = 64
        ef = 725

        num_elements = 5000
        num_test_elements = 100

        num_threads = 4
        k = 15
        
        label_err_thresh=5  ### max number of missing labels allowed per test item
        item_err_thresh=5   ### max number of items allowed with incorrect labels 

        dists_err_thresh=50 ### for two matrices, d1 and d2, dists_err_thresh controls max 
                            ### number of value pairs that are allowed to be different in d1 and d2
                            ### i.e., number of values that are (d1-d2)**2>1e-3

        for space,dim in [('ip', 48), ('l2', 152), ('cosine', 512)]:

            # Generating sample data
            data = np.float32(np.random.random((num_elements, dim)))
            test_data = np.float32(np.random.random((num_test_elements, dim)))

            # Declaring index
            p = hnswlib.Index(space=space, dim=dim)  # possible options are l2, cosine or ip
            print(f"Running pickle tests for {p}")

            p.num_threads=num_threads  # by default using all available cores

            p0=pickle.loads(pickle.dumps(p)) ### pickle un-initialized Index
            p.init_index(max_elements = num_elements, ef_construction = ef_construction, M = M)
            p0.init_index(max_elements = num_elements, ef_construction = ef_construction, M = M)

            p.ef=ef  ### Note: ef parameter can be set only after calling p.init_index,
            p0.ef=ef ###       so we have to set p0.ef 

            p1=pickle.loads(pickle.dumps(p)) ### pickle Index before adding items

            ### add items to ann index p,p0,p1 
            p.add_items(data)  
            p1.add_items(data) 
            p0.add_items(data)
            
            p2=pickle.loads(pickle.dumps(p)) ### pickle Index before adding items

            self.assertTrue(np.allclose(p.get_items(), p0.get_items()), "items for p and p0 must be same")
            self.assertTrue(np.allclose(p0.get_items(), p1.get_items()), "items for p0 and p1 must be same")
            self.assertTrue(np.allclose(p1.get_items(), p2.get_items()), "items for p1 and p2 must be same")

            ### Test if returned distances are same
            l, d   = p.knn_query(test_data, k=k)
            l0, d0 = p0.knn_query(test_data, k=k)
            l1, d1 = p1.knn_query(test_data, k=k)
            l2, d2 = p2.knn_query(test_data, k=k)

            self.assertLessEqual(np.sum(((d-d0)**2.)>1e-3), dists_err_thresh, msg=f"knn distances returned by p and p0 must match")
            self.assertLessEqual(np.sum(((d0-d1)**2.)>1e-3), dists_err_thresh, msg=f"knn distances returned by p0 and p1 must match")
            self.assertLessEqual(np.sum(((d1-d2)**2.)>1e-3), dists_err_thresh, msg=f"knn distances returned by p1 and p2 must match")

            ### check if ann results match brute-force search
            ###   allow for 2 labels to be missing from ann results
            self.check_ann_results(space, data, test_data, k, l, d, 
                                   err_thresh = label_err_thresh, 
                                   total_thresh = item_err_thresh, 
                                   dists_thresh = dists_err_thresh)
            
            self.check_ann_results(space, data, test_data, k, l2, d2, 
                                   err_thresh=label_err_thresh, 
                                   total_thresh=item_err_thresh, 
                                   dists_thresh=dists_err_thresh)

            ### Check ef parameter value
            self.assertEqual(p.ef, ef, "incorrect value of p.ef")
            self.assertEqual(p0.ef, ef, "incorrect value of p0.ef")
            self.assertEqual(p2.ef, ef, "incorrect value of p2.ef")
            self.assertEqual(p1.ef, ef, "incorrect value of p1.ef")

            ### Check M parameter value
            self.assertEqual(p.M, M, "incorrect value of p.M")
            self.assertEqual(p0.M, M, "incorrect value of p0.M")
            self.assertEqual(p1.M, M, "incorrect value of p1.M")
            self.assertEqual(p2.M, M, "incorrect value of p2.M")

            ### Check ef_construction parameter value
            self.assertEqual(p.ef_construction, ef_construction, "incorrect value of p.ef_construction")
            self.assertEqual(p0.ef_construction, ef_construction, "incorrect value of p0.ef_construction")
            self.assertEqual(p1.ef_construction, ef_construction, "incorrect value of p1.ef_construction")
            self.assertEqual(p2.ef_construction, ef_construction, "incorrect value of p2.ef_construction")


if __name__ == "__main__":
    unittest.main()
