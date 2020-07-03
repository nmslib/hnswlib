import unittest


class RandomSelfTestCase(unittest.TestCase):
    def testRandomSelf(self):
      for idx in range(16):
        print("\n**** Index save-load test ****\n")
        import hnswlib
        import numpy as np
        
        np.random.seed(idx)
        dim = 16
        num_elements = 10000

        # Generating sample data
        data = np.float32(np.random.random((num_elements, dim)))
        elabels = np.arange(len(data))

        # Declaring index
        p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

        # Initing index
        # max_elements - the maximum number of elements, should be known beforehand
        #     (probably will be made optional in the future)
        #
        # ef_construction - controls index search speed/build speed tradeoff
        # M - is tightly connected with internal dimensionality of the data
        #     stronlgy affects the memory consumption

        p.init_index(max_elements = num_elements, ef_construction = 100, M = 16)

        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search
        p.set_ef(100)

        p.set_num_threads(4)  # by default using all available cores

        # We split the data in two batches:
        data1 = data[:num_elements // 2]
        data2 = data[num_elements // 2:]
        elabels1 = elabels[:num_elements // 2]
        elabels2 = elabels[num_elements // 2:]

        print("Adding first batch of %d elements" % (len(data1)))
        p.add_items(data1, elabels1)

        # Query the elements for themselves and measure recall:
        print("Doing knn query")
        results = p.knn_query(data1, k=1)
        labels = np.array([i[0][1] for i in results])

        # Check the recall:
        self.assertAlmostEqual(np.mean(labels == elabels1),1.0,3)

        print("Getting items")
        items=p.get_items(labels)

        # Check that the returned element data is correct:
        diff_with_gt_labels=np.mean(np.abs(data1-items))
        self.assertAlmostEqual(diff_with_gt_labels, 0, delta = 1e-4)

        # Serializing and deleting the index.
        # We need the part to check that serialization is working properly.

        index_path='first_half.bin'
        print("Saving index to '%s'" % index_path)
        p.save_index("first_half.bin")
        print("Saved. Deleting...")
        del p
        print("Deleted")

        print("\n**** Mark delete test ****\n")
        # Reiniting, loading the index
        print("Reiniting")
        p = hnswlib.Index(space='l2', dim=dim)

        print("\nLoading index from 'first_half.bin'\n")
        p.load_index("first_half.bin", use_mmap=False)
        p.set_ef(100)

        print("Adding the second batch of %d elements" % (len(data2)))
        p.add_items(data2, elabels2)

        # Serializing and deleting the index:
        index_path = 'full.bin'
        print("Saving index to '%s'" % index_path)
        p.save_index(index_path)
        del p

        # Reiniting, loading the index
        p_mmap = hnswlib.Index(space='l2', dim=dim)  # you can change the sa

        print("\nLoading index from '%s' with use_mmap=True\n" % index_path)
        p_mmap.load_index(index_path, use_mmap=True)

        # Query the elements for themselves and measure recall:
        results = p_mmap.knn_query(data, k=1)
        labels = np.array([i[0][1] for i in results])

        # Check the recall:
        self.assertAlmostEqual(np.mean(labels == elabels),1.0,3)

        items=p_mmap.get_items(labels)

        # Check that the returned element data is correct:
        diff_with_gt_labels=np.mean(np.abs(data-items))
        self.assertAlmostEqual(diff_with_gt_labels, 0, delta = 1e-4) # deleting index.

        # Checking that all labels are returned correctly:
        sorted_labels=sorted(p_mmap.get_ids_list())
        self.assertEqual(np.sum(~np.asarray(sorted_labels)==np.asarray(range(num_elements))),0)

        # Reiniting, loading the index
        p = hnswlib.Index(space='l2', dim=dim)  # you can change the sa

        print("\nLoading index from '%s' with use_mmap=False\n" % index_path)
        p.load_index(index_path, use_mmap=False)

        print("Comparing loaded indexes\n")
        self.assertEqual(p_mmap.get_ef_construction(), p.get_ef_construction())
        self.assertEqual(p_mmap.get_M(), p.get_M())
        self.assertEqual(p_mmap.get_max_elements(), p.get_max_elements())
        self.assertEqual(p_mmap.get_current_count(), p.get_current_count())
        self.assertTrue(np.array_equal(p_mmap.get_ids_list(), p.get_ids_list()))
        self.assertTrue(np.array_equal(p_mmap.get_items(sorted_labels), p.get_items(sorted_labels)))

        print("Querying index\n")
        results = p.knn_query(data1, k=1)
        labels1 = np.array([i[0][1] for i in results])

        print("Deleting labels\n")
        for l in labels1:
            p.mark_deleted(l)

        results = p.knn_query(data2, k=1)
        labels2 = np.array([i[0][1] for i in results])
        items = p.get_items(labels2)
        diff_with_gt_labels=np.mean(np.abs(data2-items))
        self.assertAlmostEqual(diff_with_gt_labels, 0, delta = 1e-3) # console


        results = p.knn_query(data1, k=1)
        labels1_after = np.array([i[0][1] for i in results])
        for la in labels1_after:
            for lb in labels1:
                if la == lb:
                    self.assertTrue(False)
        print("All the data in data1 are removed")

        # checking saving/loading index with elements marked as deleted
        p.save_index("with_deleted.bin")
        p = hnswlib.Index(space='l2', dim=dim)
        p.load_index("with_deleted.bin", use_mmap=True)
        p.set_ef(100)

        results = p.knn_query(data1, k=1)
        labels1_after = np.array([i[0][1] for i in results])
        for la in labels1_after:
            for lb in labels1:
                if la == lb:
                    self.assertTrue(False)



if __name__ == "__main__":
    unittest.main()
