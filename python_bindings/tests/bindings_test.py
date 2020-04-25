import unittest


class RandomSelfTestCase(unittest.TestCase):
    def testRandomSelf(self):
        import hnswlib
        import numpy as np
        for dim in [3,4,7,8,12,16,19,20,23,29,32,33]:
            for space in ['l2', 'cosine']:
                print("Testing dim=%d space=%s"%())
                
                num_elements = 10000

                # Generating sample data
                data = np.float32(np.random.random((num_elements, dim)))

                # Declaring index
                p = hnswlib.Index(space=space, dim=dim)  # possible options are l2, cosine or ip

                # Initing index
                # max_elements - the maximum number of elements, should be known beforehand
                #     (probably will be made optional in the future)
                #
                # ef_construction - controls index search speed/build speed tradeoff
                # M - is tightly connected with internal dimensionality of the data
                #     stronlgy affects the memory consumption

                p.init_index(max_elements = num_elements, ef_construction = 150, M = 16)

                # Controlling the recall by setting ef:
                # higher ef leads to better accuracy, but slower search
                p.set_ef(30)

                p.set_num_threads(4)  # by default using all available cores

                # We split the data in two batches:
                data1 = data[:num_elements // 2]
                data2 = data[num_elements // 2:]

                print("Adding first batch of %d elements" % (len(data1)))
                p.add_items(data1)

                # Query the elements for themselves and measure recall:
                labels, distances = p.knn_query(data1, k=1)
                self.assertAlmostEqual(np.mean(labels.reshape(-1) == np.arange(len(data1))),1.0,3)

                # Serializing and deleting the index:
                index_path='first_half.bin'
                print("Saving index to '%s'" % index_path)
                p.save_index("first_half.bin")
                del p

                # Reiniting, loading the index
                p = hnswlib.Index(space='l2', dim=dim)  # you can change the sa

                print("\nLoading index from 'first_half.bin'\n")
                p.load_index("first_half.bin")

                print("Adding the second batch of %d elements" % (len(data2)))
                p.add_items(data2)

                # Query the elements for themselves and measure recall:
                labels, distances = p.knn_query(data, k=1)

                self.assertAlmostEqual(np.mean(labels.reshape(-1) == np.arange(len(data))),1.0,3)


if __name__ == "__main__":
    unittest.main()
