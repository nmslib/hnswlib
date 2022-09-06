import unittest

import numpy as np

import hnswlib


def get_data():
    data1 = np.asarray([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])
    return data1


def get_tag():
    tags1 = [1, 1, 0, 0, 0]
    tags2 = [1, 0, 0, 0, 1]
    return tags1, tags2


def test_no_tag(self):
    # result in order [[4 3 0 1 2]]
    data1 = get_data()
    for space, expected_distances in [
        ('l2', [[0., 1., 2., 2., 2.]]),
        ('ip', [[-2., -1., 0., 0., 0.]]),
        ('cosine', [[0, 1.835e-01, 4.23e-01, 4.23e-01, 4.23e-01]])]:
        data2 = np.concatenate([np.zeros([data1.shape[0], 5]), data1, np.zeros([data1.shape[0], 5])], axis=1)

        p = hnswlib.PFIndex(space=space, dim=data2.shape[1])
        p.init_index(max_elements=5, ef=100, M=16)
        p.add(data2)

        # test space
        filterList = hnswlib.QueryFilterList()
        labels, distances = p.knn_query(data=np.asarray(data2[-1:]), k=5, query_list=filterList)
        diff = np.mean(np.abs(distances - expected_distances))
        self.assertAlmostEqual(diff, 0, delta=1e-3)


def test_tag_true(self):
    # result in order [[0 1]]
    data1 = get_data()
    tag1, tag2 = get_tag()

    data2 = np.concatenate([np.zeros([data1.shape[0], 5]), data1, np.zeros([data1.shape[0], 5])], axis=1)

    p = hnswlib.PFIndex(space="ip", dim=data2.shape[1])
    p.init_index(max_elements=5, ef=100, M=16)
    p.add(data2)

    for id, tag in enumerate(tag1):
        p.add_field_tag(id, "tag1", tag)

    for id, tag in enumerate(tag2):
        p.add_field_tag(id, "tag2", tag)

    # filter tag1 = 1 -> [[0, 1]]
    filterList = hnswlib.QueryFilterList()
    queryFilter = hnswlib.QueryFilter("tag1")
    queryFilter.add(1, True)
    filterList.add(queryFilter)

    labels, distances = p.knn_query(data=np.asarray(data2[-1:]), k=5, query_list=filterList)
    diff = np.mean(np.abs(distances - [[0., 0.]]))
    self.assertAlmostEqual(diff, 0, delta=1e-3)

    # filter id = 1 -> [[0]]
    idFilter = hnswlib.QueryFilter("ID")
    idFilter.add(1, False)
    # &
    filterList.add(idFilter)

    labels, distances = p.knn_query(data=np.asarray(data2[-1:]), k=5, query_list=filterList)
    diff = np.mean(np.abs(distances - [[0.]]))
    self.assertAlmostEqual(diff, 0, delta=1e-3)


def test_tag_false(self):
    # result in order [[0,1]]
    data1 = get_data()
    tag1, tag2 = get_tag()
    expected_distances = [[-2., -1., 0.]]

    data2 = np.concatenate([np.zeros([data1.shape[0], 5]), data1, np.zeros([data1.shape[0], 5])], axis=1)
    dim = data2.shape[1]
    p = hnswlib.PFIndex(space="ip", dim=dim)
    p.init_index(max_elements=5, ef=100, M=16)
    p.add(data2)

    for id, tag in enumerate(tag1):
        p.add_field_tag(id, "tag1", tag)

    for id, tag in enumerate(tag2):
        p.add_field_tag(id, "tag2", tag)

    # filter tag1 = 1
    filterList = hnswlib.QueryFilterList()
    queryFilter = hnswlib.QueryFilter("tag1")
    queryFilter.add(1, False)
    filterList.add(queryFilter)

    labels, distances = p.knn_query(data=np.asarray(data2[-1:]), k=5, query_list=filterList)
    diff = np.mean(np.abs(distances - expected_distances))
    self.assertAlmostEqual(diff, 0, delta=1e-3)


def test_save_load(self, dir):
    # result in order [[4 3 2]]
    data1 = get_data()
    tag1, tag2 = get_tag()

    data2 = np.concatenate([np.zeros([data1.shape[0], 5]), data1, np.zeros([data1.shape[0], 5])], axis=1)
    p = hnswlib.PFIndex(space="ip", dim=data2.shape[1])
    p.init_index(max_elements=5, ef=100, M=16)
    p.add(data2)
    for id, tag in enumerate(tag1):
        p.add_field_tag(id, "tag1", tag)
    for id, tag in enumerate(tag2):
        p.add_field_tag(id, "tag2", tag)

    # filter tag1 = 1
    expected_distances = [[0., 0.]]
    filterList = hnswlib.QueryFilterList()
    queryFilter = hnswlib.QueryFilter("tag1")
    queryFilter.add(1, True)
    filterList.add(queryFilter)

    labels, distances = p.knn_query(data=np.asarray(data2[-1:]), k=5, query_list=filterList)
    diff = np.mean(np.abs(distances - expected_distances))
    self.assertAlmostEqual(diff, 0, delta=1e-3)

    print("TAG saving index to '%s' ..." % dir)
    p.save_index(dir)
    del p

    p = hnswlib.PFIndex(space='ip', dim=data2.shape[1])  # you can change the sa
    print("TAG loading index from '%s'" % dir)
    p.load_index(dir, max_elements=5)
    # filter tag1 = 1
    expected_distances = [[0., 0.]]
    filterList = hnswlib.QueryFilterList()
    queryFilter = hnswlib.QueryFilter("tag1")
    queryFilter.add(1, True)
    filterList.add(queryFilter)

    labels, distances = p.knn_query(data=np.asarray(data2[-1:]), k=5, query_list=filterList)
    diff = np.mean(np.abs(distances - expected_distances))
    self.assertAlmostEqual(diff, 0, delta=1e-3)



class TagTestCase(unittest.TestCase):
    def runCase(self):
        # 1、test_space
        test_no_tag(self)
        print("TAG test_no_tag succuss")

        # 2、test_tag_true
        test_tag_true(self)
        print("TAG test_tag_true succuss")

        # 3、test_tag_false
        test_tag_false(self)
        print("TAG test_tag_false succuss")

        # 4、test_save_load
        test_save_load(self, "tag_idx")
        print("TAG test succuss")