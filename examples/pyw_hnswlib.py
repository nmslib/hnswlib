import hnswlib
import numpy as np
import threading
import pickle


class Index():
    def __init__(self, space, dim):
        self.index = hnswlib.Index(space, dim)
        self.lock = threading.Lock()
        self.dict_labels = {}
        self.cur_ind = 0

    def init_index(self, max_elements, ef_construction = 200, M = 16):
        self.index.init_index(max_elements = max_elements, ef_construction = ef_construction, M = M)

    def add_items(self, data, ids=None):
        if ids is not None:
            assert len(data) == len(ids)
        num_added = len(data)
        with self.lock:
            start = self.cur_ind
            self.cur_ind += num_added
        int_labels = []

        if ids is not None:
            for dl in ids:
                int_labels.append(start)
                self.dict_labels[start] = dl
                start += 1
        else:
            for _ in range(len(data)):
                int_labels.append(start)
                self.dict_labels[start] = start
                start += 1
        self.index.add_items(data=data, ids=np.asarray(int_labels))

    def set_ef(self, ef):
        self.index.set_ef(ef)

    def load_index(self, path):
        self.index.load_index(path)
        with open(path + ".pkl", "rb") as f:
            self.cur_ind, self.dict_labels = pickle.load(f)

    def save_index(self, path):
        self.index.save_index(path)
        with open(path + ".pkl", "wb") as f:
            pickle.dump((self.cur_ind, self.dict_labels), f)

    def set_num_threads(self, num_threads):
        self.index.set_num_threads(num_threads)

    def knn_query(self, data, k=1):
        labels_int, distances = self.index.knn_query(data=data, k=k)
        labels = []
        for li in labels_int:
            line = []
            for l in li:
                line.append(self.dict_labels[l])
            labels.append(line)
        return labels, distances
