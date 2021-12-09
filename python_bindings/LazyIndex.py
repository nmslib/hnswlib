import hnswlib
"""
    A python wrapper for lazy indexing, preserves the same api as hnswlib.Index but initializes the index only after adding items for the first time with `add_items`.
"""
class LazyIndex(hnswlib.Index):
    def __init__(self, space, dim,max_elements=1024, ef_construction=200, M=16):
        super().__init__(space, dim)
        self.init_max_elements=max_elements
        self.init_ef_construction=ef_construction
        self.init_M=M
    def init_index(self, max_elements=0,M=0,ef_construction=0):
        if max_elements>0:
            self.init_max_elements=max_elements
        if ef_construction>0:
            self.init_ef_construction=ef_construction
        if M>0:
            self.init_M=M
        super().init_index(self.init_max_elements, self.init_M, self.init_ef_construction)
    def add_items(self, data, ids=None, num_threads=-1):
        if self.max_elements==0:
            self.init_index()
        return super().add_items(data,ids, num_threads)
    def get_items(self, ids=None):
        if self.max_elements==0:
            return []
        return super().get_items(ids)
    def knn_query(self, data,k=1, num_threads=-1):
        if self.max_elements==0:
            return [], []
        return super().knn_query(data, k, num_threads)
    def resize_index(self, size):
        if self.max_elements==0:
            return self.init_index(size)
        else:
            return super().resize_index(size)
    def set_ef(self, ef):
        if self.max_elements==0:
            self.init_ef_construction=ef
            return
        super().set_ef(ef)
    def get_max_elements(self):
        return self.max_elements
    def get_current_count(self):
        return self.element_count
