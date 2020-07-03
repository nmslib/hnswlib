#include <iostream>
#include <thread>
#include <atomic>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "hnswlib/hnswlib.h"

namespace py = pybind11;

typedef float src_t;
typedef float dist_t;
typedef size_t label_t;
typedef hnswlib::AlgorithmInterface<dist_t, label_t>::Neighbour neighbour_t;
typedef py::array_t<neighbour_t> result_t;

template<typename T>
class FeatureVectorsProcessor {
public:
    FeatureVectorsProcessor(py::object feature_vectors, size_t expected_dimensions) {
        auto data = py::array::ensure(feature_vectors);
        if (!data) {
            throw std::runtime_error("data must be a numpy array");
        }

        bool isArrayOfArray = data.dtype().kind() == 'O';
        if (isArrayOfArray) {
            //data is an array of array?
            if (data.ndim() != 1) {
                throw std::runtime_error("data outer array must be n*1");
            }

            rows_ = data.shape(0);
        } else {
            size_t features;
            if (data.ndim() == 2) {
                rows_ = data.shape(0);
                features = data.shape(1);
            }
            else if (data.ndim() == 1) {
                rows_ = 1;
                features = data.shape(0);
            } else {
                throw std::runtime_error("data must be a 1d/2d array");
            }

            if (features != expected_dimensions) {
                throw std::runtime_error("wrong dimensionality of the data vectors");
            }
        }

        if (rows_ == 0) {
            feature_vector_accessor_ = nullptr;
        } else if (isArrayOfArray) {
            feature_vector_accessor_ = get1DArrayAccessor(data, expected_dimensions);
        }
        else {
            feature_vector_accessor_ = get2DArrayAccessor(data);
        }
    }

    size_t getRows() {
        return rows_;
    }

    /*
     * replacement for the openmp '#pragma omp parallel for' directive
     * only handles a subset of functionality (no reductions etc)
     * Process ids from start (inclusive) to end (EXCLUSIVE)
     *
     * The method is borrowed from nmslib
     */
    template<class Function>
    void process(unsigned int num_threads, Function fn) const {
        if (rows_ == 0) return;

        py::gil_scoped_release l;

        if (num_threads == 1 || rows_ <= num_threads * 4) {
            // avoid using threads when the number of searches is small
            for (size_t id = 0; id < rows_; id++) {
                fn(id, 0, feature_vector_accessor_(id));
            }
        } else {
            std::vector<std::thread> threads;
            std::atomic<size_t> current(0);

            // keep track of exceptions in threads
            // https://stackoverflow.com/a/32428427/1713196
            std::exception_ptr lastException = nullptr;
            std::mutex lastExceptMutex;

            for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id) {
                threads.push_back(std::thread([&, thread_id] {
                    while (true) {
                        size_t id = current.fetch_add(1);

                        if ((id >= rows_)) {
                            break;
                        }

                        try {
                            fn(id, thread_id, feature_vector_accessor_(id));
                        } catch (...) {
                            std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                            lastException = std::current_exception();
                            /*
                             * This will work even when current is the largest value that
                             * size_t can fit, because fetch_add returns the previous value
                             * before the increment (what will result in overflow
                             * and produce 0 instead of current + 1).
                             */
                            current = rows_;
                            break;
                        }
                    }
                }));
            }
            for (auto &thread : threads) {
                thread.join();
            }
            if (lastException) {
                std::rethrow_exception(lastException);
            }
        }
    }

private:
    size_t rows_;
    std::function<const T* (size_t)> feature_vector_accessor_;

    std::function<const T* (size_t)> get2DArrayAccessor(const py::array &data) const {
        auto items = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(data);
        if (!items) {
            throw std::runtime_error("Unexpected type for data!");
        }

        return [items](size_t row) -> const T* {
            return items.data(row);
        };
    }

    std::function<const T* (size_t)> get1DArrayAccessor(const py::array &data, int expected_dimensions) const {
        std::vector<py::array_t<T>> feature_vectors;
        feature_vectors.reserve(rows_);

        for(size_t i = 0; i < rows_; ++i) {
            auto feature_vector = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(*(py::handle *) data.data(i));
            if (!feature_vector) {
                throw std::runtime_error("Unexpected type for data, expected array!");
            }
            if (feature_vector.ndim() != 1) {
                throw std::runtime_error("data inner array must be n*1");
            }
            if (feature_vector.shape(0) != expected_dimensions) {
                throw std::runtime_error("wrong dimensionality of the data vectors");
            }
            feature_vectors.push_back(feature_vector);
        }

        return [feature_vectors](size_t row) -> const T* {
            return feature_vectors[row].data();
        };
    }
};

class IndexWriter : public std::streambuf {
    py::object writable_;

public:
    IndexWriter(py::object writable) : writable_(writable) {
    }

    std::streamsize xsputn(const char* s, std::streamsize n) {
        py::buffer_info buffer(s, n, true);
        py::memoryview buffer1(buffer);
#if PY_MAJOR_VERSION == 2
        writable_(buffer1);
        return n;
#else
        return writable_(buffer1).cast<std::streamsize>();
#endif
    }
};

class Index {
public:
    Index(const std::string &space_name, const int dim) :
            space_name(space_name), dim(dim) {
        normalize=false;
        if(space_name=="l2") {
            l2space = new hnswlib::L2Space(dim);
            normalizer = NULL;
        }
        else if(space_name=="ip") {
            l2space = new hnswlib::InnerProductSpace(dim);
            normalizer = NULL;
        }
        else if(space_name=="cosine") {
            l2space = new hnswlib::InnerProductSpace(dim);
            normalize=true;
            normalizer = new hnswlib::FloatNormalizer(dim);
        } else {
            throw new std::runtime_error("Unknown space specified.");
        }
        appr_alg = NULL;
        index_inited = false;
        num_threads_default = std::thread::hardware_concurrency();
    }

    void init_new_index(const size_t maxElements, const size_t M, const size_t efConstruction, const size_t random_seed) {
        if (appr_alg) {
            throw new std::runtime_error("The index is already initiated.");
        }
        appr_alg = new hnswlib::HierarchicalNSW<dist_t, label_t>(l2space, maxElements, M, efConstruction, random_seed);
        index_inited = true;
    }

    void set_ef(size_t ef) {
        appr_alg->ef_ = ef;
    }

    size_t get_ef_construction() {
        return appr_alg->ef_construction_;
    }

    size_t get_M() {
        return appr_alg->M_;
    }

    void set_num_threads(int num_threads) {
        if (num_threads > 0) {
            this->num_threads_default = num_threads;
        }
    }

    void saveIndex(const std::string &path_to_index) {
        appr_alg->saveIndex(path_to_index);
    }

    void writeIndex(py::object write_function) {
        IndexWriter writer(write_function);

        std::ostream stream(&writer);
        appr_alg->saveIndex(stream);
    }

    void loadIndex(const std::string &path_to_index, bool use_mmap, size_t max_elements) {
        if (appr_alg) {
            std::cerr<<"Warning: Calling load_index for an already inited index. Old index is being deallocated.";
            delete appr_alg;
        }
        appr_alg = new hnswlib::HierarchicalNSW<dist_t, label_t>(l2space, path_to_index, use_mmap, max_elements);
    }

    void addItems(py::object feature_vectors,
                  py::array_t < label_t, py::array::c_style | py::array::forcecast > ids,
                  int num_threads = -1) {
        if (num_threads <= 0) {
            num_threads = num_threads_default;
        }

        if (normalize) {
            auto processor = FeatureVectorsProcessor<src_t>(feature_vectors, dim);
            std::vector<dist_t> norm_array(num_threads * dim);
            processor.process(num_threads, [&](size_t row, int thread_id, const src_t *input_vector) {
                dist_t *norm_vector = norm_array.data() + (thread_id * dim);
                normalizer->normalize_vector(input_vector, norm_vector);
                appr_alg->addPoint(norm_vector, ids.at(row));
            });
        } else {
            auto processor = FeatureVectorsProcessor<dist_t>(feature_vectors, dim);
            processor.process(num_threads, [&](size_t row, int thread_id, const dist_t *input_vector) {
                appr_alg->addPoint(input_vector, ids.at(row));
            });
        }
    }

    py::list getData(py::array_t<label_t, py::array::c_style | py::array::forcecast> &ids) {
        py::list features;

        int size = ids.size();
        for (int i = 0; i < size; i++) {
            auto data = appr_alg->getDataByLabel(ids.at(i));
            auto data_array = py::array_t<dist_t>(l2space->get_dimension(), data);
            py::detail::array_proxy(data_array.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;

            features.append(std::move(data_array));
        }
        return features;
    }

    py::array_t<label_t> getIdsList() const {
        if (appr_alg->has_deletions_) {
            auto size = appr_alg->label_lookup_.size();
            py::array_t<label_t> labels_array(size);

            label_t *labels = labels_array.mutable_data(0);
            for (auto kv : appr_alg->label_lookup_) {
                *labels = kv.first;
                labels++;
            }

            return labels_array;
        } else {
            auto labels_array = py::array_t<label_t>(
                {appr_alg->cur_element_count_},
                {appr_alg->size_data_per_element_},
                appr_alg->getExternalLabelP(0));
            py::detail::array_proxy(labels_array.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;

            return labels_array;
        }
    }

    result_t knnQuery(py::object feature_vectors, size_t k = 1, int num_threads = -1) {
        k = std::min(k, appr_alg->cur_element_count_);
        if (k == 0) return result_t();

        if (num_threads <= 0)
            num_threads = num_threads_default;

        if (normalize) {
            auto processor = FeatureVectorsProcessor<src_t>(feature_vectors, dim);
            result_t all_results = getKnnResultArray(processor.getRows(), k);
            std::vector<dist_t> norm_array(num_threads*dim);

            processor.process(num_threads, [&](size_t row, int thread_id, const src_t *input_vector) {
                dist_t *norm_vector = norm_array.data() + (thread_id * dim);
                normalizer->normalize_vector(input_vector, norm_vector);
                appr_alg->searchKnn(norm_vector, k, all_results.mutable_data(row));
            });

            return all_results;
        } else {
            auto processor = FeatureVectorsProcessor<dist_t>(feature_vectors, dim);
            result_t all_results = getKnnResultArray(processor.getRows(), k);

            processor.process(num_threads, [&](size_t row, int thread_id, const dist_t *input_vector) {
                appr_alg->searchKnn(input_vector, k, all_results.mutable_data(row));
            });

            return all_results;
        }
    }

    inline result_t getKnnResultArray(size_t rows, size_t k) {
        return rows == 1 ? result_t(k) : result_t({rows, k});
    }

    void markDeleted(label_t& label) {
        appr_alg->markDelete(label);
    }

    void resizeIndex(size_t new_size) {
        appr_alg->resizeIndex(new_size);
    }

    size_t getMaxElements() const {
        return appr_alg->max_elements_;
    }

    size_t getCurrentCount() const {
        return appr_alg->cur_element_count_;
    }

    const std::string space_name;
    const int dim;


    bool index_inited;
    bool normalize;
    int num_threads_default;
    hnswlib::HierarchicalNSW<dist_t, label_t> *appr_alg;
    hnswlib::SpaceInterface<dist_t> *l2space;
    hnswlib::NormalizerInterface<src_t, dist_t> *normalizer;

    ~Index() {
        delete l2space;
        if (appr_alg)
            delete appr_alg;
        if (normalizer)
            delete normalizer;
    }
};

PYBIND11_MODULE(hnswlib, m) {
    PYBIND11_NUMPY_DTYPE(neighbour_t, distance, label);

    py::class_<Index>(m, "Index")
    .def(py::init<const std::string &, const int>(), py::arg("space"), py::arg("dim"))
    .def("init_index", &Index::init_new_index, py::arg("max_elements"), py::arg("M")=16,
        py::arg("ef_construction")=200, py::arg("random_seed")=100)
    .def("knn_query", &Index::knnQuery, py::arg("data"), py::arg("k")=1, py::arg("num_threads")=-1)
    .def("add_items", &Index::addItems, py::arg("data"), py::arg("ids"), py::arg("num_threads")=-1)
    .def("get_items", &Index::getData, py::arg("ids"))
    .def("get_ids_list", &Index::getIdsList)
    .def("set_ef", &Index::set_ef, py::arg("ef"))
    .def("get_ef_construction", &Index::get_ef_construction)
    .def("get_M", &Index::get_M)
    .def("set_num_threads", &Index::set_num_threads, py::arg("num_threads"))
    .def("save_index", &Index::saveIndex, py::arg("path_to_index"))
    .def("write_index", &Index::writeIndex, py::arg("write_function"))
    .def("load_index", &Index::loadIndex, py::arg("path_to_index"), py::arg("use_mmap")=false, py::arg("max_elements")=0)
    .def("mark_deleted", &Index::markDeleted, py::arg("label"))
    .def("resize_index", &Index::resizeIndex, py::arg("new_size"))
    .def("get_max_elements", &Index::getMaxElements)
    .def("get_current_count", &Index::getCurrentCount)
    .def("__repr__", [](const Index &a) {
            return py::str("HNSW-lib index of space {0} with dimension {1}").format(a.space_name, a.dim);
        }
    );
}
