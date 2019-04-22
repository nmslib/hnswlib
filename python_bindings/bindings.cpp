#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../hnswlib/hnswlib.h"
#include <thread>

namespace py = pybind11;

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib 
 */
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if ((id >= end)) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
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

template<typename dist_t, typename data_t=float>
class Index {
public:
    Index(const std::string &space_name, const int dim) :
            space_name(space_name), dim(dim) {
        normalize=false;
        if(space_name=="l2") {
            l2space = new hnswlib::L2Space(dim);
        }
        else if(space_name=="ip") {
            l2space = new hnswlib::InnerProductSpace(dim);
        }
        else if(space_name=="cosine") {
            l2space = new hnswlib::InnerProductSpace(dim);
            normalize=true;
        }
        appr_alg = NULL;
        ep_added = true;
        index_inited = false;
        num_threads_default = std::thread::hardware_concurrency();
    }

    void init_new_index(const size_t maxElements, const size_t M, const size_t efConstruction, const size_t random_seed) {
        if (appr_alg) {
            throw new std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, maxElements, M, efConstruction, random_seed);
        index_inited = true;
        ep_added = false;
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
        this->num_threads_default = num_threads;
    }

    void saveIndex(const std::string &path_to_index) {
        appr_alg->saveIndex(path_to_index);
    }

    void loadIndex(const std::string &path_to_index, size_t max_elements) {
        if (appr_alg) {
            std::cerr<<"Warning: Calling load_index for an already inited index. Old index is being deallocated.";
            delete appr_alg;
        }
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, path_to_index, false, max_elements);
		cur_l = appr_alg->cur_element_count;
    }
	void normalize_vector(float *data, float *norm_array){
		float norm=0.0f;
		for(int i=0;i<dim;i++)
			norm+=data[i]*data[i];
		norm= 1.0f / (sqrtf(norm) + 1e-30f);
		for(int i=0;i<dim;i++)
			norm_array[i]=data[i]*norm;
	}

    void addItems(py::object input, py::object ids_ = py::none(), int num_threads = -1) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        if (num_threads <= 0)
            num_threads = num_threads_default;

        size_t rows, features;

        if (buffer.ndim != 2 && buffer.ndim != 1) throw std::runtime_error("data must be a 1d/2d array");
        if (buffer.ndim == 2) {
            rows = buffer.shape[0];
            features = buffer.shape[1];
        }
        else{
            rows = 1;
            features = buffer.shape[0];
        }

        if (features != dim)
            throw std::runtime_error("wrong dimensionality of the vectors");

        // avoid using threads when the number of searches is small:

        if(rows<=num_threads*4){
            num_threads=1;
        }

        std::vector<size_t> ids;

        if (!ids_.is_none()) {
            py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
            auto ids_numpy = items.request();
            if(ids_numpy.ndim==1 && ids_numpy.shape[0]==rows) {
                std::vector<size_t> ids1(ids_numpy.shape[0]);
                for (size_t i = 0; i < ids1.size(); i++) {
                    ids1[i] = items.data()[i];
                }
                ids.swap(ids1);
            }
            else if(ids_numpy.ndim==0 && rows==1) {
                ids.push_back(*items.data());
            }
            else
                throw std::runtime_error("wrong dimensionality of the labels");
        }


        {

            int start = 0;
            if (!ep_added) {
                size_t id = ids.size() ? ids.at(0) : (cur_l);
				float *vector_data=(float *) items.data(0);
                                std::vector<float> norm_array(dim);
				if(normalize){					
					normalize_vector(vector_data, norm_array.data());					
					vector_data = norm_array.data();
					
				}
				appr_alg->addPoint((void *) vector_data, (size_t) id);
                start = 1;
                ep_added = true;
            }

            py::gil_scoped_release l;
            if(normalize==false) {
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t id = ids.size() ? ids.at(row) : (cur_l+row);
                    appr_alg->addPoint((void *) items.data(row), (size_t) id);
                });
            } else{
                std::vector<float> norm_array(num_threads * dim);
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    // normalize vector:
					size_t start_idx = threadId * dim;
                    normalize_vector((float *) items.data(row), (norm_array.data()+start_idx));

                    size_t id = ids.size() ? ids.at(row) : (cur_l+row);
                    appr_alg->addPoint((void *) (norm_array.data()+start_idx), (size_t) id);
                });
            };
            cur_l+=rows;
        }
    }

    std::vector<std::vector<data_t>> getDataReturnList(py::object ids_ = py::none()) {
        std::vector<size_t> ids;
        if (!ids_.is_none()) {
            py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
            auto ids_numpy = items.request();
            std::vector<size_t> ids1(ids_numpy.shape[0]);
            for (size_t i = 0; i < ids1.size(); i++) {
                ids1[i] = items.data()[i];
            }
            ids.swap(ids1);
        }

        std::vector<std::vector<data_t>> data;
        for (auto id : ids) {
            data.push_back(appr_alg->template getDataByLabel<data_t>(id));
        }
        return data;
    }

    std::vector<unsigned int> getIdsList() {

        std::vector<unsigned int> ids;

        for(auto kv : appr_alg->label_lookup_) {
            ids.push_back(kv.first);
        }
        return ids;
    }

    py::object knnQuery_return_numpy(py::object input, size_t k = 1, int num_threads = -1) {

        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        hnswlib::labeltype *data_numpy_l;
        dist_t *data_numpy_d;
        size_t rows, features;

        if (num_threads <= 0)
            num_threads = num_threads_default;

        {
            py::gil_scoped_release l;

            if (buffer.ndim != 2 && buffer.ndim != 1) throw std::runtime_error("data must be a 1d/2d array");
            if (buffer.ndim == 2) {
                rows = buffer.shape[0];
                features = buffer.shape[1];
            }
            else{
                rows = 1;
                features = buffer.shape[0];
            }


            // avoid using threads when the number of searches is small:

            if(rows<=num_threads*4){
                num_threads=1;
            }

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];

            if(normalize==false) {
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                                std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                                        (void *) items.data(row), k);
                                if (result.size() != k)
                                    throw std::runtime_error(
                                            "Cannot return the results in a contigious 2D array. Probably ef or M is to small");
                                for (int i = k - 1; i >= 0; i--) {
                                    auto &result_tuple = result.top();
                                    data_numpy_d[row * k + i] = result_tuple.first;
                                    data_numpy_l[row * k + i] = result_tuple.second;
                                    result.pop();
                                }
                            }
                );
            }
            else{
                std::vector<float> norm_array(num_threads*features);
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                                float *data= (float *) items.data(row);

                                size_t start_idx = threadId * dim;
								normalize_vector((float *) items.data(row), (norm_array.data()+start_idx));

                                std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                                        (void *) (norm_array.data()+start_idx), k);
                                if (result.size() != k)
                                    throw std::runtime_error(
                                            "Cannot return the results in a contigious 2D array. Probably ef or M is to small");
                                for (int i = k - 1; i >= 0; i--) {
                                    auto &result_tuple = result.top();
                                    data_numpy_d[row * k + i] = result_tuple.first;
                                    data_numpy_l[row * k + i] = result_tuple.second;
                                    result.pop();
                                }
                            }
                );
            }

        }
        py::capsule free_when_done_l(data_numpy_l, [](void *f) {
            delete[] f;
        });
        py::capsule free_when_done_d(data_numpy_d, [](void *f) {
            delete[] f;
        });


        return py::make_tuple(
                py::array_t<hnswlib::labeltype>(
                        {rows, k}, // shape
                        {k * sizeof(hnswlib::labeltype),
                         sizeof(hnswlib::labeltype)}, // C-style contiguous strides for double
                        data_numpy_l, // the data pointer
                        free_when_done_l),
                py::array_t<dist_t>(
                        {rows, k}, // shape
                        {k * sizeof(dist_t), sizeof(dist_t)}, // C-style contiguous strides for double
                        data_numpy_d, // the data pointer
                        free_when_done_d));

    }

    std::string space_name;
    int dim;


    bool index_inited;
    bool ep_added;
    bool normalize;
    int num_threads_default;
    hnswlib::labeltype cur_l;
    hnswlib::HierarchicalNSW<dist_t> *appr_alg;
    hnswlib::SpaceInterface<float> *l2space;

    ~Index() {
        delete l2space;
        if (appr_alg)
            delete appr_alg;
    }
};

PYBIND11_PLUGIN(hnswlib) {
        py::module m("hnswlib");

        py::class_<Index<float>>(m, "Index")
        .def(py::init<const std::string &, const int>(), py::arg("space"), py::arg("dim"))
        .def("init_index", &Index<float>::init_new_index, py::arg("max_elements"), py::arg("M")=16,
        py::arg("ef_construction")=200, py::arg("random_seed")=100)
        .def("knn_query", &Index<float>::knnQuery_return_numpy, py::arg("data"), py::arg("k")=1, py::arg("num_threads")=-1)
        .def("add_items", &Index<float>::addItems, py::arg("data"), py::arg("ids") = py::none(), py::arg("num_threads")=-1)
        .def("get_items", &Index<float, float>::getDataReturnList, py::arg("ids") = py::none())
        .def("get_ids_list", &Index<float>::getIdsList)
        .def("set_ef", &Index<float>::set_ef, py::arg("ef"))
        .def("get_ef_construction", &Index<float>::get_ef_construction)
        .def("get_M", &Index<float>::get_M)
        .def("set_num_threads", &Index<float>::set_num_threads, py::arg("num_threads"))
        .def("save_index", &Index<float>::saveIndex, py::arg("path_to_index"))
        .def("load_index", &Index<float>::loadIndex, py::arg("path_to_index"), py::arg("max_elements")=0)
        .def("__repr__",
        [](const Index<float> &a) {
            return "<HNSW-lib index>";
        }
        );
        return m.ptr();
}
