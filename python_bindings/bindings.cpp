#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../hnswlib/hnswlib.h"
#include <omp.h>

namespace py = pybind11;

template<typename dist_t>
class Index {
public:
    Index(const std::string &space_name, const int dim) :
            space_name(space_name),dim(dim) {
        l2space = new hnswlib::L2Space(dim);
        appr_alg = NULL;
        ep_added = true;
        index_inited = false;
        num_threads = omp_get_max_threads();
    }
    void init_new_index(const size_t maxElements, const size_t M, const size_t efConstruction){
        cur_l=0;
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, maxElements, M, efConstruction, false);
        index_inited = true;
        ep_added = false;
    }
    void set_ef(size_t ef){
        appr_alg->ef_=ef;
    }

    void set_num_threads(int num_threads) {
        this->num_threads = num_threads;
        omp_set_num_threads(num_threads);
    }
    void saveIndex(const std::string &path_to_index){
        appr_alg->saveIndex(path_to_index);
    }
    void loadIndex(const std::string &path_to_index){
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, path_to_index);
    }
    py::object addItems(py::object input, py::object ids_ = py::none()) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        size_t rows = buffer.shape[0], features = buffer.shape[1];

        std::vector<size_t> ids;

        if (!ids_.is_none()) {
            py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
            auto ids_numpy = items.request();
            std::vector<size_t> ids1(ids_numpy.shape[0]);
            for(size_t i=0;i<ids1.size();i++){
                ids1[i]=items.data()[i];
            }
            ids.swap(ids1);
        }

        hnswlib::tableint *data_numpy;

        {

            py::gil_scoped_release l;


            if (buffer.ndim != 2) throw std::runtime_error("data must be a 2d array");

            if (features != dim)
                throw std::runtime_error("wrong dimensionality of the vectors");

            data_numpy = new hnswlib::tableint[rows];
            int start=0;
            if (!ep_added){
                size_t id = ids.size() ? ids.at(0) : (cur_l++);
                data_numpy[0] = appr_alg->addPoint((void *) items.data(0), (size_t) id);
                start=1;
                ep_added = true;
            }
            if (num_threads == 1) {
                for (size_t row = start; row < rows; row++) {
                    size_t id = ids.size() ? ids.at(row) : (cur_l++);
                    data_numpy[row] = appr_alg->addPoint((void *) items.data(row), (size_t) id);
                }
            } else {
#pragma omp parallel for
                for (size_t row = start; row < rows; row++) {
                    size_t id = ids.size() ? ids.at(row) : (cur_l++);
                    data_numpy[row] = appr_alg->addPoint((void *) items.data(row), (size_t) id);
                }
            }

        }
        py::capsule free_when_done(data_numpy, [](void *f) {
            delete[] f;
        });

        return py::array_t<hnswlib::tableint >(
                {rows}, // shape
                {sizeof(hnswlib::tableint)}, // C-style contiguous strides for double
                data_numpy, // the data pointer
                free_when_done);


    }

    py::object knnQuery_return_numpy(py::object input, size_t k = 1) {

        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        hnswlib::labeltype *data_numpy_l;
        dist_t *data_numpy_d;
        size_t rows, features;
        {
            //py::gil_scoped_release l;

            if (buffer.ndim != 2) throw std::runtime_error("data must be a 2d array");

            rows = buffer.shape[0];
            features = buffer.shape[1];


            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];

#pragma omp parallel for
            for (size_t row = 0; row < rows; row++) {
                std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void *) items.data(row), k);
                if (result.size() != k)
                    std::runtime_error(
                            "Cannot return the results in a contigious 2D array. Probably ef or M is to small");
                for (int i = 0; i < k; i++) {
                    auto &result_tuple = result.top();
                    data_numpy_d[row * k + i] = result_tuple.first;
                    data_numpy_l[row * k + i] = result_tuple.second;
                    result.pop();
                }
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
                        {rows,k}, // shape
                        {k*sizeof(hnswlib::labeltype),sizeof(hnswlib::labeltype)}, // C-style contiguous strides for double
                        data_numpy_l, // the data pointer
                        free_when_done_l),
                py::array_t<dist_t>(
                        {rows,k}, // shape
                        {k*sizeof(dist_t),sizeof(dist_t)}, // C-style contiguous strides for double
                        data_numpy_d, // the data pointer
                        free_when_done_d));

    }

    std::string space_name;
    int dim;


    bool index_inited;
    bool ep_added;
    int num_threads;
    hnswlib::labeltype cur_l;
    hnswlib::HierarchicalNSW<dist_t> *appr_alg;
    hnswlib::L2Space *l2space;
    ~Index(){
        delete l2space;
        if (appr_alg)
            delete appr_alg;
    }
};

PYBIND11_PLUGIN(hnswlib) {
        py::module m("hnswlib");

        py::class_<Index<float>>(m, "Index")
        .def(py::init<const std::string &, const int>(),py::arg("space"), py::arg("dim"))
        .def("init_index", &Index<float>::init_new_index,py::arg("max_elements"), py::arg("M")=16, py::arg("ef_construction")=200)
        .def("knn_query", &Index<float>::knnQuery_return_numpy,py::arg("data"), py::arg("k")=1)
        .def("add_items", &Index<float>::addItems,py::arg("data"), py::arg("ids") = py::none())
        .def("set_ef", &Index<float>::set_ef,py::arg("ef"))
        .def("set_num_threads", &Index<float>::set_num_threads, py::arg("num_threads"))
        .def("save_index", &Index<float>::saveIndex,py::arg("path_to_index"))
        .def("load_index", &Index<float>::loadIndex,py::arg("path_to_index"))
        .def("__repr__",
        [](const Index<float> &a) {
            return "<HNSW-lib index>";
        }
        );
        return m.ptr();
}
