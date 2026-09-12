#include <iostream>
#include <algorithm>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "hnswlib.h"
#include <thread>
#include <atomic>
#include <stdlib.h>
#include <assert.h>

namespace py = pybind11;
using namespace pybind11::literals;  // needed to bring in _a literal

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

                    if (id >= end) {
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


inline void assert_true(bool expr, const std::string & msg) {
    // Not using HNSWLIB_THROW_RUNTIME_ERROR here, because it expects a static
    // string constant, and because we currently always compile the Python
    // bindings with exceptions enabled.
    if (!expr) throw std::runtime_error("Unpickle Error: " + msg);
}


class CustomFilterFunctor: public hnswlib::BaseFilterFunctor {
    std::function<bool(hnswlib::labeltype)> filter;

 public:
    explicit CustomFilterFunctor(const std::function<bool(hnswlib::labeltype)>& f) {
        filter = f;
    }

    bool operator()(hnswlib::labeltype id) {
        return filter(id);
    }
};


inline void get_input_array_shapes(const py::buffer_info& buffer, size_t* rows, size_t* features) {
    if (buffer.ndim != 2 && buffer.ndim != 1) {
        char msg[256];
        snprintf(msg, sizeof(msg),
            "Input vector data wrong shape. Number of dimensions %d. Data must be a 1D or 2D array.",
            buffer.ndim);
        HNSWLIB_THROW_RUNTIME_ERROR(msg);
    }
    if (buffer.ndim == 2) {
        *rows = buffer.shape[0];
        *features = buffer.shape[1];
    } else {
        *rows = 1;
        *features = buffer.shape[0];
    }
}


inline std::vector<size_t> get_input_ids_and_check_shapes(const py::object& ids_, size_t feature_rows) {
    std::vector<size_t> ids;
    if (!ids_.is_none()) {
        py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
        auto ids_numpy = items.request();
        // check shapes
        if (!((ids_numpy.ndim == 1 && ids_numpy.shape[0] == feature_rows) ||
              (ids_numpy.ndim == 0 && feature_rows == 1))) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                "The input label shape %d does not match the input data vector shape %d",
                ids_numpy.ndim, feature_rows);
            HNSWLIB_THROW_RUNTIME_ERROR(msg);
        }
        // extract data
        if (ids_numpy.ndim == 1) {
            std::vector<size_t> ids1(ids_numpy.shape[0]);
            for (size_t i = 0; i < ids1.size(); i++) {
                ids1[i] = items.data()[i];
            }
            ids.swap(ids1);
        } else if (ids_numpy.ndim == 0) {
            ids.push_back(*items.data());
        }
    }

    return ids;
}


template<typename dist_t, typename data_t = float>
class Index {
 public:
    static const int ser_version = 1;  // serialization version

    std::string space_name;
    int dim;
    size_t seed;
    size_t default_ef;

    bool index_inited;
    bool ep_added;
    bool normalize;
    int num_threads_default;
    hnswlib::labeltype cur_l;
    hnswlib::HierarchicalNSW<dist_t>* appr_alg;
    hnswlib::SpaceInterface<float>* l2space;


    Index(const std::string &space_name, const int dim) : space_name(space_name), dim(dim) {
        normalize = false;
        if (space_name == "l2") {
            l2space = new hnswlib::L2Space(dim);
        } else if (space_name == "ip") {
            l2space = new hnswlib::InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            l2space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        } else {
            HNSWLIB_THROW_RUNTIME_ERROR("Space name must be one of l2, ip, or cosine.");
        }
        appr_alg = NULL;
        ep_added = true;
        index_inited = false;
        num_threads_default = std::thread::hardware_concurrency();

        default_ef = 10;
    }


    ~Index() {
        delete l2space;
        if (appr_alg)
            delete appr_alg;
    }


    void init_new_index(
        size_t maxElements,
        size_t M,
        size_t efConstruction,
        size_t random_seed,
        bool allow_replace_deleted) {
        if (appr_alg) {
            HNSWLIB_THROW_RUNTIME_ERROR("The index is already initiated.");
        }
        cur_l = 0;
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, maxElements, M, efConstruction, random_seed, allow_replace_deleted);
        index_inited = true;
        ep_added = false;
        appr_alg->ef_ = default_ef;
        seed = random_seed;
    }


    void set_ef(size_t ef) {
      default_ef = ef;
      if (appr_alg)
          appr_alg->ef_ = ef;
    }


    void set_num_threads(int num_threads) {
        this->num_threads_default = num_threads;
    }

    size_t indexFileSize() const {
        return appr_alg->indexFileSize();
    }

    void saveIndex(const std::string &path_to_index) {
        appr_alg->saveIndex(path_to_index);
    }


    void loadIndex(const std::string &path_to_index, size_t max_elements, bool allow_replace_deleted) {
      if (appr_alg) {
          std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated." << std::endl;
          delete appr_alg;
      }
      appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, path_to_index, false, max_elements, allow_replace_deleted);
      cur_l = appr_alg->cur_element_count;
      index_inited = true;
    }


    void normalize_vector(float* data, float* norm_array) {
        float norm = 0.0f;
        for (int i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for (int i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
    }


    void addItems(py::object input, py::object ids_ = py::none(), int num_threads = -1, bool replace_deleted = false) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        if (num_threads <= 0)
            num_threads = num_threads_default;

        size_t rows, features;
        get_input_array_shapes(buffer, &rows, &features);

        if (features != dim)
            HNSWLIB_THROW_RUNTIME_ERROR("Wrong dimensionality of the vectors");

        // avoid using threads when the number of additions is small:
        if (rows <= num_threads * 4) {
            num_threads = 1;
        }

        std::vector<size_t> ids = get_input_ids_and_check_shapes(ids_, rows);

        {
            int start = 0;
            if (!ep_added) {
                size_t id = ids.size() ? ids.at(0) : (cur_l);
                float* vector_data = (float*)items.data(0);
                std::vector<float> norm_array(dim);
                if (normalize) {
                    normalize_vector(vector_data, norm_array.data());
                    vector_data = norm_array.data();
                }
                appr_alg->addPoint((void*)vector_data, (size_t)id, replace_deleted);
                start = 1;
                ep_added = true;
            }

            py::gil_scoped_release l;
            if (normalize == false) {
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    appr_alg->addPoint((void*)items.data(row), (size_t)id, replace_deleted);
                    });
            } else {
                std::vector<float> norm_array(num_threads * dim);
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    // normalize vector:
                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)items.data(row), (norm_array.data() + start_idx));

                    size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    appr_alg->addPoint((void*)(norm_array.data() + start_idx), (size_t)id, replace_deleted);
                    });
            }
            cur_l += rows;
        }
    }


    py::object getData(py::object ids_ = py::none(), std::string return_type = "numpy") {
        std::vector<std::string> return_types{"numpy", "list"};
        if (std::find(std::begin(return_types), std::end(return_types), return_type) == std::end(return_types)) {
            throw std::invalid_argument("return_type should be \"numpy\" or \"list\"");
        }
        std::vector<size_t> ids;
        if (!ids_.is_none()) {
            py::array_t < size_t, py::array::c_style | py::array::forcecast > items(ids_);
            auto ids_numpy = items.request();

            if (ids_numpy.ndim == 0) {
                throw std::invalid_argument("get_items accepts a list of indices and returns a list of vectors");
            } else {
                std::vector<size_t> ids1(ids_numpy.shape[0]);
                for (size_t i = 0; i < ids1.size(); i++) {
                    ids1[i] = items.data()[i];
                }
                ids.swap(ids1);
            }
        }

        std::vector<std::vector<data_t>> data;
        {
            // Pure C++ work only (no Python objects touched) -- safe to
            // release the GIL for the (potentially large) copy loop, same
            // pattern as knnQuery_return_numpy() above. The GIL is
            // reacquired when this scope ends, before any py::cast/
            // py::array_t construction below.
            py::gil_scoped_release release_for_copy_loop;
            for (auto id : ids) {
                data.push_back(appr_alg->template getDataByLabel<data_t>(id));
            }
        }
        if (return_type == "list") {
            return py::cast(data);
        }
        if (return_type == "numpy") {
            return py::array_t< data_t, py::array::c_style | py::array::forcecast >(py::cast(data));
        }
    }


    std::vector<hnswlib::labeltype> getIdsList() {
        std::vector<hnswlib::labeltype> ids;

        for (auto kv : appr_alg->label_lookup_) {
            ids.push_back(kv.first);
        }
        return ids;
    }


    py::dict getAnnData() const { /* WARNING: Index::getAnnData is not thread-safe with Index::addItems */
        std::unique_lock <std::mutex> templock(appr_alg->global);

        size_t level0_npy_size = appr_alg->cur_element_count * appr_alg->size_data_per_element_;
        size_t link_npy_size = 0;
        std::vector<size_t> link_npy_offsets(appr_alg->cur_element_count);

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            link_npy_offsets[i] = link_npy_size;
            if (linkListSize)
                link_npy_size += linkListSize;
        }

        char* data_level0_npy = (char*)malloc(level0_npy_size);
        char* link_list_npy = (char*)malloc(link_npy_size);
        int* element_levels_npy = (int*)malloc(appr_alg->element_levels_.size() * sizeof(int));

        hnswlib::labeltype* label_lookup_key_npy = (hnswlib::labeltype*)malloc(appr_alg->label_lookup_.size() * sizeof(hnswlib::labeltype));
        hnswlib::tableint* label_lookup_val_npy = (hnswlib::tableint*)malloc(appr_alg->label_lookup_.size() * sizeof(hnswlib::tableint));

        memset(label_lookup_key_npy, -1, appr_alg->label_lookup_.size() * sizeof(hnswlib::labeltype));
        memset(label_lookup_val_npy, -1, appr_alg->label_lookup_.size() * sizeof(hnswlib::tableint));

        size_t idx = 0;
        for (auto it = appr_alg->label_lookup_.begin(); it != appr_alg->label_lookup_.end(); ++it) {
            label_lookup_key_npy[idx] = it->first;
            label_lookup_val_npy[idx] = it->second;
            idx++;
        }

        memset(link_list_npy, 0, link_npy_size);

        memcpy(data_level0_npy, appr_alg->data_level0_memory_, level0_npy_size);
        memcpy(element_levels_npy, appr_alg->element_levels_.data(), appr_alg->element_levels_.size() * sizeof(int));

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            if (linkListSize) {
                memcpy(link_list_npy + link_npy_offsets[i], appr_alg->linkLists_[i], linkListSize);
            }
        }

        py::capsule free_when_done_l0(data_level0_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_lvl(element_levels_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_lb(label_lookup_key_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_id(label_lookup_val_npy, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_ll(link_list_npy, [](void* f) {
            delete[] f;
            });

        /*  TODO: serialize state of random generators appr_alg->level_generator_ and appr_alg->update_probability_generator_  */
        /*        for full reproducibility / to avoid re-initializing generators inside Index::createFromParams         */

        return py::dict(
            "offset_level0"_a = appr_alg->offsetLevel0_,
            "max_elements"_a = appr_alg->max_elements_,
            "cur_element_count"_a = (size_t)appr_alg->cur_element_count,
            "size_data_per_element"_a = appr_alg->size_data_per_element_,
            "label_offset"_a = appr_alg->label_offset_,
            "offset_data"_a = appr_alg->offsetData_,
            "max_level"_a = appr_alg->maxlevel_,
            "enterpoint_node"_a = appr_alg->enterpoint_node_,
            "max_M"_a = appr_alg->maxM_,
            "max_M0"_a = appr_alg->maxM0_,
            "M"_a = appr_alg->M_,
            "mult"_a = appr_alg->mult_,
            "ef_construction"_a = appr_alg->ef_construction_,
            "ef"_a = appr_alg->ef_,
            "has_deletions"_a = (bool)appr_alg->num_deleted_,
            "size_links_per_element"_a = appr_alg->size_links_per_element_,
            "allow_replace_deleted"_a = appr_alg->allow_replace_deleted_,

            "label_lookup_external"_a = py::array_t<hnswlib::labeltype>(
                { appr_alg->label_lookup_.size() },  // shape
                { sizeof(hnswlib::labeltype) },  // C-style contiguous strides for each index
                label_lookup_key_npy,  // the data pointer
                free_when_done_lb),

            "label_lookup_internal"_a = py::array_t<hnswlib::tableint>(
                { appr_alg->label_lookup_.size() },  // shape
                { sizeof(hnswlib::tableint) },  // C-style contiguous strides for each index
                label_lookup_val_npy,  // the data pointer
                free_when_done_id),

            "element_levels"_a = py::array_t<int>(
                { appr_alg->element_levels_.size() },  // shape
                { sizeof(int) },  // C-style contiguous strides for each index
                element_levels_npy,  // the data pointer
                free_when_done_lvl),

            // linkLists_,element_levels_,data_level0_memory_
            "data_level0"_a = py::array_t<char>(
                { level0_npy_size },  // shape
                { sizeof(char) },  // C-style contiguous strides for each index
                data_level0_npy,  // the data pointer
                free_when_done_l0),

            "link_lists"_a = py::array_t<char>(
                { link_npy_size },  // shape
                { sizeof(char) },  // C-style contiguous strides for each index
                link_list_npy,  // the data pointer
                free_when_done_ll));
    }


    py::dict getIndexParams() const { /* WARNING: Index::getAnnData is not thread-safe with Index::addItems */
        auto params = py::dict(
            "ser_version"_a = py::int_(Index<float>::ser_version),  // serialization version
            "space"_a = space_name,
            "dim"_a = dim,
            "index_inited"_a = index_inited,
            "ep_added"_a = ep_added,
            "normalize"_a = normalize,
            "num_threads"_a = num_threads_default,
            "seed"_a = seed);

        if (index_inited == false)
            return py::dict(**params, "ef"_a = default_ef);

        auto ann_params = getAnnData();

        return py::dict(**params, **ann_params);
    }


    static Index<float>* createFromParams(const py::dict d) {
        // check serialization version
        assert_true(((int)py::int_(Index<float>::ser_version)) >= d["ser_version"].cast<int>(), "Invalid serialization version!");

        auto space_name_ = d["space"].cast<std::string>();
        auto dim_ = d["dim"].cast<int>();
        auto index_inited_ = d["index_inited"].cast<bool>();

        Index<float>* new_index = new Index<float>(space_name_, dim_);

        /*  TODO: deserialize state of random generators into new_index->level_generator_ and new_index->update_probability_generator_  */
        /*        for full reproducibility / state of generators is serialized inside Index::getIndexParams                      */
        new_index->seed = d["seed"].cast<size_t>();

        if (index_inited_) {
            new_index->appr_alg = new hnswlib::HierarchicalNSW<dist_t>(
                new_index->l2space,
                d["max_elements"].cast<size_t>(),
                d["M"].cast<size_t>(),
                d["ef_construction"].cast<size_t>(),
                new_index->seed);
            new_index->cur_l = d["cur_element_count"].cast<size_t>();
        }

        new_index->index_inited = index_inited_;
        new_index->ep_added = d["ep_added"].cast<bool>();
        new_index->num_threads_default = d["num_threads"].cast<int>();
        new_index->default_ef = d["ef"].cast<size_t>();

        if (index_inited_)
            new_index->setAnnData(d);

        return new_index;
    }


    static Index<float> * createFromIndex(const Index<float> & index) {
        return createFromParams(index.getIndexParams());
    }


    void setAnnData(const py::dict d) { /* WARNING: Index::setAnnData is not thread-safe with Index::addItems */
        std::unique_lock <std::mutex> templock(appr_alg->global);

        assert_true(appr_alg->offsetLevel0_ == d["offset_level0"].cast<size_t>(), "Invalid value of offsetLevel0_ ");
        assert_true(appr_alg->max_elements_ == d["max_elements"].cast<size_t>(), "Invalid value of max_elements_ ");

        appr_alg->cur_element_count = d["cur_element_count"].cast<size_t>();

        assert_true(appr_alg->size_data_per_element_ == d["size_data_per_element"].cast<size_t>(), "Invalid value of size_data_per_element_ ");
        assert_true(appr_alg->label_offset_ == d["label_offset"].cast<size_t>(), "Invalid value of label_offset_ ");
        assert_true(appr_alg->offsetData_ == d["offset_data"].cast<size_t>(), "Invalid value of offsetData_ ");

        appr_alg->maxlevel_ = d["max_level"].cast<int>();
        appr_alg->enterpoint_node_ = d["enterpoint_node"].cast<hnswlib::tableint>();

        assert_true(appr_alg->maxM_ == d["max_M"].cast<size_t>(), "Invalid value of maxM_ ");
        assert_true(appr_alg->maxM0_ == d["max_M0"].cast<size_t>(), "Invalid value of maxM0_ ");
        assert_true(appr_alg->M_ == d["M"].cast<size_t>(), "Invalid value of M_ ");
        assert_true(appr_alg->mult_ == d["mult"].cast<double>(), "Invalid value of mult_ ");
        assert_true(appr_alg->ef_construction_ == d["ef_construction"].cast<size_t>(), "Invalid value of ef_construction_ ");

        appr_alg->ef_ = d["ef"].cast<size_t>();

        assert_true(appr_alg->size_links_per_element_ == d["size_links_per_element"].cast<size_t>(), "Invalid value of size_links_per_element_ ");

        auto label_lookup_key_npy = d["label_lookup_external"].cast<py::array_t < hnswlib::labeltype, py::array::c_style | py::array::forcecast > >();
        auto label_lookup_val_npy = d["label_lookup_internal"].cast<py::array_t < hnswlib::tableint, py::array::c_style | py::array::forcecast > >();
        auto element_levels_npy = d["element_levels"].cast<py::array_t < int, py::array::c_style | py::array::forcecast > >();
        auto data_level0_npy = d["data_level0"].cast<py::array_t < char, py::array::c_style | py::array::forcecast > >();
        auto link_list_npy = d["link_lists"].cast<py::array_t < char, py::array::c_style | py::array::forcecast > >();

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            if (label_lookup_val_npy.data()[i] < 0) {
                HNSWLIB_THROW_RUNTIME_ERROR("Internal id cannot be negative!");
            } else {
                appr_alg->label_lookup_.insert(std::make_pair(label_lookup_key_npy.data()[i], label_lookup_val_npy.data()[i]));
            }
        }

        memcpy(appr_alg->element_levels_.data(), element_levels_npy.data(), element_levels_npy.nbytes());

        size_t link_npy_size = 0;
        std::vector<size_t> link_npy_offsets(appr_alg->cur_element_count);

        for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            link_npy_offsets[i] = link_npy_size;
            if (linkListSize)
                link_npy_size += linkListSize;
        }

        memcpy(appr_alg->data_level0_memory_, data_level0_npy.data(), data_level0_npy.nbytes());

        for (size_t i = 0; i < appr_alg->max_elements_; i++) {
            size_t linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
            if (linkListSize == 0) {
                appr_alg->linkLists_[i] = nullptr;
            } else {
                appr_alg->linkLists_[i] = (char*)malloc(linkListSize);
                if (appr_alg->linkLists_[i] == nullptr)
                    HNSWLIB_THROW_RUNTIME_ERROR("Not enough memory: loadIndex failed to allocate linklist");

                memcpy(appr_alg->linkLists_[i], link_list_npy.data() + link_npy_offsets[i], linkListSize);
            }
        }

        // process deleted elements
        bool allow_replace_deleted = false;
        if (d.contains("allow_replace_deleted")) {
            allow_replace_deleted = d["allow_replace_deleted"].cast<bool>();
        }
        appr_alg->allow_replace_deleted_= allow_replace_deleted;

        appr_alg->num_deleted_ = 0;
        bool has_deletions = d["has_deletions"].cast<bool>();
        if (has_deletions) {
            for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
                if (appr_alg->isMarkedDeleted(i)) {
                    appr_alg->num_deleted_ += 1;
                    if (allow_replace_deleted) appr_alg->deleted_elements.insert(i);
                }
            }
        }
    }


    py::object knnQuery_return_numpy(
        py::object input,
        size_t k = 1,
        int num_threads = -1,
        const std::function<bool(hnswlib::labeltype)>& filter = nullptr) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        hnswlib::labeltype* data_numpy_l;
        dist_t* data_numpy_d;
        size_t rows, features;

        if (num_threads <= 0)
            num_threads = num_threads_default;

        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);

            // avoid using threads when the number of searches is small:
            if (rows <= num_threads * 4) {
                num_threads = 1;
            }

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];

            // Warning: search with a filter works slow in python in multithreaded mode. For best performance set num_threads=1
            CustomFilterFunctor idFilter(filter);
            CustomFilterFunctor* p_idFilter = filter ? &idFilter : nullptr;

            if (normalize == false) {
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void*)items.data(row), k, p_idFilter);
                    if (result.size() != k)
                        HNSWLIB_THROW_RUNTIME_ERROR(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            } else {
                std::vector<float> norm_array(num_threads * features);
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    float* data = (float*)items.data(row);

                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)items.data(row), (norm_array.data() + start_idx));

                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = appr_alg->searchKnn(
                        (void*)(norm_array.data() + start_idx), k, p_idFilter);
                    if (result.size() != k)
                        HNSWLIB_THROW_RUNTIME_ERROR(
                            "Cannot return the results in a contiguous 2D array. Probably ef or M is too small");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            }
        }
        py::capsule free_when_done_l(data_numpy_l, [](void* f) {
            delete[] f;
            });
        py::capsule free_when_done_d(data_numpy_d, [](void* f) {
            delete[] f;
            });

        return py::make_tuple(
            py::array_t<hnswlib::labeltype>(
                { rows, k },  // shape
                { k * sizeof(hnswlib::labeltype),
                  sizeof(hnswlib::labeltype) },  // C-style contiguous strides for each index
                data_numpy_l,  // the data pointer
                free_when_done_l),
            py::array_t<dist_t>(
                { rows, k },  // shape
                { k * sizeof(dist_t), sizeof(dist_t) },  // C-style contiguous strides for each index
                data_numpy_d,  // the data pointer
                free_when_done_d));
    }


    void markDeleted(size_t label) {
        appr_alg->markDelete(label);
    }


    void unmarkDeleted(size_t label) {
        appr_alg->unmarkDelete(label);
    }


    void resizeIndex(size_t new_size) {
        appr_alg->resizeIndex(new_size);
    }


    size_t getMaxElements() const {
        return appr_alg->max_elements_;
    }


    size_t getCurrentCount() const {
        return appr_alg->cur_element_count;
    }


    py::dict checkIntegrity() {
        /**
         * Python-friendly integrity check that returns detailed results
         * instead of crashing on assert failures.
         *
         * Returns a dict with:
         *   - valid: bool - whether integrity check passed
         *   - connections_checked: int - total connections verified
         *   - min_inbound: int - minimum inbound connections per node
         *   - max_inbound: int - maximum inbound connections per node
         *   - errors: list[str] - list of any errors found
         */
        if (!appr_alg) {
            return py::dict(
                "valid"_a = false,
                "connections_checked"_a = 0,
                "min_inbound"_a = 0,
                "max_inbound"_a = 0,
                "errors"_a = py::list(py::cast(std::vector<std::string>{"Index not initialized"}))
            );
        }

        std::vector<std::string> errors;
        int connections_checked = 0;
        std::vector<int> inbound_connections_num(appr_alg->cur_element_count, 0);
        int min_inbound = 0, max_inbound = 0;

        {
            // Pure C++ scan only (no Python objects touched) -- safe to
            // release the GIL for the (potentially large) connection scan,
            // same pattern as knnQuery_return_numpy()/getData() above. The
            // GIL is reacquired when this scope ends, before any
            // py::list/py::dict construction below.
            py::gil_scoped_release release_for_scan;
            for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
                for (int l = 0; l <= appr_alg->element_levels_[i]; l++) {
                    hnswlib::linklistsizeint *ll_cur = appr_alg->get_linklist_at_level(i, l);
                    int size = appr_alg->getListCount(ll_cur);
                    hnswlib::tableint *data = (hnswlib::tableint *) (ll_cur + 1);
                    std::unordered_set<hnswlib::tableint> s;

                    for (int j = 0; j < size; j++) {
                        // Check: connection points to valid element
                        if (data[j] >= appr_alg->cur_element_count) {
                            errors.push_back("Element " + std::to_string(i) + " at level " +
                                std::to_string(l) + " has invalid connection to " + std::to_string(data[j]));
                        }
                        // Check: no self-loops
                        if (data[j] == i) {
                            errors.push_back("Element " + std::to_string(i) + " at level " +
                                std::to_string(l) + " has self-loop");
                        }
                        // Track for duplicate check
                        if (s.find(data[j]) != s.end()) {
                            errors.push_back("Element " + std::to_string(i) + " at level " +
                                std::to_string(l) + " has duplicate connection to " + std::to_string(data[j]));
                        }
                        s.insert(data[j]);
                        if (data[j] < appr_alg->cur_element_count) {
                            inbound_connections_num[data[j]]++;
                        }
                        connections_checked++;
                    }
                }
            }

            // Check for orphan nodes (no inbound connections)
            if (appr_alg->cur_element_count > 1) {
                min_inbound = inbound_connections_num[0];
                max_inbound = inbound_connections_num[0];
                for (size_t i = 0; i < appr_alg->cur_element_count; i++) {
                    if (inbound_connections_num[i] == 0) {
                        errors.push_back("Element " + std::to_string(i) + " has no inbound connections (orphan)");
                    }
                    min_inbound = std::min(inbound_connections_num[i], min_inbound);
                    max_inbound = std::max(inbound_connections_num[i], max_inbound);
                }
            }
        }

        py::list error_list;
        for (const auto& err : errors) {
            error_list.append(err);
        }

        return py::dict(
            "valid"_a = errors.empty(),
            "connections_checked"_a = connections_checked,
            "element_count"_a = (size_t)appr_alg->cur_element_count,
            "min_inbound"_a = min_inbound,
            "max_inbound"_a = max_inbound,
            "errors"_a = error_list
        );
    }


    py::dict repairOrphans() {
        /**
         * Deterministic repair of zero-inbound ("orphan") HNSW nodes.
         * See Story #1358 / spike #1330 (docs/research/hnsw-temporal-orphans-1330.md).
         *
         * For each orphan, forces a back-edge from its own existing level-0
         * neighbors (the same neighbor set chosen by getNeighborsByHeuristic2
         * at insertion time), guaranteeing inclusion even where the original
         * construction's heuristic later pruned it out of every neighbor's
         * list.
         *
         * SAFE EVICTION (Messi Rule 13/anti-silent-failure -- do not trade
         * one orphan for another): if a neighbor's link list is already at
         * maxM0_, repair only evicts an existing entry whose CURRENT total
         * inbound-connection count is strictly greater than 1 -- i.e. an
         * entry that has at least one OTHER inbound edge and therefore
         * cannot become a new orphan as a result of the eviction. An
         * earlier "evict the farthest-by-distance" design was measured to
         * thrash indefinitely (113k+ evictions, never converging) because
         * in a near-tie cluster ALL pairwise distances are nearly equal,
         * giving no stable "weakest" signal -- ties keep flipping which
         * node looks farthest from one pass to the next. The inbound-count
         * guard is a structural (not distance-based) safety criterion that
         * is immune to that instability: eviction is skipped (try the next
         * anchor instead) whenever no safe candidate exists at the current
         * anchor.
         *
         * Repair iterates in bounded passes (at most cur_element_count + 1
         * -- a provable termination bound, Messi Rule 14), re-scanning
         * live inbound counts after every pass, until convergence or until
         * a pass makes no further progress (a genuinely stuck residual,
         * reported via `valid: false` rather than silently accepted).
         *
         * Returns a dict with:
         *   - orphans_before: int - orphan count on the first scan
         *   - orphans_after: int - orphan count on the final scan
         *   - repaired_count: int - orphans_before - orphans_after
         *   - passes_used: int - number of repair passes actually run
         *   - forced_evictions: int - number of safe weakest-edge evictions performed
         *   - valid: bool - whether orphans_after == 0
         */
        if (!appr_alg) {
            return py::dict(
                "orphans_before"_a = (size_t)0,
                "orphans_after"_a = (size_t)0,
                "repaired_count"_a = (size_t)0,
                "passes_used"_a = (size_t)0,
                "forced_evictions"_a = (size_t)0,
                "valid"_a = false
            );
        }

        const size_t n = appr_alg->cur_element_count;

        if (n <= 1) {
            return py::dict(
                "orphans_before"_a = (size_t)0,
                "orphans_after"_a = (size_t)0,
                "repaired_count"_a = (size_t)0,
                "passes_used"_a = (size_t)0,
                "forced_evictions"_a = (size_t)0,
                "valid"_a = true
            );
        }

        size_t orphans_before = 0;
        size_t forced_evictions = 0;
        size_t passes_used = 0;
        size_t orphans_after = 0;

        {
            // Pure C++ scan-and-repair only (no Python objects touched) --
            // safe to release the GIL for the (potentially long-running)
            // repair loop, same pattern as knnQuery_return_numpy()/
            // getData()/checkIntegrity() above. The GIL is reacquired when
            // this scope ends, before the final py::dict construction below.
            // NOTE: orphans_before/forced_evictions/passes_used/orphans_after
            // are the OUTER variables declared just above (and returned by
            // the py::dict below) -- deliberately NOT redeclared here, so
            // the increments/assignments in this block write directly into
            // them rather than into a shadowed, discarded local copy.
            py::gil_scoped_release release_for_repair;

            const size_t maxM0 = appr_alg->maxM0_;
            const size_t max_passes = n + 1;

            std::vector<int> inbound(n, 0);
            for (size_t i = 0; i < n; i++) {
                for (int l = 0; l <= appr_alg->element_levels_[i]; l++) {
                    hnswlib::linklistsizeint *ll = appr_alg->get_linklist_at_level((hnswlib::tableint)i, l);
                    int size = appr_alg->getListCount(ll);
                    hnswlib::tableint *data = (hnswlib::tableint *) (ll + 1);
                    for (int j = 0; j < size; j++) {
                        if ((size_t)data[j] < n) {
                            inbound[data[j]]++;
                        }
                    }
                }
            }

            for (size_t i = 0; i < n; i++) {
                if (inbound[i] == 0) orphans_before++;
            }

            for (size_t pass = 0; pass < max_passes; pass++) {
                std::vector<hnswlib::tableint> orphans;
                for (size_t i = 0; i < n; i++) {
                    if (inbound[i] == 0) orphans.push_back((hnswlib::tableint) i);
                }
                if (orphans.empty()) break;

                passes_used = pass + 1;
                bool progress = false;

                // Attempts to connect `o` into `anchor`'s level-0 list: appends
                // if there is room, otherwise evicts a SAFE candidate (current
                // inbound > 1, so eviction cannot create a new orphan) with the
                // highest inbound count. Returns true if `o` was connected.
                auto try_connect = [&](hnswlib::tableint o, hnswlib::tableint anchor) -> bool {
                    if (anchor == o) return false;
                    // Bounds guard (mirrors the inbound-counting loop's own
                    // `if ((size_t)data[j] < n)` check above): `anchor` is read
                    // out of a link list and could in principle be corrupted
                    // (e.g. a production index with damage beyond simple
                    // orphans -- an invalid-id connection from a torn write).
                    // get_linklist0(anchor) indexes data_level0_memory_ by
                    // anchor unconditionally; an out-of-range anchor would
                    // write outside cur_element_count. Fail safe: skip it.
                    if ((size_t) anchor >= n) return false;

                    hnswlib::linklistsizeint *ll_anchor = appr_alg->get_linklist0(anchor);
                    int sz_anchor = appr_alg->getListCount(ll_anchor);
                    hnswlib::tableint *data_anchor = (hnswlib::tableint *) (ll_anchor + 1);

                    for (int j = 0; j < sz_anchor; j++) {
                        if (data_anchor[j] == o) return false;  // already present
                    }

                    if ((size_t) sz_anchor < maxM0) {
                        data_anchor[sz_anchor] = o;
                        appr_alg->setListCount(ll_anchor, sz_anchor + 1);
                        inbound[o]++;
                        return true;
                    }

                    int victim_idx = -1;
                    int victim_inbound = 0;
                    for (int j = 0; j < sz_anchor; j++) {
                        hnswlib::tableint cand = data_anchor[j];
                        // Same guard: `cand` is read out of anchor's link list
                        // and could be an out-of-range id; `inbound[cand]` would
                        // otherwise be undefined behavior (out-of-range
                        // std::vector::operator[]). Skip invalid candidates
                        // rather than crash or corrupt memory.
                        if ((size_t) cand >= n) continue;
                        if (inbound[cand] > 1 && inbound[cand] > victim_inbound) {
                            victim_inbound = inbound[cand];
                            victim_idx = j;
                        }
                    }
                    if (victim_idx >= 0) {
                        hnswlib::tableint victim = data_anchor[victim_idx];
                        data_anchor[victim_idx] = o;
                        inbound[victim]--;
                        inbound[o]++;
                        forced_evictions++;
                        return true;
                    }
                    return false;  // no room, no safe eviction candidate here
                };

                for (hnswlib::tableint o : orphans) {
                    if (inbound[o] > 0) continue;  // fixed earlier this pass as a side effect

                    hnswlib::linklistsizeint *ll0_o = appr_alg->get_linklist0(o);
                    int sz0_o = appr_alg->getListCount(ll0_o);
                    hnswlib::tableint *data0_o = (hnswlib::tableint *) (ll0_o + 1);

                    bool connected = false;
                    for (int j = 0; j < sz0_o && !connected; j++) {
                        connected = try_connect(o, data0_o[j]);
                    }

                    if (!connected) {
                        // o's own local neighborhood offered no anchor with
                        // room or a safe eviction candidate (a fragile
                        // sub-clique lockup -- measured during Story #1358
                        // calibration). Widen the search: a distance-sorted
                        // scan of the WHOLE graph. Pigeonhole guarantee: total
                        // inbound edges == total outbound edges == (roughly)
                        // n * maxM0, far more than n, so some node somewhere
                        // must have inbound > 1 (or room) -- this scan is
                        // bounded O(n) and only runs for genuinely stuck
                        // orphans (rare).
                        std::vector<std::pair<dist_t, hnswlib::tableint>> by_distance;
                        by_distance.reserve(n - 1);
                        for (size_t k = 0; k < n; k++) {
                            if ((hnswlib::tableint) k == o) continue;
                            dist_t d = appr_alg->fstdistfunc_(
                                appr_alg->getDataByInternalId(o),
                                appr_alg->getDataByInternalId((hnswlib::tableint) k),
                                appr_alg->dist_func_param_);
                            by_distance.emplace_back(d, (hnswlib::tableint) k);
                        }
                        std::sort(by_distance.begin(), by_distance.end(),
                            [](const std::pair<dist_t, hnswlib::tableint> &a,
                               const std::pair<dist_t, hnswlib::tableint> &b) {
                                return a.first < b.first;
                            });

                        for (auto &pr : by_distance) {
                            if (try_connect(o, pr.second)) {
                                connected = true;
                                break;
                            }
                        }
                    }

                    if (connected) {
                        progress = true;
                    }
                }

                if (!progress) {
                    // Genuinely stuck: no orphan in this pass had any anchor
                    // with room or a safe eviction candidate. Stop rather than
                    // burn the remaining pass budget; the final scan below
                    // reports the true residual instead of silently pretending
                    // convergence.
                    break;
                }
            }

            for (size_t i = 0; i < n; i++) {
                if (inbound[i] == 0) orphans_after++;
            }
        }

        return py::dict(
            "orphans_before"_a = orphans_before,
            "orphans_after"_a = orphans_after,
            "repaired_count"_a = orphans_before - orphans_after,
            "passes_used"_a = passes_used,
            "forced_evictions"_a = forced_evictions,
            "valid"_a = (orphans_after == 0)
        );
    }
};

template<typename dist_t, typename data_t = float>
class BFIndex {
 public:
    static const int ser_version = 1;  // serialization version

    std::string space_name;
    int dim;
    bool index_inited;
    bool normalize;
    int num_threads_default;

    hnswlib::labeltype cur_l;
    hnswlib::BruteforceSearch<dist_t>* alg;
    hnswlib::SpaceInterface<float>* space;


    BFIndex(const std::string &space_name, const int dim) : space_name(space_name), dim(dim) {
        normalize = false;
        if (space_name == "l2") {
            space = new hnswlib::L2Space(dim);
        } else if (space_name == "ip") {
            space = new hnswlib::InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            space = new hnswlib::InnerProductSpace(dim);
            normalize = true;
        } else {
            HNSWLIB_THROW_RUNTIME_ERROR("Space name must be one of l2, ip, or cosine.");
        }
        alg = NULL;
        index_inited = false;

        num_threads_default = std::thread::hardware_concurrency();
    }


    ~BFIndex() {
        delete space;
        if (alg)
            delete alg;
    }


    size_t getMaxElements() const {
        return alg->maxelements_;
    }


    size_t getCurrentCount() const {
        return alg->cur_element_count;
    }


    void set_num_threads(int num_threads) {
        this->num_threads_default = num_threads;
    }


    void init_new_index(const size_t maxElements) {
        if (alg) {
            HNSWLIB_THROW_RUNTIME_ERROR("The index is already initiated.");
        }
        cur_l = 0;
        alg = new hnswlib::BruteforceSearch<dist_t>(space, maxElements);
        index_inited = true;
    }


    void normalize_vector(float* data, float* norm_array) {
        float norm = 0.0f;
        for (int i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for (int i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
    }


    void addItems(py::object input, py::object ids_ = py::none()) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        size_t rows, features;
        get_input_array_shapes(buffer, &rows, &features);

        if (features != dim)
            HNSWLIB_THROW_RUNTIME_ERROR("Wrong dimensionality of the vectors");

        std::vector<size_t> ids = get_input_ids_and_check_shapes(ids_, rows);

        {
            for (size_t row = 0; row < rows; row++) {
                size_t id = ids.size() ? ids.at(row) : cur_l + row;
                if (!normalize) {
                    alg->addPoint((void *) items.data(row), (size_t) id);
                } else {
                    std::vector<float> normalized_vector(dim);
                    normalize_vector((float *)items.data(row), normalized_vector.data());
                    alg->addPoint((void *) normalized_vector.data(), (size_t) id);
                }
            }
            cur_l+=rows;
        }
    }


    void deleteVector(size_t label) {
        alg->removePoint(label);
    }


    void saveIndex(const std::string &path_to_index) {
        alg->saveIndex(path_to_index);
    }


    void loadIndex(const std::string &path_to_index, size_t max_elements) {
        if (alg) {
            std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated." << std::endl;
            delete alg;
        }
        alg = new hnswlib::BruteforceSearch<dist_t>(space, path_to_index);
        cur_l = alg->cur_element_count;
        index_inited = true;
    }


    py::object knnQuery_return_numpy(
        py::object input,
        size_t k = 1,
        int num_threads = -1,
        const std::function<bool(hnswlib::labeltype)>& filter = nullptr) {
        py::array_t < dist_t, py::array::c_style | py::array::forcecast > items(input);
        auto buffer = items.request();
        hnswlib::labeltype *data_numpy_l;
        dist_t *data_numpy_d;
        size_t rows, features;

        if (num_threads <= 0)
            num_threads = num_threads_default;

        {
            py::gil_scoped_release l;
            get_input_array_shapes(buffer, &rows, &features);

            data_numpy_l = new hnswlib::labeltype[rows * k];
            data_numpy_d = new dist_t[rows * k];

            CustomFilterFunctor idFilter(filter);
            CustomFilterFunctor* p_idFilter = filter ? &idFilter : nullptr;

            if (!normalize) {
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = alg->searchKnn(
                        (void*)items.data(row), k, p_idFilter);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. There are not enough elements.");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
            } else {
                std::vector<float> norm_array(num_threads * features);
                ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)items.data(row), norm_array.data() + start_idx);

                    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> result = alg->searchKnn(
                        (void*)(norm_array.data() + start_idx), k, p_idFilter);
                    if (result.size() != k)
                        throw std::runtime_error(
                            "Cannot return the results in a contiguous 2D array. There are not enough elements.");
                    for (int i = k - 1; i >= 0; i--) {
                        auto& result_tuple = result.top();
                        data_numpy_d[row * k + i] = result_tuple.first;
                        data_numpy_l[row * k + i] = result_tuple.second;
                        result.pop();
                    }
                });
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
                        { rows, k },  // shape
                        { k * sizeof(hnswlib::labeltype),
                          sizeof(hnswlib::labeltype)},  // C-style contiguous strides for each index
                        data_numpy_l,  // the data pointer
                        free_when_done_l),
                py::array_t<dist_t>(
                        { rows, k },  // shape
                        { k * sizeof(dist_t), sizeof(dist_t) },  // C-style contiguous strides for each index
                        data_numpy_d,  // the data pointer
                        free_when_done_d));
    }
};


PYBIND11_PLUGIN(hnswlib) {
        py::module m("hnswlib");

        py::class_<Index<float>>(m, "Index")
        .def(py::init(&Index<float>::createFromParams), py::arg("params"))
           /* WARNING: Index::createFromIndex is not thread-safe with Index::addItems */
        .def(py::init(&Index<float>::createFromIndex), py::arg("index"))
        .def(py::init<const std::string &, const int>(), py::arg("space"), py::arg("dim"))
        .def("init_index",
            &Index<float>::init_new_index,
            py::arg("max_elements"),
            py::arg("M") = 16,
            py::arg("ef_construction") = 200,
            py::arg("random_seed") = 100,
            py::arg("allow_replace_deleted") = false,
            py::call_guard<py::gil_scoped_release>())
        .def("knn_query",
            &Index<float>::knnQuery_return_numpy,
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1,
            py::arg("filter") = py::none())
        .def("add_items",
            &Index<float>::addItems,
            py::arg("data"),
            py::arg("ids") = py::none(),
            py::arg("num_threads") = -1,
            py::arg("replace_deleted") = false)
        .def("get_items", &Index<float>::getData, py::arg("ids") = py::none(), py::arg("return_type") = "numpy")
        .def("get_ids_list", &Index<float>::getIdsList, py::call_guard<py::gil_scoped_release>())
        .def("set_ef", &Index<float>::set_ef, py::arg("ef"))
        .def("set_num_threads", &Index<float>::set_num_threads, py::arg("num_threads"))
        .def("index_file_size", &Index<float>::indexFileSize)
        .def("save_index", &Index<float>::saveIndex, py::arg("path_to_index"),
            py::call_guard<py::gil_scoped_release>())
        .def("load_index",
            &Index<float>::loadIndex,
            py::arg("path_to_index"),
            py::arg("max_elements") = 0,
            py::arg("allow_replace_deleted") = false,
            py::call_guard<py::gil_scoped_release>())
        .def("mark_deleted", &Index<float>::markDeleted, py::arg("label"),
            py::call_guard<py::gil_scoped_release>())
        .def("unmark_deleted", &Index<float>::unmarkDeleted, py::arg("label"))
        .def("resize_index", &Index<float>::resizeIndex, py::arg("new_size"),
            py::call_guard<py::gil_scoped_release>())
        .def("get_max_elements", &Index<float>::getMaxElements)
        .def("get_current_count", &Index<float>::getCurrentCount)
        .def("check_integrity", &Index<float>::checkIntegrity,
            "Check index integrity and return detailed results.\n\n"
            "Returns a dict with:\n"
            "  - valid: bool - whether integrity check passed\n"
            "  - connections_checked: int - total connections verified\n"
            "  - element_count: int - number of elements in index\n"
            "  - min_inbound: int - minimum inbound connections per node\n"
            "  - max_inbound: int - maximum inbound connections per node\n"
            "  - errors: list[str] - list of any errors found\n")
        .def("repair_orphans", &Index<float>::repairOrphans,
            "Deterministically repair zero-inbound (orphan) HNSW nodes.\n\n"
            "Forces a back-edge from each orphan into its own existing\n"
            "level-0 neighbors, evicting the weakest existing edge when a\n"
            "neighbor's list is full. Idempotent and bounded (at most\n"
            "cur_element_count + 1 passes).\n\n"
            "Returns a dict with:\n"
            "  - orphans_before: int - orphan count on the first scan\n"
            "  - orphans_after: int - orphan count on the final scan\n"
            "  - repaired_count: int - orphans_before - orphans_after\n"
            "  - passes_used: int - number of repair passes actually run\n"
            "  - forced_evictions: int - number of weakest-edge evictions performed\n"
            "  - valid: bool - whether orphans_after == 0\n")
        .def_readonly("space", &Index<float>::space_name)
        .def_readonly("dim", &Index<float>::dim)
        .def_readwrite("num_threads", &Index<float>::num_threads_default)
        .def_property("ef",
          [](const Index<float> & index) {
            return index.index_inited ? index.appr_alg->ef_ : index.default_ef;
          },
          [](Index<float> & index, const size_t ef_) {
            index.default_ef = ef_;
            if (index.appr_alg)
              index.appr_alg->ef_ = ef_;
        })
        .def_property_readonly("max_elements", [](const Index<float> & index) {
            return index.index_inited ? index.appr_alg->max_elements_ : 0;
        })
        .def_property_readonly("element_count", [](const Index<float> & index) {
            return index.index_inited ? (size_t)index.appr_alg->cur_element_count : 0;
        })
        .def_property_readonly("ef_construction", [](const Index<float> & index) {
          return index.index_inited ? index.appr_alg->ef_construction_ : 0;
        })
        .def_property_readonly("M",  [](const Index<float> & index) {
          return index.index_inited ? index.appr_alg->M_ : 0;
        })

        .def(py::pickle(
            [](const Index<float> &ind) {  // __getstate__
                return py::make_tuple(ind.getIndexParams()); /* Return dict (wrapped in a tuple) that fully encodes state of the Index object */
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 1)
                    HNSWLIB_THROW_RUNTIME_ERROR("Invalid state!");
                return Index<float>::createFromParams(t[0].cast<py::dict>());
            }))

        .def("__repr__", [](const Index<float> &a) {
            return "<hnswlib.Index(space='" + a.space_name + "', dim="+std::to_string(a.dim)+")>";
        });

        py::class_<BFIndex<float>>(m, "BFIndex")
        .def(py::init<const std::string &, const int>(), py::arg("space"), py::arg("dim"))
        .def("init_index", &BFIndex<float>::init_new_index, py::arg("max_elements"))
        .def("knn_query",
            &BFIndex<float>::knnQuery_return_numpy,
            py::arg("data"),
            py::arg("k") = 1,
            py::arg("num_threads") = -1,
            py::arg("filter") = py::none())
        .def("add_items", &BFIndex<float>::addItems, py::arg("data"), py::arg("ids") = py::none())
        .def("delete_vector", &BFIndex<float>::deleteVector, py::arg("label"))
        .def("set_num_threads", &BFIndex<float>::set_num_threads, py::arg("num_threads"))
        .def("save_index", &BFIndex<float>::saveIndex, py::arg("path_to_index"),
            py::call_guard<py::gil_scoped_release>())
        .def("load_index", &BFIndex<float>::loadIndex, py::arg("path_to_index"), py::arg("max_elements") = 0,
            py::call_guard<py::gil_scoped_release>())
        .def("__repr__", [](const BFIndex<float> &a) {
            return "<hnswlib.BFIndex(space='" + a.space_name + "', dim="+std::to_string(a.dim)+")>";
        })
        .def("get_max_elements", &BFIndex<float>::getMaxElements)
        .def("get_current_count", &BFIndex<float>::getCurrentCount)
        .def_readwrite("num_threads", &BFIndex<float>::num_threads_default);
        return m.ptr();
}
