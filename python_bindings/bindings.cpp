#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "hnswlib/hnswlib.h"
#include <thread>
#include <atomic>
#include <stdlib.h>
#include <assert.h>

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

//
// std::priority_queue<std::pair<dist_t, labeltype >>
// searchKnn(const void *query_data, size_t k) const {
//     std::priority_queue<std::pair<dist_t, labeltype >> result;
//     if (cur_element_count == 0) return result;
//
//     tableint currObj = enterpoint_node_;
//     dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
//
//     for (int level = maxlevel_; level > 0; level--) {
//         bool changed = true;
//         while (changed) {
//             changed = false;
//             unsigned int *data;
//
//             data = (unsigned int *) get_linklist(currObj, level);
//             int size = getListCount(data);
//             metric_hops++;
//             metric_distance_computations+=size;
//
//             tableint *datal = (tableint *) (data + 1);
//             for (int i = 0; i < size; i++) {
//                 tableint cand = datal[i];
//                 if (cand < 0 || cand > max_elements_)
//                     throw std::runtime_error("cand error");
//                 dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
//
//                 if (d < curdist) {
//                     curdist = d;
//                     currObj = cand;
//                     changed = true;
//                 }
//             }
//         }
//     }
//


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

    default_ef=10;

  }
  std::string space_name;
  int dim;
  size_t seed;
  size_t default_ef;

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

    void init_new_index(const size_t maxElements, const size_t M, const size_t efConstruction, const size_t random_seed) {
        if (appr_alg) {
            throw new std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        appr_alg = new hnswlib::HierarchicalNSW<dist_t>(l2space, maxElements, M, efConstruction, random_seed);
        index_inited = true;
        ep_added = false;
        appr_alg->ef_ = default_ef;
        seed=random_seed;
    }


    void set_ef(size_t ef) {
      default_ef=ef;
      if (appr_alg)
        appr_alg->ef_ = ef;
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

    std::vector<hnswlib::labeltype> getIdsList() {

        std::vector<hnswlib::labeltype> ids;

        for(auto kv : appr_alg->label_lookup_) {
            ids.push_back(kv.first);
        }
        return ids;
    }

    inline void assert_true(bool expr, const std::string & msg) {
      if (expr == false)
        throw std::runtime_error("assert failed: "+msg);
      return;
    }


    py::tuple getAnnData() const {

      unsigned int level0_npy_size = appr_alg->cur_element_count * appr_alg->size_data_per_element_;
      unsigned int link_npy_size = appr_alg->cur_element_count * appr_alg->maxlevel_ * appr_alg->size_links_per_element_;
      unsigned int link_npy_stride = appr_alg->maxlevel_ * appr_alg->size_links_per_element_;

      char* data_level0_npy = (char *) malloc(level0_npy_size);
      char* link_list_npy = (char *) malloc(link_npy_size);

      memset(data_level0_npy, 0, level0_npy_size);
      memset(link_list_npy, 0, link_npy_size);

      memcpy(data_level0_npy, appr_alg->data_level0_memory_, level0_npy_size);


      for (size_t i = 0; i < appr_alg->cur_element_count; i++){
        unsigned int linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
        if (linkListSize){
          memcpy(link_list_npy+(link_npy_stride * i), appr_alg->linkLists_[i], linkListSize);
          // std::cout << linkListSize << " " << appr_alg->maxlevel_ << " " << appr_alg->element_levels_[i] << " generator: " << appr_alg->level_generator_ << std::endl;
        }
      }

      py::capsule free_when_done_l0(data_level0_npy, [](void *f) {
          delete[] f;
      });
      py::capsule free_when_done_ll(link_list_npy, [](void *f) {
          delete[] f;
      });

      return py::make_tuple(appr_alg->offsetLevel0_,
                            appr_alg->max_elements_,
                            appr_alg->cur_element_count,
                            appr_alg->size_data_per_element_,
                            appr_alg->label_offset_,
                            appr_alg->offsetData_,
                            appr_alg->maxlevel_,
                            appr_alg->enterpoint_node_,
                            appr_alg->maxM_,
                            appr_alg->maxM0_,
                            appr_alg->M_,
                            appr_alg->mult_,
                            appr_alg->ef_construction_,
                            appr_alg->ef_,
                            appr_alg->has_deletions_,
                            appr_alg->size_links_per_element_,
                            appr_alg->label_lookup_,
                            appr_alg->element_levels_,
                            py::array_t<char>(
                                    {level0_npy_size}, // shape
                                    {sizeof(char)}, // C-style contiguous strides for double
                                    data_level0_npy, // the data pointer
                                    free_when_done_l0),
                            py::array_t<char>(
                                    {link_npy_size}, // shape
                                    {sizeof(char)}, // C-style contiguous strides for double
                                    link_list_npy, // the data pointer
                                    free_when_done_ll)
                          );

    }


    py::tuple getIndexParams() const {
        return py::make_tuple(py::make_tuple(space_name, dim, index_inited, ep_added, normalize, num_threads_default, seed, default_ef),
                              index_inited == true ? getAnnData() : py::make_tuple());

    }


    static Index<float> * createFromParams(const py::tuple t) {
      py::tuple index_params=t[0].cast<py::tuple>();
      py::tuple ann_params=t[1].cast<py::tuple>();

      auto space_name_=index_params[0].cast<std::string>();
      auto dim_=index_params[1].cast<int>();
      auto index_inited_=index_params[2].cast<bool>();

      Index<float> *new_index = new Index<float>(index_params[0].cast<std::string>(), index_params[1].cast<int>());

      new_index->seed = index_params[6].cast<size_t>();


      if (index_inited_){
        ////                      hnswlib::HierarchicalNSW<dist_t>(l2space,            maxElements,                  M,                             efConstruction,                random_seed);
        new_index->appr_alg = new hnswlib::HierarchicalNSW<dist_t>(new_index->l2space, ann_params[1].cast<size_t>(), ann_params[10].cast<size_t>(), ann_params[12].cast<size_t>(), new_index->seed);
        new_index->cur_l = ann_params[2].cast<size_t>();
      }

      new_index->index_inited = index_inited_;
      new_index->ep_added=index_params[3].cast<bool>();
      new_index->num_threads_default=index_params[5].cast<int>();
      new_index->default_ef=index_params[7].cast<size_t>();

      if (index_inited_)
        new_index->setAnnData(ann_params);


      return new_index;
    }

    void setAnnData(const py::tuple t) {
      assert_true(appr_alg->offsetLevel0_ == t[0].cast<size_t>(), "Invalid value of offsetLevel0_ ");
      assert_true(appr_alg->max_elements_ == t[1].cast<size_t>(), "Invalid value of max_elements_ ");

      appr_alg->cur_element_count = t[2].cast<size_t>();

      assert_true(appr_alg->size_data_per_element_ == t[3].cast<size_t>(), "Invalid value of size_data_per_element_ ");
      assert_true(appr_alg->label_offset_ == t[4].cast<size_t>(), "Invalid value of label_offset_ ");
      assert_true(appr_alg->offsetData_ == t[5].cast<size_t>(), "Invalid value of offsetData_ ");

      appr_alg->maxlevel_ = t[6].cast<int>();
      appr_alg->enterpoint_node_ = t[7].cast<hnswlib::tableint>();

      assert_true(appr_alg->maxM_ == t[8].cast<size_t>(), "Invalid value of maxM_ ");
      assert_true(appr_alg->maxM0_ == t[9].cast<size_t>(), "Invalid value of maxM0_ ");
      assert_true(appr_alg->M_ == t[10].cast<size_t>(), "Invalid value of M_ ");
      assert_true(appr_alg->mult_ == t[11].cast<double>(), "Invalid value of mult_ ");
      assert_true(appr_alg->ef_construction_ == t[12].cast<size_t>(), "Invalid value of ef_construction_ ");

      appr_alg->ef_ = t[13].cast<size_t>();
      appr_alg->has_deletions_=t[14].cast<bool>();
      assert_true(appr_alg->size_links_per_element_ == t[15].cast<size_t>(), "Invalid value of size_links_per_element_ ");

      auto label_lookup_dict = t[16].cast<py::dict>();
      auto element_levels_list = t[17].cast<py::list>();
      auto data_level0_npy = t[18].cast<py::array_t<char>>();
      auto link_list_npy = t[19].cast<py::array_t<char>>();

      for (auto el: label_lookup_dict){
        appr_alg->label_lookup_.insert(
          std::make_pair(
                    el.first.cast<hnswlib::labeltype>(),
                    el.second.cast<hnswlib::tableint>()));
      }


      int idx = 0;
      for (auto el : element_levels_list){
        appr_alg->element_levels_[idx]=el.cast<int>();
        idx++;
      }


      memcpy(appr_alg->data_level0_memory_, data_level0_npy.data(), data_level0_npy.nbytes());

      for (size_t i = 0; i < appr_alg->max_elements_; i++) {
          unsigned int linkListSize = appr_alg->element_levels_[i] > 0 ? appr_alg->size_links_per_element_ * appr_alg->element_levels_[i] : 0;
          if (linkListSize == 0) {
              appr_alg->linkLists_[i] = nullptr;
          } else {
            appr_alg->linkLists_[i] = (char *) malloc(linkListSize);
            if (appr_alg->linkLists_[i] == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");

            memcpy(appr_alg->linkLists_[i], (link_list_npy.data()+(appr_alg->maxlevel_ * appr_alg->size_links_per_element_ * i)), linkListSize);

          }
      }


      // TODO: use global lock for de-/serialization
      // std::unique_lock <std::mutex> templock(global);
      // int maxlevelcopy = maxlevel_;
      // if (curlevel <= maxlevelcopy)
      //     templock.unlock();

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
                                            "Cannot return the results in a contigious 2D array. Probably ef or M is too small");
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
                                            "Cannot return the results in a contigious 2D array. Probably ef or M is too small");
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

    void markDeleted(size_t label) {
        appr_alg->markDelete(label);
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

};



PYBIND11_PLUGIN(hnswlib) {
        py::module m("hnswlib");

        // py::class_<Index<float>, std::shared_ptr<Index<float> >>(m, "Index")
        py::class_<Index<float>>(m, "Index")
        .def(py::init(&Index<float>::createFromParams), py::arg("params")) //createFromParams(const py::tuple t)
        .def(py::init<const std::string &, const int>(), py::arg("space"), py::arg("dim"))
        .def("init_index", &Index<float>::init_new_index, py::arg("max_elements"), py::arg("M")=16, py::arg("ef_construction")=200, py::arg("random_seed")=100)
        .def("knn_query", &Index<float>::knnQuery_return_numpy, py::arg("data"), py::arg("k")=1, py::arg("num_threads")=-1)
        .def("add_items", &Index<float>::addItems, py::arg("data"), py::arg("ids") = py::none(), py::arg("num_threads")=-1)
        .def("get_items", &Index<float, float>::getDataReturnList, py::arg("ids") = py::none())
        .def("get_ids_list", &Index<float>::getIdsList)
        .def("set_ef", &Index<float>::set_ef, py::arg("ef"))
        .def("set_num_threads", &Index<float>::set_num_threads, py::arg("num_threads"))
        .def("save_index", &Index<float>::saveIndex, py::arg("path_to_index"))
        .def("load_index", &Index<float>::loadIndex, py::arg("path_to_index"), py::arg("max_elements")=0)
        .def("mark_deleted", &Index<float>::markDeleted, py::arg("label"))
        .def("resize_index", &Index<float>::resizeIndex, py::arg("new_size"))
        .def_readonly("space_name", &Index<float>::space_name)
        .def_readonly("dim", &Index<float>::dim)
        .def_readwrite("num_threads", &Index<float>::num_threads_default)
        .def_property("ef",
          [](const Index<float> & index) {
            return index.index_inited ? index.appr_alg->ef_ : index.default_ef;
          },
          [](Index<float> & index, const size_t ef_) {
            // index.set_ef(ef_);
            index.default_ef=ef_;
            if (index.appr_alg)
              index.appr_alg->ef_ = ef_;

            // if (index.index_inited)
              // index.appr_alg->ef_ = ef_;
            // else
              // throw std::runtime_error("must call init_index prior to setting ef parameter");
        })
        .def_property_readonly("max_elements", [](const Index<float> & index) {
            return index.index_inited ? index.appr_alg->max_elements_ : 0;
        })
        .def_property_readonly("element_count", [](const Index<float> & index) {
            return index.index_inited ? index.appr_alg->cur_element_count : 0;
        })
        .def_property_readonly("ef_construction", [](const Index<float> & index) {
          return index.index_inited ? index.appr_alg->ef_construction_ : 0;
        })
        .def_property_readonly("M",  [](const Index<float> & index) {
          return index.index_inited ? index.appr_alg->M_ : 0;
        })

        .def(py::pickle(
            [](const Index<float> &ind) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return ind.getIndexParams();
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                return Index<float>::createFromParams(t);
            }
        ))

        .def("check_integrity", [](const Index<float> & index) {
          index.appr_alg->checkIntegrity();
          std::cout<< index.default_ef << " " << index.appr_alg->ef_ << std::endl;
          return index.appr_alg->ef_;
              // return index.getIndexParams();
              // return index.appr_alg->element_levels_;

              // std::stringstream output(std::stringstream::out|std::stringstream::binary);
              //
              // .def("get_params", &Index<float>::getIndexParams)
              // .def("set_params",  &Index<float>::setIndexParams,  py::arg("t"))// [](Index<float> & index, py::tuple t) {
              //
              // if (index.index_inited)
              //   index.saveIndexToStream(output);
              //
              // /* Return a tuple that fully encodes the state of the object */
              // return py::make_tuple(index.space_name, index.dim,
              //                       index.index_inited, index.ep_added,
              //                       index.normalize, index.num_threads_default,
              //                       py::bytes(output.str()),
              //                       index.index_inited == false ? 10 : index.appr_alg->ef_,
              //                       index.index_inited == false ? 0  : index.appr_alg->max_elements_,
              //                       index.index_inited == false ? 0  : index.appr_alg->cur_element_count
              //                     );
        })


                    // .def(py::pickle(
        //     [](const Index<float> & index) { // __getstate__
        //         /* Return a tuple that fully encodes the state of the object */
        //         return index.getIndexParams();
        //     },
        //     [](Index<float> & index, py::tuple t) { // __setstate__
        //         if (t.size() != 2)
        //             throw std::runtime_error("Invalid state!");
        //
        //         /* Invoke Index constructor (need to use in-place version) */
        //         // py::tuple index_params = t[0].cast<py::tuple>();
        //         // Index<float> new_index(index_params[0].cast<std::string>(), index_params[1].cast<int>());
        //         index.setIndexParams(t);
        //         return  index;
        //
        //         /* Create a new C++ instance */
        //         // Pickleable p(t[0].cast<std::string>());
        //
        //         /* Assign any additional state */
        //         // p.setExtra(t[1].cast<int>());
        //
        //         // return p;
        //     }
        // ))

        // .def("__getstate__", &Index<float>::getIndexParams) // __getstate__
        // .def("__setstate__", &Index<float>::setIndexParams)  // __setstate__
          // .def("__setstate__", [](Index<float> & index, py::tuple t) { // __setstate__
            // py::tuple index_params = t[0].cast<py::tuple>();
            // new (&index) Index<float>(index_params[0].cast<std::string>(), index_params[1].cast<int>());
            // index.setIndexParams(t);
            // return index;
        // })
        // .def("__getstate__", [](const Index<float> & index) { // __getstate__
        //       return index.getIndexParams();
        //
        //       // std::stringstream output(std::stringstream::out|std::stringstream::binary);
        //       //
        //       // .def("get_params", &Index<float>::getIndexParams)
        //       // .def("set_params",  &Index<float>::setIndexParams,  py::arg("t"))// [](Index<float> & index, py::tuple t) {
        //       //
        //       // if (index.index_inited)
        //       //   index.saveIndexToStream(output);
        //       //
        //       // /* Return a tuple that fully encodes the state of the object */
        //       // return py::make_tuple(index.space_name, index.dim,
        //       //                       index.index_inited, index.ep_added,
        //       //                       index.normalize, index.num_threads_default,
        //       //                       py::bytes(output.str()),
        //       //                       index.index_inited == false ? 10 : index.appr_alg->ef_,
        //       //                       index.index_inited == false ? 0  : index.appr_alg->max_elements_,
        //       //                       index.index_inited == false ? 0  : index.appr_alg->cur_element_count
        //       //                     );
        // })
        // .def("set_state", [](Index<float> & index, py::tuple t) { // __setstate__
        //   index.setIndexParams(t);
        // })
        //
        // .def("__setstate__", [](Index<float> & index, py::tuple t) { // __setstate__
        //       // delete &index;
        //       /* Invoke Index constructor (need to use in-place version) */
        //       // py::tuple index_params = t[0].cast<py::tuple>();
        //       // new (&index) Index<float>(index_params[0].cast<std::string>(), index_params[1].cast<int>());
        //       index.setIndexParams(t);
        //       // if (t.size() != 10)
        //       //     throw std::runtime_error("Invalid state!");
        //       //
        //
        //       // index.index_inited=t[2].cast<bool>();
        //       // index.ep_added=t[3].cast<bool>();
        //       // index.normalize=t[4].cast<bool>();
        //       // index.num_threads_default=t[5].cast<int>();
        //       //
        //       // if (index.index_inited){
        //       //   std::stringstream input(t[6].cast<std::string>(), std::stringstream::in|std::stringstream::binary);
        //       //   index.loadIndexFromStream(input, t[8].cast<int>()); // use max_elements from state
        //       //   index.appr_alg->ef_=(t[7].cast<size_t>());
        //       // }
        //
        // })
        .def("__repr__", [](const Index<float> &a) {
            return "<hnswlib.Index(space='" + a.space_name + "', dim="+std::to_string(a.dim)+")>";
        });

        return m.ptr();
}
