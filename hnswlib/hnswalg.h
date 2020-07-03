#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include "mmap.h"
#include <atomic>
#include <cassert>
#include <random>
#include <stdlib.h>
#include <unordered_set>
#include <list>


namespace hnswlib {
    typedef unsigned int tableint;
    typedef unsigned char flagint;
    typedef unsigned char linklistsizeint;

    template<typename dist_t, typename labeltype=size_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t, labeltype> {
        const int INDEX_MAGIC_NUMBER = 0x484e5357;  //HNSW
        const int INDEX_VERSION = 2;

    public:
        HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, bool mmap=false, size_t max_elements=0,
                        tableint max_update_element_locks=65536) :
                link_list_update_locks_(max_update_element_locks) {
            if (mmap) mmapLoad(location, s);
            else loadIndex(location, s, max_elements);
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, linklistsizeint M=16,
                        linklistsizeint ef_construction=200, size_t random_seed=100, tableint max_update_element_locks=65536) :
                link_list_locks_(max_elements), link_list_update_locks_(max_update_element_locks) {
            if (M > std::numeric_limits<linklistsizeint>::max() / 2)
                throw new std::runtime_error("M is too large");

            max_elements_ = max_elements;
            element_levels_ = (linklistsizeint *) malloc(sizeof(linklistsizeint) * max_elements_);

            has_deletions_ = false;
            space_ = s;
            M_ = M;
            maxM_ = M_;
            maxM0_ = M_ * 2;
            ef_construction_ = std::max(ef_construction,M_);
            ef_ = 10;

            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);

//             struct Data {
//                 flagint flags;
//                 linklistsizeint link_count;
//                 tableint[maxM0_] links;
//                 dist_t[space_->get_dimension()] vector;
//                 labeltype label;
//             };
            size_t size_links_level0 = sizeof(flagint) + sizeof(linklistsizeint) + maxM0_ * sizeof(tableint);
            size_data_per_element_ = size_links_level0 + sizeof(dist_t) * space_->get_dimension() + sizeof(labeltype);
            data_offset_ = size_links_level0;
            label_offset_ = size_links_level0 + sizeof(dist_t) * space_->get_dimension();

            data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            cur_element_count_ = 0;

            visited_list_pool_ = new VisitedListPool(1, max_elements);



            //initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            // struct LinkData {
            //     linklistsizeint link_count;
            //     tableint[maxM_] links;
            // };
            linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");

            size_links_per_element_ = sizeof(linklistsizeint) + maxM_ * sizeof(tableint);
            mult_ = 1 / log(1.0 * M_);
        }

        struct CompareByFirst {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept {
                return a.first < b.first;
            }
        };

        ~HierarchicalNSW() {
            if (!mmap_.isValid()) {
                free(data_level0_memory_);
                for (tableint i = 0; i < cur_element_count_; i++) {
                    if (element_levels_[i] > 0)
                        free(linkLists_[i]);
                }
                free(element_levels_);
            }
            free(linkLists_);
            delete visited_list_pool_;
        }

        size_t max_elements_;
        size_t cur_element_count_;
        size_t size_data_per_element_;
        size_t size_links_per_element_;
        size_t data_offset_;
        size_t label_offset_;

        linklistsizeint M_;
        linklistsizeint maxM_;
        linklistsizeint maxM0_;
        size_t ef_construction_;
        size_t ef_;
        double mult_;
        int maxlevel_;

        VisitedListPool *visited_list_pool_;
        std::mutex cur_element_count_guard_;
        std::mutex enterpoint_node_guard_;

        std::vector<std::mutex> link_list_locks_;

        // Locks to prevent race condition during update/insert of an element at same time.
        // Note: Locks for additions can also be used to prevent this race condition if the querying of KNN is not exposed along with update/inserts i.e multithread insert/update/query in parallel.
        std::vector<std::mutex> link_list_update_locks_;
        tableint enterpoint_node_;

        char *data_level0_memory_;
        char **linkLists_;
        linklistsizeint *element_levels_;

        bool has_deletions_;

        SpaceInterface<dist_t> *space_;
        MMap mmap_;
        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;
        std::unordered_map<labeltype, tableint> label_lookup_;


        inline flagint *getFlagsById(tableint internal_id) const {
            return (flagint *) (data_level0_memory_ + internal_id * size_data_per_element_);
        };

        inline linklistsizeint *get_linklist0(tableint internal_id) const {
            return (linklistsizeint *) (getFlagsById(internal_id) + 1);
        };

        inline linklistsizeint *get_linklist(tableint internal_id, int level) const {
            return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
        };

        inline linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        };

        inline labeltype& getExternalLabel(tableint internal_id) const {
            return *getExternalLabelP(internal_id);
        }

        inline labeltype *getExternalLabelP(tableint internal_id) const {
            return (labeltype *) (((char *) getFlagsById(internal_id)) + label_offset_);
        }

        inline dist_t *getDataByInternalId(tableint internal_id) const {
            return (dist_t *) (((char *) getFlagsById(internal_id)) + data_offset_);
        }

        inline tableint *getDataListP(linklistsizeint *list) const {
            return (tableint *) (list + 1);
        }

        int getRandomLevel(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int) r;
        }


        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const dist_t *data_point, int layer) {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id)) {
                dist_t dist = space_->calculate_distance(data_point, getDataByInternalId(ep_id));
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty()) {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound) {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

                linklistsizeint *data;
                if (layer == 0) {
                    data = get_linklist0(curNodeNum);
                } else {
                    data = get_linklist(curNodeNum, layer);
                }
                linklistsizeint size = *data;
                tableint *datal = getDataListP(data);

                for (linklistsizeint j = 0; j < size; j++) {
                    tableint candidate_id = datal[j];

                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
                    const dist_t *currObj1 = getDataByInternalId(candidate_id);

                    dist_t dist1 = space_->calculate_distance(data_point, currObj1);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

        mutable std::atomic<long> metric_distance_computations;
        mutable std::atomic<long> metric_hops;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerST(tableint ep_id, const dist_t *data_point, size_t ef, bool collect_metrics=false) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            if (!has_deletions_ || !isMarkedDeleted(ep_id)) {
                dist_t dist = space_->calculate_distance(data_point, getDataByInternalId(ep_id));
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            while (!candidate_set.empty()) {

                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                if ((-current_node_pair.first) > lowerBound) {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                linklistsizeint *data = get_linklist0(current_node_id);
                linklistsizeint size = *data;
                tableint *list = getDataListP(data);

                if(collect_metrics){
                    metric_hops++;
                    metric_distance_computations+=size;
                }

                for (linklistsizeint j = 0; j < size; j++) {
                    tableint candidate_id = list[j];
                    if (!(visited_array[candidate_id] == visited_array_tag)) {

                        visited_array[candidate_id] = visited_array_tag;

                        const dist_t *currObj1 = getDataByInternalId(candidate_id);
                        dist_t dist = space_->calculate_distance(data_point, currObj1);

                        if (top_candidates.size() < ef || lowerBound > dist) {
                            candidate_set.emplace(-dist, candidate_id);

                            if (!has_deletions_ || !isMarkedDeleted(candidate_id))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
            if (top_candidates.size() < M) {
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size()) {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;

                for (std::pair<dist_t, tableint> second_pair : return_list) {
                    dist_t curdist =
                            space_->calculate_distance(getDataByInternalId(second_pair.second),
                                                       getDataByInternalId(curent_pair.second));
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good) {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }

        tableint mutuallyConnectNewElement(const dist_t *data_point, tableint cur_c,
                                           std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
                                           linklistsizeint level,
                                           bool isUpdate) {

            linklistsizeint Mcurmax = level ? maxM_ : maxM0_;
            getNeighborsByHeuristic2(top_candidates, M_);
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors[0];
            {
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur && !isUpdate) {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                *ll_cur = selectedNeighbors.size();
                tableint *data = getDataListP(ll_cur);

                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];
                }
            }

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

                std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                linklistsizeint sz_link_list_other = *ll_other;

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                tableint *data = getDataListP(ll_other);

                bool is_cur_c_present = false;
                if (isUpdate) {
                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        if (data[j] == cur_c) {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
                if (!is_cur_c_present) {
                    if (sz_link_list_other < Mcurmax) {
                        data[sz_link_list_other] = cur_c;
                        *ll_other = sz_link_list_other + 1;
                    } else {
                        // finding the "weakest" element to replace it with the new one
                        dist_t d_max = space_->calculate_distance(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]));
                        // Heuristic:
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (linklistsizeint j = 0; j < sz_link_list_other; j++) {
                            candidates.emplace(space_->calculate_distance(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx])),
                                               data[j]);
                        }

                        getNeighborsByHeuristic2(candidates, Mcurmax);

                        linklistsizeint indx = 0;
                        while (candidates.size() > 0) {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }
                        *ll_other = indx;
                        // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            dist_t d = space_->calculate_distance(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]));
                            if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
            }

            return next_closest_entry_point;
        }

        void setEf(size_t ef) {
            ef_ = ef;
        }

        void resizeIndex(size_t new_max_elements){
            if (mmap_.isValid())
                throw std::runtime_error("Cannot resize, index loaded via mmap.");

            if (new_max_elements<cur_element_count_)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");


            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, new_max_elements);

            element_levels_ = (linklistsizeint *) realloc(element_levels_, new_max_elements * sizeof(linklistsizeint));
            if (element_levels_ == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate element levels");

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char * data_level0_memory_new = (char *) malloc(new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
            memcpy(data_level0_memory_new, data_level0_memory_, cur_element_count_ * size_data_per_element_);
            free(data_level0_memory_);
            data_level0_memory_=data_level0_memory_new;

            // Reallocate all other layers
            char ** linkLists_new = (char **) malloc(sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
            memcpy(linkLists_new, linkLists_, cur_element_count_ * sizeof(void *));
            free(linkLists_);
            linkLists_=linkLists_new;

            max_elements_=new_max_elements;
        }

        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            saveIndex(output);
            output.close();
        }

        void saveIndex(std::ostream &output) {
            writeBinaryPOD(output, INDEX_MAGIC_NUMBER);
            writeBinaryPOD(output, INDEX_VERSION);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count_);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, data_offset_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);
            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);

            output.write(data_level0_memory_, cur_element_count_ * size_data_per_element_);
            checkIOError(output);

            if (cur_element_count_ > 0) {
                output.write((const char *) element_levels_, cur_element_count_ * sizeof(linklistsizeint));
                checkIOError(output);
            }
            for (size_t i = 0; i < cur_element_count_; i++) {
                if (element_levels_[i]) {
                    output.write(linkLists_[i], size_links_per_element_ * element_levels_[i]);
                    checkIOError(output);
                }
            }
        }

        void mmapLoad(const std::string &location, SpaceInterface<dist_t> *s) {
            char *input = mmap_.load(location);

            int actual_magic_number, actual_version;
            input = readBinaryMmap(input, actual_magic_number);
            input = readBinaryMmap(input, actual_version);
            if (INDEX_MAGIC_NUMBER != actual_magic_number || INDEX_VERSION != actual_version)
                throw std::runtime_error("Wrong index file type or version.");

            input = readBinaryMmap(input, max_elements_);
            input = readBinaryMmap(input, cur_element_count_);
            max_elements_ = cur_element_count_; //The memory is read-only, can't add more

            input = readBinaryMmap(input, size_data_per_element_);
            input = readBinaryMmap(input, label_offset_);
            input = readBinaryMmap(input, data_offset_);
            input = readBinaryMmap(input, maxlevel_);
            input = readBinaryMmap(input, enterpoint_node_);
            input = readBinaryMmap(input, maxM_);
            input = readBinaryMmap(input, maxM0_);
            input = readBinaryMmap(input, M_);
            input = readBinaryMmap(input, mult_);
            input = readBinaryMmap(input, ef_construction_);

            space_ = s;

            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            visited_list_pool_ = new VisitedListPool(1, max_elements_);
            std::vector<std::mutex>(max_elements_).swap(link_list_locks_);
            ef_ = 10;
            has_deletions_ = false;

            data_level0_memory_ = input;
            input += cur_element_count_ * size_data_per_element_;
            checkMmapBufferValidity(input);

            element_levels_ = (linklistsizeint *) input;
            input += cur_element_count_ * sizeof(linklistsizeint);
            checkMmapBufferValidity(input);

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
            for (size_t i = 0; i < cur_element_count_; i++) {
                label_lookup_[getExternalLabel(i)] = i;

                if (element_levels_[i] == 0) {
                    linkLists_[i] = nullptr;
                } else {
                    linkLists_[i] = input;
                    input += element_levels_[i] * size_links_per_element_;
                    checkMmapBufferValidity(input);
                }
            }

            for (size_t i = 0; i < cur_element_count_; i++) {
                if(isMarkedDeleted(i))
                    has_deletions_=true;
            }
        }

        template<typename T>
        char* readBinaryMmap(char *addr, T &podRef) {
            podRef = *((T*) addr);
            return checkMmapBufferValidity(addr + sizeof(T));
        }

        char* checkMmapBufferValidity(char *new_ptr) {
            if (!mmap_.isInBuffer(new_ptr))
                throw std::runtime_error("Error loading index: eof reached before expectation.");
            return new_ptr;
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i=0) {


            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            int actual_magic_number, actual_version;
            readBinaryPOD(input, actual_magic_number);
            readBinaryPOD(input, actual_version);
            if (INDEX_MAGIC_NUMBER != actual_magic_number || INDEX_VERSION != actual_version)
                throw std::runtime_error("Wrong index file type or version.");

            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count_);

            size_t max_elements=max_elements_i;
            if(max_elements < cur_element_count_)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, data_offset_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);

            space_ = s;

            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            visited_list_pool_ = new VisitedListPool(1, max_elements_);
            std::vector<std::mutex>(max_elements_).swap(link_list_locks_);
            ef_ = 10;
            has_deletions_ = false;

            data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input.read(data_level0_memory_, cur_element_count_ * size_data_per_element_);
            checkIOError(input);

            element_levels_ = (linklistsizeint *) malloc(sizeof(linklistsizeint) * max_elements);
            if (element_levels_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate element_levels_");
            input.read((char *) element_levels_, cur_element_count_ * sizeof(linklistsizeint));
            checkIOError(input);

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            for (size_t i = 0; i < cur_element_count_; i++) {
                label_lookup_[getExternalLabel(i)] = i;
                if (element_levels_[i] == 0) {
                    linkLists_[i] = nullptr;
                } else {
                    size_t linkListSize = element_levels_[i] * size_links_per_element_;
                    linkLists_[i] = (char *) malloc(linkListSize);
                    if (linkLists_[i] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    input.read(linkLists_[i], linkListSize);
                    checkIOError(input);
                }
            }

            for (size_t i = 0; i < cur_element_count_; i++) {
                if(isMarkedDeleted(i))
                    has_deletions_=true;
            }
            input.close();
        }

        dist_t* getDataByLabel(const labeltype& label) {
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
                throw std::runtime_error("Label not found");
            }
            return getDataByInternalId(search->second);
            // const dist_t* data_ptr = getDataByInternalId(search->second);
            // return std::vector<dist_t>(data_ptr, data_ptr + space_->get_dimension());
        }

        static const flagint DELETE_MARK = 0x01;
//        static const flagint REUSE_MARK = 0x10;
        /**
         * Marks an element with the given label deleted, does NOT really change the current graph.
         * @param label
         */
        void markDelete(labeltype& label) {
            has_deletions_=true;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            markDeletedInternal(search->second);
        }

        /**
         * Uses the first 8 bits of the memory for the linked list to store the mark,
         * whereas maxM0_ has to be limited to the lower 24 bits, however, still large enough in almost all cases.
         * @param internalId
         */
        void markDeletedInternal(tableint internalId) {
            *getFlagsById(internalId) |= DELETE_MARK;
        }

        /**
         * Remove the deleted mark of the node.
         * @param internalId
         */
        void unmarkDeletedInternal(tableint internalId) {
            *getFlagsById(internalId) &= ~DELETE_MARK;
        }

        /**
         * Checks the first 8 bits of the memory to see if the element is marked deleted.
         * @param internalId
         * @return
         */
        bool isMarkedDeleted(tableint internalId) const {
            return *getFlagsById(internalId) & DELETE_MARK;
        }

        void updatePoint(const dist_t *dataPoint, tableint internalId, float updateNeighborProbability) {
            // update the feature vector associated with existing point with new vector
            memcpy(getDataByInternalId(internalId), dataPoint, sizeof(dist_t) * space_->get_dimension());

            int maxLevelCopy = maxlevel_;
            tableint entryPointCopy = enterpoint_node_;
            // If point to be updated is entry point and graph just contains single element then just return.
            if (entryPointCopy == internalId && cur_element_count_ == 1)
                return;

            int elemLevel = element_levels_[internalId];
            std::uniform_real_distribution<float> distribution(0.0, 1.0);
            for (int layer = 0; layer <= elemLevel; layer++) {
                std::unordered_set<tableint> sCand;
                std::unordered_set<tableint> sNeigh;
                std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
                if (listOneHop.size() == 0)
                    continue;

                sCand.insert(internalId);

                for (auto&& elOneHop : listOneHop) {
                    sCand.insert(elOneHop);

                    if (distribution(update_probability_generator_) > updateNeighborProbability)
                        continue;

                    sNeigh.insert(elOneHop);

                    std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                    for (auto&& elTwoHop : listTwoHop) {
                        sCand.insert(elTwoHop);
                    }
                }

                for (auto&& neigh : sNeigh) {
//                    if (neigh == internalId)
//                        continue;

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    int size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;
                    int elementsToKeep = std::min(int(ef_construction_), size);
                    for (auto&& cand : sCand) {
                        if (cand == neigh)
                            continue;

                        dist_t distance = space_->calculate_distance(getDataByInternalId(neigh), getDataByInternalId(cand));
                        if (candidates.size() < elementsToKeep) {
                            candidates.emplace(distance, cand);
                        } else {
                            if (distance < candidates.top().first) {
                                candidates.pop();
                                candidates.emplace(distance, cand);
                            }
                        }
                    }

                    // Retrieve neighbours using heuristic and set connections.
                    getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                    {
                        std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                        linklistsizeint *ll_cur = get_linklist_at_level(neigh, layer);
                        int candSize = candidates.size();
                        *ll_cur = candSize;
                        tableint *data = getDataListP(ll_cur);
                        for (int idx = 0; idx < candSize; idx++) {
                            data[idx] = candidates.top().second;
                            candidates.pop();
                        }
                    }
                }
            }

            repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
        };

        void repairConnectionsForUpdate(const dist_t *dataPoint, tableint entryPointInternalId, tableint dataPointInternalId, int dataPointLevel, int maxLevel) {
            tableint currObj = entryPointInternalId;
            if (dataPointLevel < maxLevel) {
                dist_t curdist = space_->calculate_distance(dataPoint, getDataByInternalId(currObj));
                for (int level = maxLevel; level > dataPointLevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        linklistsizeint *data = get_linklist_at_level(currObj, level);
                        linklistsizeint size = *data;
                        tableint *datal = getDataListP(data);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                        for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                            tableint cand = datal[i];
                            dist_t d = space_->calculate_distance(dataPoint, getDataByInternalId(cand));
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (dataPointLevel > maxLevel)
                throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

            for (int level = dataPointLevel; level >= 0; level--) {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                        currObj, dataPoint, level);

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
                while (topCandidates.size() > 0) {
                    if (topCandidates.top().second != dataPointInternalId)
                        filteredTopCandidates.push(topCandidates.top());

                    topCandidates.pop();
                }

                // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
                // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
                if (filteredTopCandidates.size() > 0) {
                    bool epDeleted = isMarkedDeleted(entryPointInternalId);
                    if (epDeleted) {
                        filteredTopCandidates.emplace(space_->calculate_distance(dataPoint, getDataByInternalId(entryPointInternalId)), entryPointInternalId);
                        if (filteredTopCandidates.size() > ef_construction_)
                            filteredTopCandidates.pop();
                    }

                    currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
                }
            }
        }

        std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
            std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
            linklistsizeint *data = get_linklist_at_level(internalId, level);
            linklistsizeint size = *data;
            std::vector<tableint> result(size);
            tableint *ll = getDataListP(data);
            memcpy(result.data(), ll, size * sizeof(tableint));
            return result;
        };

        void addPoint(const dist_t *data_point, const labeltype& label) {
            if (mmap_.isValid())
                throw std::runtime_error("Cannot add, index loaded via mmap.");

            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);

                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end()) {
                    tableint existingInternalId = search->second;

                    templock_curr.unlock();

                    std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (link_list_update_locks_.size() - 1))]);
                    updatePoint(data_point, existingInternalId, 1.0);
                    return;
                }

                if (cur_element_count_ >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                label_lookup_[label] = cur_element_count_;
                cur_c = cur_element_count_;
                cur_element_count_++;
            }

            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (link_list_update_locks_.size() - 1))]);
            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);

            element_levels_[cur_c] = curlevel;


            std::unique_lock <std::mutex> templock(enterpoint_node_guard_);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;


            memset(getFlagsById(cur_c), 0, size_data_per_element_);

            // Initialisation of the data and label
            memcpy(getExternalLabelP(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, sizeof(dist_t) * space_->get_dimension());


            if (curlevel) {
                linkLists_[cur_c] = (char *) calloc(curlevel, size_links_per_element_);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            }

            if ((signed)currObj != -1) {

                if (curlevel < maxlevelcopy) {

                    dist_t curdist = space_->calculate_distance(data_point, getDataByInternalId(currObj));
                    for (int level = maxlevelcopy; level > curlevel; level--) {
                        bool changed = true;
                        while (changed) {
                            changed = false;
                            std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                            linklistsizeint *data = get_linklist(currObj,level);
                            linklistsizeint size = *data;
                            tableint *datal = getDataListP(data);
                            for (linklistsizeint i = 0; i < size; i++) {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = space_->calculate_distance(data_point, getDataByInternalId(cand));
                                if (d < curdist) {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                    if (level > maxlevelcopy || level < 0)  // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level);
                    if (epDeleted) {
                        top_candidates.emplace(space_->calculate_distance(data_point, getDataByInternalId(enterpoint_copy)), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
                }
            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;
            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
        };

        typedef typename AlgorithmInterface<dist_t, labeltype>::Neighbour neighbour_t;
        std::vector<neighbour_t> searchKnn(const dist_t *query_data, size_t k) const {
            if (cur_element_count_ == 0) return std::vector<neighbour_t>();

            std::vector<neighbour_t> result(std::min(k, cur_element_count_));
            searchKnn(query_data, k, result.data());
            return result;
        }

        void searchKnn(const dist_t *query_data, size_t k, neighbour_t *result_buffer) const {
            tableint currObj = enterpoint_node_;
            dist_t curdist = space_->calculate_distance(query_data, getDataByInternalId(enterpoint_node_));

            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    linklistsizeint *data = get_linklist(currObj, level);
                    linklistsizeint size = *data;
                    tableint *datal = getDataListP(data);

                    metric_hops++;
                    metric_distance_computations+=size;

                    for (linklistsizeint i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = space_->calculate_distance(query_data, getDataByInternalId(cand));

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            top_candidates = searchBaseLayerST(currObj, query_data, std::max(ef_, k), true);
            while (top_candidates.size() > k) {
                top_candidates.pop();
            }

            if (top_candidates.size() != k) {
                throw std::runtime_error("Unable to retrieve k neighbours. Set a higher ef.");
            }

            for (int i = k - 1; i >= 0; i--) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result_buffer[i] = {rez.first, getExternalLabel(rez.second)};
                top_candidates.pop();
            }
        }

        void checkIntegrity() const {
            int connections_checked=0;
            std::vector <int > inbound_connections_num(cur_element_count_, 0);
            for(int i = 0;i < cur_element_count_; i++){
                for(int l = 0;l <= element_levels_[i]; l++){
                    linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                    linklistsizeint size = *ll_cur;
                    tableint *data = getDataListP(ll_cur);
                    std::unordered_set<tableint> s;
                    for (int j=0; j<size; j++){
                        assert(data[j] >= 0);
                        assert(data[j] < cur_element_count_);
                        assert(data[j] != i);
                        inbound_connections_num[data[j]]++;
                        s.insert(data[j]);
                        connections_checked++;
                    }
                    assert(s.size() == size);
                }
            }
            if(cur_element_count_ > 1){
                int min1=inbound_connections_num[0], max1=inbound_connections_num[0];
                for(int i=0; i < cur_element_count_; i++){
                    assert(inbound_connections_num[i] > 0);
                    min1=std::min(inbound_connections_num[i],min1);
                    max1=std::max(inbound_connections_num[i],max1);
                }
                std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }
            std::cout << "integrity ok, checked " << connections_checked << " connections\n";

        }

    };

}
