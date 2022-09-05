
#ifndef HNSWPF_CORE_H_
#define HNSWPF_CORE_H_

#include <dirent.h>
#include <ftw.h>
#include "hnswlib.h"
#include "hnswalg.h"

#include "hnsw_tags.h"
#include "hnsw_query.h"

namespace hnswlib {


const static std::string INDEX_NAME_VEC = "hnsw_vec.bin";
const static std::string INDEX_NAME_TAG = "hnsw_tag.bin";


template<typename dist_t>
class SearchHNSW : public HierarchicalNSW<dist_t> {
public:
    // parent param
    using HierarchicalNSW<dist_t>::element_levels_;
    using HierarchicalNSW<dist_t>::cur_element_count;
    using HierarchicalNSW<dist_t>::getDataByInternalId;
    using HierarchicalNSW<dist_t>::dist_func_param_;
    using HierarchicalNSW<dist_t>::get_linklist;
    using HierarchicalNSW<dist_t>::getListCount;
    using HierarchicalNSW<dist_t>::metric_hops;
    using HierarchicalNSW<dist_t>::metric_distance_computations;
    using HierarchicalNSW<dist_t>::max_elements_;
    using HierarchicalNSW<dist_t>::num_deleted_;
    using HierarchicalNSW<dist_t>::ef_;
    using HierarchicalNSW<dist_t>::enterpoint_node_;
    using HierarchicalNSW<dist_t>::getExternalLabel;
    using HierarchicalNSW<dist_t>::label_lookup_;
    using HierarchicalNSW<dist_t>::visited_list_pool_;
    using HierarchicalNSW<dist_t>::isMarkedDeleted;
    using HierarchicalNSW<dist_t>::get_linklist0;
    using HierarchicalNSW<dist_t>::data_level0_memory_;
    using HierarchicalNSW<dist_t>::size_data_per_element_;
    using HierarchicalNSW<dist_t>::offsetData_;
    using HierarchicalNSW<dist_t>::offsetLevel0_;
    using HierarchicalNSW<dist_t>::fstdistfunc_;
    using HierarchicalNSW<dist_t>::searchKnn;
    using HierarchicalNSW<dist_t>::maxM_;
    using HierarchicalNSW<dist_t>::maxM0_;
    using HierarchicalNSW<dist_t>::maxlevel_;

    // child param
    // invert dic
    std::unordered_map<std::string, hnswlib::TagIndex> tag_index_dict;

    // tag lock
    std::mutex  tag_lock;


public:
    SearchHNSW(SpaceInterface <dist_t> *s,
                size_t max_elements, size_t M = 16,
                size_t ef_construction = 200, size_t random_seed = 100) :
                HierarchicalNSW<dist_t>(s, max_elements, M, ef_construction, random_seed) {}

    void addPoint(const void *data_point, labeltype external_id) {
        HierarchicalNSW<dist_t>::addPoint(data_point, external_id, -1);
    }


    //  addFieldTag support parallel
    void addFieldTag(int external_id, const std::string &field, int tag) {
        // 1、double check field
        TagIndex &tagIndex = tag_index_dict[field];
        if (tagIndex.name.empty()) {
            std::lock_guard<std::mutex> lock(tag_lock);
            if (tagIndex.name.empty()) {
                tagIndex.name = field;
                tagIndex.tag_to_externals[tag] = {external_id};
                return;
            }
        }

        // 2、insert tag
        tagIndex.insert(tag, external_id);
    }


    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, const std::list<QueryFilter> &queryFilters) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_) {
                        throw std::runtime_error("cand error");
                    }

                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        if (num_deleted_) {
            top_candidates = searchBaseLayerST<true, true>(currObj, query_data, std::max(ef_, k), queryFilters);
        } else {
            top_candidates = searchBaseLayerST<false, true>(currObj, query_data, std::max(ef_, k), queryFilters);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            const std::pair<dist_t, tableint> &rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    inline bool checkCondition(const tableint &external_id, const std::list<QueryFilter> &queryFilters) const {
        if (queryFilters.empty()) {
            return true;
        }

        bool result = true;
        for (const QueryFilter &queryFilter: queryFilters) {
            bool tmp_res = false;
            if (queryFilter.excludes.empty() && queryFilter.includes.empty()) {
                continue;
            }

            // id pre filter
            if (queryFilter.name == "ID") {
                // tag check： Subset is or
                if (queryFilter.excludes.find(external_id) == queryFilter.excludes.end()) {
                    continue;
                }
                if (queryFilter.includes.find(external_id) != queryFilter.includes.end()) {
                    continue;
                }
            } else {
                const auto &it = tag_index_dict.find(queryFilter.name);
                if (!queryFilter.excludes.empty() && it == tag_index_dict.end()) {
                    continue;
                }

                if (it != tag_index_dict.end()) {
                    const TagIndex &tag_index = it->second;
                    for (int tag : queryFilter.includes) {
                        const auto &tag_candidates_it = tag_index.tag_to_externals.find(tag);
                        if (tag_candidates_it != tag_index.tag_to_externals.end()) {
                            const std::unordered_set<int> &tag_candidates = tag_candidates_it->second;
                            if (tag_candidates.count((int) external_id) > 0) {
                                tmp_res = true;
                                break;
                            }
                        }
                    }

                    if (tmp_res) {
                        continue;
                    }

                    for (int tag : queryFilter.excludes) {
                        const auto &tag_candidates_it = tag_index.tag_to_externals.find(tag);
                        if (tag_candidates_it == tag_index.tag_to_externals.end()) {
                            tmp_res = true;
                            break;
                        }
                        if (tag_candidates_it != tag_index.tag_to_externals.end()) {
                            const std::unordered_set<int> &tag_candidates = tag_candidates_it->second;
                            if (tag_candidates.count((int) external_id) == 0) {
                                tmp_res = true;
                                break;
                            }
                        }
                    }
                    if (tmp_res) {
                        continue;
                    }
                }
            }

            return false;
        }
        return result;
    }


    tableint getInternalId(int external_id) const {
        auto search = label_lookup_.find(external_id);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        return search->second;
    }

    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                  std::pair<dist_t, tableint> const &b) const noexcept {
            return a.first < b.first;
        }
    };


    template<bool has_deletions, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef,
                      const std::list<QueryFilter> &queryFilters) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if ((!has_deletions || !isMarkedDeleted(ep_id)) && checkCondition(getExternalLabel(ep_id), queryFilters)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;
        bool has_filter = false;
        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

            if ((-current_node_pair.first) > lowerBound &&
                (top_candidates.size() == ef || (has_deletions == false && !has_filter))) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint *) data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations += size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                             _MM_HINT_T0);////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {

                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    if (top_candidates.size() < ef || lowerBound > dist) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                     offsetLevel0_,///////////
                                     _MM_HINT_T0);////////////////////////
#endif

                        if (!has_deletions || !isMarkedDeleted(candidate_id)) {
                            if (checkCondition(getExternalLabel(candidate_id), queryFilters)) {
                                top_candidates.emplace(dist, candidate_id);
                            } else {
                                has_filter = true;
                            }
                        }

                        if (top_candidates.size() > ef) {
                            top_candidates.pop();
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }


    // saveIndex
    void saveIndex(const std::string &dir) {
        // if not exist, then create
        if (nullptr == opendir(dir.c_str())) {
            mkdir(dir.c_str(), 0755);
        }
        std::string vec_index_path = dir + "/" + INDEX_NAME_VEC;
        std::string tag_index_path = dir + "/" + INDEX_NAME_TAG;

        HierarchicalNSW<dist_t>::saveIndex(vec_index_path);

        std::ofstream output(tag_index_path, std::ios::binary);
        int map_size = tag_index_dict.size();
        writeBinaryPOD(output, map_size);

        for (auto &tag_index : tag_index_dict) {
            TagIndex::writeString(output, tag_index.first);
            tag_index.second.serialize(output);
        }
        output.close();
    }


    // loadIndex
    void loadIndex(const std::string &dir, SpaceInterface <dist_t> *s, size_t max_elements_i = 0) {
        std::string vec_index_path = dir + "/" + INDEX_NAME_VEC;
        std::string tag_index_path = dir + "/" + INDEX_NAME_TAG;

        HierarchicalNSW<dist_t>::loadIndex(vec_index_path, s, max_elements_i);

        std::ifstream input(tag_index_path, std::ios::binary);
        int map_size = 0;
        readBinaryPOD(input, map_size);

        for (int i = 0; i < map_size; i++) {
            std::string name = TagIndex::readString(input);
            TagIndex &tag_index = tag_index_dict[name];
            tag_index.name = name;
            tag_index.deserialize(input);
        }
        input.close();
    }

};



} // namespace hnswlib

#endif  // _HNSWPF_CORE_H_
/* vim: set ts=4 sw=4 sts=4 tw=100 */


