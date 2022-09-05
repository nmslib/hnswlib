
#ifndef HNSWPF_QUERY_H_
#define HNSWPF_QUERY_H_


namespace hnswlib {


class QueryFilter {
public:
    QueryFilter() : name("ID") { }
    QueryFilter(std::string &n) : name(n) {}

    /**
     * add a tag search
     * @param required, true need, false discard
     * @param id ID
     * @return
     */
    int add(int id, bool required) {
        if (required) {
            includes.insert(id);
        }
        if(!required){
            excludes.insert(id);
        }
        return 0;
    }

public:
    std::string name;

    std::unordered_set<int> includes;
    std::unordered_set<int> excludes;
};


class QueryFilterList {
public:
    QueryFilterList() { };

    int add(QueryFilter* queryFilter) {
        filters.push_back(*queryFilter);
        return 0;
    }
    std::list<QueryFilter> filters;
};


} // namespace hnswlib

#endif // _HNSWPF_QUERY_H_
/* vim: set ts=4 sw=4 sts=4 tw=100 */




