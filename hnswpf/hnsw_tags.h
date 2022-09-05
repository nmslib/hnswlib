#ifndef _HNSWPF_TAG_H_
#define _HNSWPF_TAG_H_

#include <mutex>
#include <cstring>
#include <utility>
#include <vector>
#include <set>
#include <sstream>
#include <unordered_set>


namespace hnswlib {



class TagIndex {
public:
    static const int max_tag_locks = 8192;

    std::vector<std::mutex> tag_lock;

    // tag filed name
    std::string name;

    // tag -> docId set
    std::unordered_map<int, std::unordered_set<int>> tag_to_externals;

public:
    TagIndex() : name(""), tag_lock(max_tag_locks) {}

    std::unordered_set<int> getExternals(int item) const {
        std::unordered_set<int> externals;

        auto find_externals = tag_to_externals.find(item);
        if (find_externals != tag_to_externals.end())
            return find_externals->second;
        return externals;
    }

    void insert(int tag, int external_id) {
        std::unique_lock<std::mutex> tl(tag_lock[tag & (max_tag_locks - 1)]);
        const auto &iter = tag_to_externals.find(tag);
        // field tag exists
        if (iter != tag_to_externals.end()) {
            tag_to_externals[tag].insert(external_id);
        } else {
            tag_to_externals[tag] = {external_id};
        }
    }


    void serialize(std::ostream &out) {
        writeString(out, name);
        writeSetMap(out, tag_to_externals);
    }

    void deserialize(std::istream &in) {
        name = readString(in);
        tag_to_externals = readSetMap(in);
    }


    static void writeSetMap(std::ostream &out,
                            const std::unordered_map<int, std::unordered_set<int>> &map) {
        int item_size = (int)map.size();
        writeBinaryPOD(out, item_size);
        for (const auto &pair : map) {
            writeBinaryPOD(out, pair.first);

            int id_size = (int)pair.second.size();
            writeBinaryPOD(out, id_size);
            // std::unordered_set<int>
            for (int iter : pair.second) {
                writeBinaryPOD(out, iter);
            }
        }
    }

    static void writeIntMap(std::ostream &out, const std::unordered_map<int, int>& map) {
        int item_size = (int)map.size();
        writeBinaryPOD(out, item_size);
        for (const auto &pair : map) {
            writeBinaryPOD(out, pair.first);
            // int
            writeBinaryPOD(out, pair.second);
        }
    }

    static std::unordered_map<int, std::unordered_set<int>> readSetMap(std::istream &in) {
        std::unordered_map<int, std::unordered_set<int>> map;

        int item_size;
        readBinaryPOD(in, item_size);
        for (int i = 0; i < item_size; i++) {
            int element;
            readBinaryPOD(in, element);
            int externals_length;
            readBinaryPOD(in, externals_length);

            std::unordered_set<int> _externals;
            for (int j = 0; j < externals_length; j++) {
                int id;
                readBinaryPOD(in, id);
                _externals.emplace(id);
            }
            map.emplace(element, _externals);
        }
        return map;
    }

    static std::unordered_map<int, int> readIntMap(std::istream &in) {
        std::unordered_map<int, int> map;

        int item_size;
        readBinaryPOD(in, item_size);
        for (int i = 0; i < item_size; i++) {
            int tag;
            readBinaryPOD(in, tag);
            int val;
            readBinaryPOD(in, val);
            map.emplace(tag, val);
        }
        return map;
    }

    static void writeString(std::ostream &out, std::string podRef) {
        int size = podRef.size();
        out.write((char *) &size, sizeof(size));
        out.write(&podRef[0], size);
    }

    static std::string readString(std::istream &in) {
        std::string str;
        int size;
        in.read((char *) &size, sizeof(size));

        str.resize(size);
        in.read(&str[0], size);
        return str;
    }

};


} // namespace hnswlib

#endif  // _HNSWPF_TAG_H_
/* vim: set ts=4 sw=4 sts=4 tw=100 */



