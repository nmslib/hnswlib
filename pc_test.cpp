#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

#include "hnswlib/hnswlib.h"

using namespace std;
using namespace hnswlib;

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}

class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    float getElapsedTimeSec() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::seconds>(time_end - time_begin).count());
    }

    float getElapsedTimeMil() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }

};

struct CompareByFirst {
    constexpr bool operator()(std::pair<int, tableint> const &a,
                                std::pair<int, tableint> const &b) const noexcept {
        return a.first < b.first;
    }
};

void pctest() {
    HierarchicalNSW<int> *algorithm;
    string data_path = "/data/data.csv";
    char path_index[1024] = "/data/dictionary.bin";
    
    size_t vecdim = 128;
    size_t vecsize = 400000;
    size_t M = 16;
    int efConstruction = 40;

    LD2Space ld2space(vecdim);

    StopW stopw = StopW();
    if (exists_test(path_index)) {
        cout << "Loading index from " << path_index << ":\n";
        algorithm = new HierarchicalNSW<int>(&ld2space, path_index, false);
        cout << "Elapsed time: " << stopw.getElapsedTimeMicro() << endl;
    } else {
        // read data from file, and add to algorithm.
        cout << "Reading data from file, and add to NSW" << endl;
        algorithm = new HierarchicalNSW<int>(&ld2space, vecsize, M, efConstruction);
        labeltype label = 0;
        string line;
        ifstream infile(data_path);
        while (getline(infile, line)) {
            label ++;
            if (label % 1000 == 0) {
                cout << "Reading data: " << label << endl;
            }
            const char * info = line.c_str();
            algorithm->addPoint(info, label);
        }
        infile.close();

        cout << "Elapsed time(sec): " << stopw.getElapsedTimeSec() << endl;
        stopw.reset();

        // save to file
        algorithm->saveIndex(path_index);
        cout << "Save Index Elapsed time(millisecond): " << stopw.getElapsedTimeMil() << endl;
        stopw.reset();
    }

    CompareByFirst comp;

    string query;
    cout << "Please enter a word: ";
    while (cin >> query) {
        stopw.reset();
        std::vector<std::pair<int, labeltype>> result = algorithm->searchKnn(query.c_str(), 3, comp);
        cout << "Result size: " << result.size() << endl;
        cout << "Search Elapsed time(millisecond): " << stopw.getElapsedTimeMil() << endl;
        for (int i=0; i < result.size(); i++) {
            labeltype label = result[i].second;
            int dist = result[i].first;
            std::vector<char> queue = algorithm->template getDataByLabel<char>(label);
            cout << i << ": " << reinterpret_cast<char*>(queue.data()) << ", score: " << dist << endl;
        }
        cout << endl;
        cout << "Please enter a word: ";
    }
    
}
