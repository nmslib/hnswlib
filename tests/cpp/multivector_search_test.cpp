#include <assert.h>
#include "../../hnswlib/hnswlib.h"

typedef unsigned int docidtype;
typedef float dist_t;

int main() {
    int dim = 16;               // Dimension of the elements
    int max_elements = 1000;    // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    int num_queries = 100;
    int num_docs = 10;          // Number of documents to search
    int ef_collection = 15;     // Number of candidate documents during the search
                                // Controlls the recall: higher ef leads to better accuracy, but slower search
    docidtype min_doc_id = 0;
    docidtype max_doc_id = 49;

    // Initing index
    hnswlib::MultiVectorL2Space<docidtype> space(dim);
    hnswlib::BruteforceSearch<dist_t>* alg_brute = new hnswlib::BruteforceSearch<dist_t>(&space, max_elements);
    hnswlib::HierarchicalNSW<dist_t>* alg_hnsw = new hnswlib::HierarchicalNSW<dist_t>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    std::uniform_int_distribution<docidtype> distrib_docid(min_doc_id, max_doc_id);

    size_t data_point_size = space.get_data_size();
    char* data = new char[data_point_size * max_elements];
    for (int i = 0; i < max_elements; i++) {
        // set vector value
        char* point_data = data + i * data_point_size;
        for (int j = 0; j < dim; j++) {
            char* vec_data = point_data + j * sizeof(float);
            float value = distrib_real(rng);
            *(float*)vec_data = value;
        }
        // set document id
        docidtype doc_id = distrib_docid(rng);
        space.set_doc_id(point_data, doc_id);
    }

    // Add data to index
    std::unordered_map<hnswlib::labeltype, docidtype> label_docid_lookup;
    for (int i = 0; i < max_elements; i++) {
        hnswlib::labeltype label = i;
        char* point_data = data + i * data_point_size;
        alg_hnsw->addPoint(point_data, label);
        alg_brute->addPoint(point_data, label);
        label_docid_lookup[label] = space.get_doc_id(point_data);
    }

    // Query random vectors and check overall recall
    float correct = 0;
    float total_num_elements = 0;
    size_t query_size = dim * sizeof(float);
    for (int i = 0; i < num_queries; i++) {
        char* query_data = new char[query_size];
        for (int j = 0; j < dim; j++) {
            size_t offset = j * sizeof(float);
            char* vec_data = query_data + offset;
            float value = distrib_real(rng);
            *(float*)vec_data = value;
        }
        hnswlib::MultiVectorSearchStopCondition<docidtype, dist_t> stop_condition(space, num_docs, ef_collection);
        std::vector<std::pair<dist_t, hnswlib::labeltype>> hnsw_results =
            alg_hnsw->searchStopConditionClosest(query_data, stop_condition);

        // check number of found documents
        std::unordered_set<docidtype> hnsw_docs;
        std::unordered_set<hnswlib::labeltype> hnsw_labels;
        for (auto pair: hnsw_results) {
            hnswlib::labeltype label = pair.second;
            hnsw_labels.emplace(label);
            docidtype doc_id = label_docid_lookup[label];
            hnsw_docs.emplace(doc_id);
        }
        assert(hnsw_docs.size() == num_docs);

        // Check overall recall
        std::vector<std::pair<dist_t, hnswlib::labeltype>> gt_results = 
            alg_brute->searchKnnCloserFirst(query_data, max_elements);
        std::unordered_set<docidtype> gt_docs;
        for (int i = 0; i < gt_results.size(); i++) {
            if (gt_docs.size() == num_docs) {
                break;
            }
            hnswlib::labeltype gt_label = gt_results[i].second;
            if (hnsw_labels.find(gt_label) != hnsw_labels.end()) {
                correct += 1;
            }
            docidtype gt_doc_id = label_docid_lookup[gt_label];
            gt_docs.emplace(gt_doc_id);
            total_num_elements += 1;
        }
        delete[] query_data;
    }
    float recall = correct / total_num_elements;
    std::cout << "random elements search recall : " << recall << "\n";
    assert(recall > 0.95);

    // Query the elements for themselves and measure recall
    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        hnswlib::MultiVectorSearchStopCondition<docidtype, dist_t> stop_condition(space, num_docs, ef_collection);
        std::vector<std::pair<float, hnswlib::labeltype>> result =
            alg_hnsw->searchStopConditionClosest(data + i * data_point_size, stop_condition);
        hnswlib::labeltype label = -1;
        if (!result.empty()) {
            label = result[0].second;
        }
        if (label == i) correct++;
    }
    recall = correct / max_elements;
    std::cout << "same elements search recall : " << recall << "\n";
    assert(recall > 0.99);

    delete[] data;
    delete alg_brute;
    delete alg_hnsw;
    return 0;
}
