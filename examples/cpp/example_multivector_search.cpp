#include "../../hnswlib/hnswlib.h"

typedef unsigned int docidtype;
typedef float dist_t;

int main() {
    int dim = 16;               // Dimension of the elements
    int max_elements = 10000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    int num_queries = 5;
    int num_docs = 5;           // Number of documents to search
    int ef_collection = 6;      // Number of candidate documents during the search
                                // Controlls the recall: higher ef leads to better accuracy, but slower search
    docidtype min_doc_id = 0;
    docidtype max_doc_id = 9;

    // Initing index
    hnswlib::MultiVectorL2Space<docidtype> space(dim);
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
        label_docid_lookup[label] = space.get_doc_id(point_data);
    }

    // Query random vectors
    size_t query_size = dim * sizeof(float);
    for (int i = 0; i < num_queries; i++) {
        char* query_data = new char[query_size];
        for (int j = 0; j < dim; j++) {
            size_t offset = j * sizeof(float);
            char* vec_data = query_data + offset;
            float value = distrib_real(rng);
            *(float*)vec_data = value;
        }
        std::cout << "Query #" << i << "\n";
        hnswlib::MultiVectorSearchStopCondition<docidtype, dist_t> stop_condition(space, num_docs, ef_collection);
        std::vector<std::pair<float, hnswlib::labeltype>> result = 
            alg_hnsw->searchStopConditionClosest(query_data, stop_condition);
        size_t num_vectors = result.size();

        std::unordered_map<docidtype, size_t> doc_counter;
        for (auto pair: result) {
            hnswlib::labeltype label = pair.second;
            docidtype doc_id = label_docid_lookup[label];
            doc_counter[doc_id] += 1;
        }
        std::cout << "Found " << doc_counter.size() << " documents, " << num_vectors << " vectors\n";
        delete[] query_data;
    }

    delete[] data;
    delete alg_hnsw;
    return 0;
}
