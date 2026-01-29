#include "AhoCorasick.h"
#include <iostream>
#include <vector>

int main() {
    // Example symbolic entities for Tesla Model Y retrieval
    std::vector<std::string> entities = {
        "Tesla", "Tesla Model Y", "electric", "autopilot", "red", "blue", 
        "interior", "battery", "long range", "performance", "range"
    };

    // Build the Trie
    AhoCorasick ac;
    ac.build(entities);

    // Example user query
    std::string query = "Show me a red Tesla Model Y with autopilot and long range battery";

    // Search for symbolic entities
    auto matches = ac.search(query);

    // Display found entities and positions
    std::cout << "Detected symbolic entities in query:\n";
    for (auto& m : matches) {
        std::cout << " - '" << m.second << "' at position " << m.first << "\n";
    }

    // Total number of entities in the Trie
    std::cout << "\nTotal symbolic entities in Trie: " << ac.numWords() << "\n";

    return 0;
}
