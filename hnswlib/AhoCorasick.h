#pragma once
#include "TrieNode.h"
#include <string>
#include <vector>

class AhoCorasick {

public:
    AhoCorasick();
    ~AhoCorasick();
    
    size_t numWords() const;

    void build(const std::vector<std::string> &entities);

    std::vector<std::pair<size_t, std::string>> search(const std::string& text) const;

    void save(const std::string& filename) const;
    void load(const std::string& filename);

private:
    TrieNode* root;
    void buildFailureLinks();
    void deleteTrie(TrieNode* node);
};