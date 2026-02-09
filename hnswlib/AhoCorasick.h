#pragma once
#include <unordered_map>
#include <vector>
#include <string>
#include <queue>
#include <algorithm>

using EntityId = size_t;

struct TrieNode {
    std::unordered_map<char, TrieNode*> children;
    TrieNode* failure = nullptr;
    std::vector<EntityId> outputs;
    bool is_end = false;

    TrieNode() : failure(nullptr), is_end(false) {}
};

class AhoCorasick {
public:
    AhoCorasick();
    ~AhoCorasick();

    EntityId addEntity(const std::string& word);
    void build();
    std::vector<std::pair<size_t, EntityId>> search(const std::string& text);

    const std::string& getEntity(EntityId id) const;
    size_t getFrequency(EntityId id) const;
    size_t numWords() const { return entities.size(); }

    void save(std::ostream &out) const;
    void load(std::istream &in);

private:
    TrieNode* root;
    std::vector<std::string> entities; 
    std::vector<size_t> frequency;     
    bool built = false;

    void deleteTrie(TrieNode* node);
    void buildFailureLinks();
};