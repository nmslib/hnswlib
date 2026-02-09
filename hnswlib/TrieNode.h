#pragma once
#include <unordered_map>
#include <vector>
#include <string>

using EntityId = size_t;

struct TrieNode {
    std::unordered_map<char, TrieNode*> children;
    TrieNode* failure;
    std::vector<EntityId> outputs;
    bool is_end = false;

    TrieNode() : failure(nullptr) {}
};