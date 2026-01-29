#pragma once
#include <unordered_map>
#include <vector>
#include <string>

struct TrieNode;

struct TrieNode {
    std::unordered_map<char, TrieNode*> children;
    TrieNode* failure;
    std::vector<std::string> outputs;

    TrieNode() : failure(nullptr) {}
};