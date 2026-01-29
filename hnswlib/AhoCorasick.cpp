#include "AhoCorasick.h"
#include <queue>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <algorithm> 

AhoCorasick::AhoCorasick() {
    root = new TrieNode();
}

AhoCorasick::~AhoCorasick() {
    deleteTrie(root);
}

void AhoCorasick::deleteTrie(TrieNode* node) {
    if (!node) return;
    for (auto& kv: node->children) {
        deleteTrie(kv.second);
    }
    delete node;
}

size_t countUniqueWordsInTrie(TrieNode* node, std::unordered_set<std::string>& seen) {
    if (!node) return 0;
    for (auto& word : node->outputs) {
        seen.insert(word);
    }
    for (auto& [c, child] : node->children) {
        countUniqueWordsInTrie(child, seen);
    }
    return seen.size();
}

size_t AhoCorasick::numWords() const {
    std::unordered_set<std::string> seen;
    return countUniqueWordsInTrie(root, seen);
}

void AhoCorasick::build(const std::vector<std::string>& entities) {
    for (const auto& word: entities) {
        TrieNode* node = root;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                node->children[c] = new TrieNode();
            }
            node = node->children[c];
        }
        node->outputs.push_back(word);
    }
    buildFailureLinks();
}

void AhoCorasick::buildFailureLinks() {
    std::queue<TrieNode*> q;
    root->failure = root;

    for (auto& kv : root->children) {
        kv.second ->failure = root;
        q.push(kv.second);
    }

    while (!q.empty()) {
        TrieNode* current = q.front(); q.pop();
        for (auto& kv : current->children) {
            char c = kv.first;
            TrieNode* child = kv.second;

            TrieNode* fail = current->failure;
            while (fail != root && fail -> children.find(c) == fail->children.end()) {
                fail = fail->failure;
            }

            if (fail->children.find(c) != fail->children.end()) {
                child->failure = fail->children[c];
            } else {
                child->failure = root;
            }

            child->outputs.insert(child -> outputs.end(),
                                  child->failure->outputs.begin(),
                                  child->failure->outputs.end());
            
            q.push(child);
        }
    }
}

std::vector<std::pair<size_t, std::string>> AhoCorasick::search(const std::string& text) const {
    std::vector<std::pair<size_t, std::string>> matches;
    TrieNode* node = root;

    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];
        while (node != root && node->children.find(c) == node->children.end())
            node = node->failure;

        if (node->children.find(c) != node->children.end())
            node = node->children.at(c);

        for (const auto& out : node->outputs)
            matches.push_back({i - out.size() + 1, out}); // start index
    }

    // sort matches by start index
    std::sort(matches.begin(), matches.end(),
              [](const auto &a, const auto &b){ return a.first < b.first; });

    std::vector<std::pair<size_t, std::string>> longest_matches;
std::unordered_map<size_t, std::pair<size_t,std::string>> best_at_start;

for (auto& m : matches) {
    size_t start = m.first;
    size_t len = m.second.size();
    // keep only the longest word for this start index
    if (best_at_start.find(start) == best_at_start.end() || len > best_at_start[start].first) {
        best_at_start[start] = {len, m.second};
    }
}

// now collect and sort by start index
for (auto& kv : best_at_start) {
    longest_matches.push_back({kv.first, kv.second.second});
}
std::sort(longest_matches.begin(), longest_matches.end(),
          [](const auto &a, const auto &b){ return a.first < b.first; });


    return longest_matches;
}



void AhoCorasick::save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    std::cout << "Trie saving not implemented yet\n";
}

void AhoCorasick::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    std::cout << "Trie loading not implemented yet\n";
}