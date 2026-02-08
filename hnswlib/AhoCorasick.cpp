#include "AhoCorasick.h"
#include <fstream>
#include <iostream>

AhoCorasick::AhoCorasick() {
    root = new TrieNode();
}

AhoCorasick::~AhoCorasick() {
    deleteTrie(root);
}

void AhoCorasick::deleteTrie(TrieNode* node) {
    if (!node) return;
    for (auto& kv : node->children) {
        deleteTrie(kv.second);
    }
    delete node;
}

EntityId AhoCorasick::addEntity(const std::string& word) {
    built = false;
    TrieNode* node = root;

    for (char c : word) {
        if (node->children.find(c) == node->children.end()) {
            node->children[c] = new TrieNode();
        }
        node = node->children[c];
    }

    // BRUTAL FIX: Handle duplicates. If is_end is true, this word already has an ID.
    if (node->is_end) {
        EntityId existingId = node->outputs[0]; 
        frequency[existingId]++; // Increment Document Frequency (for IDF)
        return existingId;
    }

    // New Entity Logic
    EntityId id = entities.size();
    node->is_end = true;
    node->outputs.push_back(id);
    
    entities.push_back(word);
    frequency.push_back(1); // First time this entity is mapped to a node

    return id;
}

void AhoCorasick::build() {
    if (built) return;
    buildFailureLinks();
    built = true;
}



void AhoCorasick::buildFailureLinks() {
    std::queue<TrieNode*> q;
    
    // Level 1 nodes point failure to root
    for (auto& kv : root->children) {
        kv.second->failure = root;
        q.push(kv.second);
    }

    while (!q.empty()) {
        TrieNode* current = q.front();
        q.pop();

        for (auto& kv : current->children) {
            char c = kv.first;
            TrieNode* child = kv.second;

            TrieNode* fail = current->failure;
            while (fail != root && fail->children.find(c) == fail->children.end()) {
                fail = fail->failure;
            }

            if (fail->children.find(c) != fail->children.end()) {
                child->failure = fail->children[c];
            } else {
                child->failure = root;
            }

            // Propagate outputs from failure links (Crucial for overlapping matches)
            child->outputs.insert(child->outputs.end(),
                                  child->failure->outputs.begin(),
                                  child->failure->outputs.end());

            q.push(child);
        }
    }
}

std::vector<std::pair<size_t, EntityId>> AhoCorasick::search(const std::string& text) {
    if (!built) build();
    
    std::vector<std::pair<size_t, EntityId>> matches;
    TrieNode* node = root;

    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];

        while (node != root && node->children.find(c) == node->children.end()) {
            node = node->failure;
        }

        if (node->children.find(c) != node->children.end()) {
            node = node->children[c];
        }

        for (EntityId id : node->outputs) {
            size_t start = i - entities[id].size() + 1;
            matches.push_back({start, id});
            // REMOVED: frequency[id]++ (Don't mix Query Pop with Doc Freq)
        }
    }

    // 1. Keep only the longest match per start position
    std::unordered_map<size_t, std::pair<size_t, EntityId>> best_at_start;
    for (auto& m : matches) {
        if (best_at_start.find(m.first) == best_at_start.end() ||
            entities[m.second].size() > best_at_start[m.first].first) {
            best_at_start[m.first] = {entities[m.second].size(), m.second};
        }
    }

    std::vector<std::pair<size_t, EntityId>> final_matches;
    for (auto& kv : best_at_start) {
        final_matches.push_back({kv.first, kv.second.second});
    }

    // 2. Sort by Frequency (ASCENDING) to prioritize RARE entities for HNSW seeds
    // Rarer entities = Stronger signal = Better Adaptive Steering
    std::sort(final_matches.begin(), final_matches.end(),
              [&](const auto& a, const auto& b) {
                  return frequency[a.second] < frequency[b.second];
              });

    return final_matches;
}

void AhoCorasick::save(std::ostream& out) const {
    size_t num_entities = entities.size();
    // Use the HNSW style: writeBinaryPOD or simple write
    out.write((char*)&num_entities, sizeof(num_entities));

    for (size_t i = 0; i < num_entities; ++i) {
        size_t len = entities[i].size();
        out.write((char*)&len, sizeof(len));
        out.write(entities[i].c_str(), len);
        
        // SAVE THE FREQUENCY - Critical for your "Rarity-based Steering" paper!
        out.write((char*)&frequency[i], sizeof(frequency[i]));
    }
}

void AhoCorasick::load(std::istream& in) {
    // 1. Clear current state (memory safety)
    deleteTrie(root);
    root = new TrieNode();
    entities.clear();
    frequency.clear();

    size_t num_entities;
    in.read((char*)&num_entities, sizeof(num_entities));

    for (size_t i = 0; i < num_entities; ++i) {
        size_t len;
        in.read((char*)&len, sizeof(len));
        std::string word(len, '\0');
        in.read(&word[0], len);
        
        size_t freq;
        in.read((char*)&freq, sizeof(freq));

        // 2. Re-insert manually into Trie structure
        TrieNode* node = root;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                node->children[c] = new TrieNode();
            }
            node = node->children[c];
        }
        node->is_end = true;
        // The current index in the 'entities' vector becomes the EntityId
        node->outputs.push_back(entities.size());
        
        // 3. Restore data structures
        entities.push_back(word);
        frequency.push_back(freq);
    }
    
    // 4. Critical: Re-run BFS to restore failure/dictionary links
    build(); 
}

const std::string& AhoCorasick::getEntity(EntityId id) const { return entities[id]; }
size_t AhoCorasick::getFrequency(EntityId id) const { return frequency[id]; }