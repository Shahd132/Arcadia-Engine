// ArcadiaEngine.cpp - STUDENT TEMPLATE
// TODO: Implement all the functions below according to the assignment requirements

#include "ArcadiaEngine.h"
#include <algorithm>
#include <queue>
#include <numeric>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <set>

using namespace std;

// =========================================================
// PART A: DATA STRUCTURES (Concrete Implementations)
// =========================================================

// --- 1. PlayerTable (Double Hashing) ---

class ConcretePlayerTable : public PlayerTable {
private:
    // TODO: Define your data structures here
    // Hint: You'll need a hash table with double hashing collision resolution
    struct Entry
    {
        int key;    //player id
        string value;   //player name
        bool isOccupied;

        Entry(): key(0), value(""),isOccupied(false) {}
    };
    vector<Entry> hashTable ;
    int tableSize;
    int hash1(int key ,int tableSize){ // h(k) = floor( n( kA mod 1 ) )
        const double A=0.618033;
        //floor(tableSize*((key*A) % 1));
        double fractionPart=key * A - floor(key * A);
        return floor(tableSize*fractionPart);
    }
    //hash2(key) = PRIME – (key %PRIME) where PRIME is a prime smaller than the TABLE_SIZE.
    int hash2(int key){
        const int prime=97;
        return (prime - key%prime);
    }
    
public:
    ConcretePlayerTable() {
        // TODO: Initialize your hash table
        tableSize =101 ;//prim
        hashTable.resize(tableSize);
    }

    void insert(int playerID, string name) override {
        // TODO: Implement double hashing insert
        // Remember to handle collisions using h1(key) + i * h2(key)

        int index1=hash1(playerID,tableSize);
        int index2=hash2(playerID);

        for(int i=0;i<tableSize; i++){
            int position=(index1+i*index2)%tableSize;
            if(! hashTable[position].isOccupied){
                hashTable[position].key=playerID;
                hashTable[position].value=name;
                hashTable[position].isOccupied=true;
                return; //inserted
            }
        }
        // Table is full
        throw runtime_error("Hash table is full");
    }

    string search(int playerID) override {
        // TODO: Implement double hashing search
        // Return "" if player not found
        int index1=hash1(playerID,tableSize);
        int index2=hash2(playerID);

        for(int i=0 ; i<tableSize;i++){
            int position=(index1+i*index2)%tableSize;

            if(! hashTable[position].isOccupied){
                return "";
            }
            if(hashTable[position].key==playerID){
                return hashTable[position].value;
            }
        }
        return ""; //not found after looping the entire table
    }
};


// --- 2. Leaderboard (Skip List) ---

class ConcreteLeaderboard : public Leaderboard {
private:
    // TODO: Define your skip list node structure and necessary variables
    // Hint: You'll need nodes with multiple forward pointers

public:
    ConcreteLeaderboard() {
        // TODO: Initialize your skip list
    }

    void addScore(int playerID, int score) override {
        // TODO: Implement skip list insertion
        // Remember to maintain descending order by score
    }

    void removePlayer(int playerID) override {
        // TODO: Implement skip list deletion
    }

    vector<int> getTopN(int n) override {
        // TODO: Return top N player IDs in descending score order
        return {};
    }
};

// --- 3. AuctionTree (Red-Black Tree) ---
class ConcreteAuctionTree : public AuctionTree {
private:
    enum Color { Red, Black };

    struct Node {
        int id;
        int price;
        Color color;
        Node* left;
        Node* right;
        Node* parent;

        Node(int id, int price, Node* NIL) {
            this->id = id;
            this->price = price;
            color = Red;
            left = NIL;
            right = NIL;
            parent = NIL;
        }
    };

    Node* root;
    Node* NIL;

    void rotateLeft(Node* node) {
        Node* x = node->right;
        node->right = x->left;
        if (x->left != NIL)
            x->left->parent = node;
        x->parent = node->parent;
        if (node->parent == NIL)
            root = x;
        else if (node == node->parent->left)
            node->parent->left = x;
        else
            node->parent->right = x;
        x->left = node;
        node->parent = x;
    }

    void rotateRight(Node* node) {
        Node* x = node->left;
        node->left = x->right;
        if (x->right != NIL)
            x->right->parent = node;
        x->parent = node->parent;
        if (node->parent == NIL)
            root = x;
        else if (node == node->parent->right)
            node->parent->right = x;
        else
            node->parent->left = x;
        x->right = node;
        node->parent = x;
    }

    void fixInsertion(Node* node) {
        while (node != root && node->parent->color == Red) {
            Node* parent = node->parent;
            Node* grandParent = parent->parent;
            if (parent == grandParent->left) {
                Node* uncle = grandParent->right;
                if (uncle->color == Red) { // Case 1
                    parent->color = Black;
                    uncle->color = Black;
                    grandParent->color = Red;
                    node = grandParent;
                }
                else {
                    if (node == parent->right) { // Case 2
                        node = parent;
                        rotateLeft(node);
                        parent = node->parent;
                        grandParent = parent->parent;
                    }
                    parent->color = Black; // Case 3
                    grandParent->color = Red;
                    rotateRight(grandParent);
                }
            }
            else {
                Node* uncle = grandParent->left;
                if (uncle->color == Red) { // Case 4
                    parent->color = Black;
                    uncle->color = Black;
                    grandParent->color = Red;
                    node = grandParent;
                }
                else {
                    if (node == parent->left) { // Case 5
                        node = parent;
                        rotateRight(node);
                        parent = node->parent;
                        grandParent = parent->parent;
                    }
                    parent->color = Black; // Case 6
                    grandParent->color = Red;
                    rotateLeft(grandParent);
                }
            }
        }
        root->color = Black;
    }

    void replaceNode(Node* oldNode, Node* newNode) {
        if (oldNode->parent == NIL)
            root = newNode;
        else if (oldNode == oldNode->parent->left)
            oldNode->parent->left = newNode;
        else
            oldNode->parent->right = newNode;
        newNode->parent = oldNode->parent;
    }

    Node* minimum(Node* node) {
        while (node->left != NIL)
            node = node->left;
        return node;
    }

    void fixDeletion(Node* node) 
    {
        while (node != root && node->color == Black) 
        {
            if (node == node->parent->left) 
            {
                Node* sibling = node->parent->right;
                if (sibling->color == Red) 
                //case4
                {
                    sibling->color = Black;
                    node->parent->color = Red;
                    rotateLeft(node->parent);
                    sibling = node->parent->right;
                }
                if (sibling->left->color == Black && sibling->right->color == Black)//case3
                {
                    sibling->color = Red;
                    node = node->parent;
                }
                else 
                {
                    if (sibling->right->color == Black)//case5
                    { 
                        sibling->left->color = Black;
                        sibling->color = Red;
                        rotateRight(sibling);
                        sibling = node->parent->right;
                    }
                    sibling->color = node->parent->color;//case6
                    node->parent->color = Black;
                    sibling->right->color = Black;
                    rotateLeft(node->parent);
                    break; 
                }
            }
            else 
            {
                Node* sibling = node->parent->left;
                if (sibling->color == Red)//case4
                {
                    sibling->color = Black;
                    node->parent->color = Red;
                    rotateRight(node->parent);
                    sibling = node->parent->left;
                }
                if (sibling->left->color == Black && sibling->right->color == Black)//case3
                {
                    sibling->color = Red;
                    node = node->parent;
                }
                else 
                {
                    if (sibling->left->color == Black)//case5
                    {
                        sibling->right->color = Black;
                        sibling->color = Red;
                        rotateLeft(sibling);
                        sibling = node->parent->left;
                    }
                    sibling->color = node->parent->color;//case6
                    node->parent->color = Black;
                    sibling->left->color = Black;
                    rotateRight(node->parent);
                    break; 
                }
            }
        }
        node->color = Black;
    }

public:
    ConcreteAuctionTree() {
        NIL = new Node(-1, -1, nullptr);
        NIL->color = Black;
        NIL->left = NIL->right = NIL->parent = NIL;
        root = NIL;
    }

    void insertItem(int itemID, int price) override {
        Node* newNode = new Node(itemID, price, NIL);
        Node* parent = NIL;
        Node* current = root;
        while (current != NIL) 
        {
            parent = current;
            if (itemID < current->id)
                current = current->left;
            else
                current = current->right;
        }
        newNode->parent = parent;
        if (parent == NIL)
            root = newNode;
        else if (itemID < parent->id)
            parent->left = newNode;
        else
            parent->right = newNode;
        fixInsertion(newNode);
    }

    void deleteItem(int itemID) override 
    {
        Node* z = root;
        while (z != NIL && z->id != itemID) 
        {
            if (itemID < z->id)
                z = z->left;
            else
                z = z->right;
        }
        if (z == NIL) return;

        Node* successor = z;
        Color originalColor = successor->color;
        Node* x;

        if (z->left == NIL) 
        {
            x = z->right;
            replaceNode(z, z->right);
        }
        else if (z->right == NIL) 
        {
            x = z->left;
            replaceNode(z, z->left);
        }
        else 
        {
            successor = minimum(z->right);
            originalColor = successor->color;
            x = successor->right;
            if (successor->parent == z)
                x->parent = successor;
            else 
            {
                replaceNode(successor, successor->right);
                successor->right = z->right;
                successor->right->parent = successor;
            }
            replaceNode(z, successor);
            successor->left = z->left;
            successor->left->parent = successor;
            successor->color = z->color;
        }
        if (originalColor == Black)
            fixDeletion(x);
        delete z;
    }
};


// =========================================================
// PART B: INVENTORY SYSTEM (Dynamic Programming)
// =========================================================

int InventorySystem::optimizeLootSplit(int n, vector<int>& coins) {
    // TODO: Implement partition problem using DP
    // Goal: Minimize |sum(subset1) - sum(subset2)|
    // Hint: Use subset sum DP to find closest sum to total/2
    return 0;
}

int InventorySystem::maximizeCarryValue(int capacity, vector<pair<int, int>>& items) {
    // TODO: Implement 0/1 Knapsack using DP
    // items = {weight, value} pairs
    // Return maximum value achievable within capacity
    return 0;
}

long long InventorySystem::countStringPossibilities(string s) {
    // TODO: Implement string decoding DP
    // Rules: "uu" can be decoded as "w" or "uu"
    //        "nn" can be decoded as "m" or "nn"
    // Count total possible decodings
    return 0;
}

// =========================================================
// PART C: WORLD NAVIGATOR (Graphs)
// =========================================================


bool WorldNavigator::pathExists(int n, vector<vector<int>>& edges, int source, int dest) {
    // TODO: Implement path existence check using BFS or DFS
    // edges are bidirectional

    //case 1 src=dest
    if(source==dest){
        return true;
    }

    //create the adjacency list(list of city neighbors)
    vector<vector<int>> adj(n);
    for (const auto& edge : edges){
        int from =edge[0];
        int to =edge[1];
        adj[from].push_back(to);
        adj[to].push_back(from);
    }

    // BFS
    vector<bool> visited(n, false);
    queue<int> q ;
    // Start from source city
    q.push(source);
    visited[source] = true;       
    
    while (!q.empty()) {
        int current =q.front();
        q.pop();
        // Check if we found destination
        if (current == dest){
            return true;
        }
        //get not visited neighbors
        for (int neighbor : adj[current]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
    return false;
}


long long WorldNavigator::minBribeCost(int n, int m, long long goldRate, long long silverRate,
    vector<vector<int>>& roadData) {
    // TODO: Implement Minimum Spanning Tree (Kruskal's or Prim's)
    // roadData[i] = {u, v, goldCost, silverCost}
    // Total cost = goldCost * goldRate + silverCost * silverRate
    // Return -1 if graph cannot be fully connected
    return -1;
}

string WorldNavigator::sumMinDistancesBinary(int n, vector<vector<int>>& roads) {
    // TODO: Implement All-Pairs Shortest Path (Floyd-Warshall)
    // Sum all shortest distances between unique pairs (i < j)
    // Return the sum as a binary string
    // Hint: Handle large numbers carefully
    return "0";
}

// =========================================================
// PART D: SERVER KERNEL (Greedy)
// =========================================================

int ServerKernel::minIntervals(vector<char>& tasks, int n) {
    // TODO: Implement task scheduler with cooling time
    // Same task must wait 'n' intervals before running again
    // Return minimum total intervals needed (including idle time)
    // Hint: Use greedy approach with frequency counting
    return 0;
}

// =========================================================
// FACTORY FUNCTIONS (Required for Testing)
// =========================================================

extern "C" {
    PlayerTable* createPlayerTable() {
        return new ConcretePlayerTable();
    }

    Leaderboard* createLeaderboard() {
        return new ConcreteLeaderboard();
    }

    AuctionTree* createAuctionTree() {
        return new ConcreteAuctionTree();
    }
}
