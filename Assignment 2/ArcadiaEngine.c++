// ArcadiaEngine.cpp - STUDENT TEMPLATE
// TODO: Implement all the functions below according to the assignment requirements
//final

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
#include <functional>


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

        Entry() : key(0), value(""), isOccupied(false) {}
    };
    vector<Entry> hashTable;
    int tableSize;
    int hash1(int key, int tableSize) { // h(k) = floor( n( kA mod 1 ) )
        const double A = 0.618033;
        //floor(tableSize*((key*A) % 1));
        double fractionPart = key * A - floor(key * A);
        return floor(tableSize * fractionPart);
    }
    //hash2(key) = PRIME – (key %PRIME) where PRIME is a prime smaller than the TABLE_SIZE.
    int hash2(int key) {
        const int prime = 97;
        return (prime - key % prime);
    }

public:
    ConcretePlayerTable() {
        // TODO: Initialize your hash table
        tableSize = 101;//prim
        hashTable.resize(tableSize);
    }

    void insert(int playerID, string name) override {
        // TODO: Implement double hashing insert
        // Remember to handle collisions using h1(key) + i * h2(key)

        int index1 = hash1(playerID, tableSize);
        int index2 = hash2(playerID);

        for (int i = 0; i < tableSize; i++) {
            int position = (index1 + i * index2) % tableSize;
            if (!hashTable[position].isOccupied) {
                hashTable[position].key = playerID;
                hashTable[position].value = name;
                hashTable[position].isOccupied = true;
                return; //inserted
            }
        }
        // Table is full
        cout << "Table is Full";
    }

    string search(int playerID) override {
        // TODO: Implement double hashing search
        // Return "" if player not found
        int index1 = hash1(playerID, tableSize);
        int index2 = hash2(playerID);

        for (int i = 0; i < tableSize; i++) {
            int position = (index1 + i * index2) % tableSize;

            if (hashTable[position].isOccupied && hashTable[position].key == playerID) {
                return hashTable[position].value;
            }
        }
        return ""; //not found after looping the entire table
    }
};

// --- 2. Leaderboard (Skip List) ---

class ConcreteLeaderboard : public Leaderboard {
private:
    struct Node {
        int playerID;
        int score;
        vector<Node*> forward;

        Node(int id, int s, int level)
            : playerID(id), score(s), forward(level, nullptr) {
        }
    };

    static const int MAX_LEVEL = 16;
    int level;
    Node* head;

    int randomLevel() {
        int lvl = 1;
        while ((rand() % 2) && lvl < MAX_LEVEL)
            lvl++;
        return lvl;
    }

    // Comparison: higher score first, tie by smaller ID
    bool comesBefore(int score1, int id1, int score2, int id2) {
        if (score1 != score2)
            return score1 > score2;
        return id1 < id2;
    }

public:
    ConcreteLeaderboard() {
        level = 1;
        head = new Node(-1, INT_MAX, MAX_LEVEL);
    }

    void addScore(int playerID, int score) override {
        vector<Node*> update(MAX_LEVEL, nullptr);
        Node* curr = head;

        // Find insertion position
        for (int i = level - 1; i >= 0; i--) {
            while (curr->forward[i] &&
                comesBefore(curr->forward[i]->score,
                    curr->forward[i]->playerID,
                    score, playerID)) {
                curr = curr->forward[i];
            }
            update[i] = curr;
        }

        curr = curr->forward[0];

        // If player exists → remove old score first
        if (curr && curr->playerID == playerID) {
            removePlayer(playerID);
        }

        int newLevel = randomLevel();
        if (newLevel > level) {
            for (int i = level; i < newLevel; i++)
                update[i] = head;
            level = newLevel;
        }

        Node* newNode = new Node(playerID, score, newLevel);
        for (int i = 0; i < newLevel; i++) {
            newNode->forward[i] = update[i]->forward[i];
            update[i]->forward[i] = newNode;
        }
    }

    /*void removePlayer(int playerID) override {
        vector<Node*> update(MAX_LEVEL, nullptr);
        Node* curr = head;

        for (int i = level - 1; i >= 0; i--) {
            while (curr->forward[i] &&
                curr->forward[i]->playerID < playerID) {
                curr = curr->forward[i];
            }
            update[i] = curr;
        }

        curr = curr->forward[0];

        if (curr && curr->playerID == playerID) {
            for (int i = 0; i < level; i++) {
                if (update[i]->forward[i] != curr)
                    break;
                update[i]->forward[i] = curr->forward[i];
            }
            delete curr;

            while (level > 1 && head->forward[level - 1] == nullptr)
                level--;
        }
    }*/
    void removePlayer(int playerID) override {
        vector<Node*> update(MAX_LEVEL, nullptr);
        Node* curr = head;

        for (int i = level - 1; i >= 0; i--) {
            while (curr->forward[i] &&
                comesBefore(curr->forward[i]->score,
                    curr->forward[i]->playerID,
                    INT_MAX, playerID)) {
                curr = curr->forward[i];
            }
            update[i] = curr;
        }

        curr = curr->forward[0];

        if (curr && curr->playerID == playerID) {
            for (int i = 0; i < level; i++) {
                if (update[i]->forward[i] != curr)
                    break;
                update[i]->forward[i] = curr->forward[i];
            }
            delete curr;

            while (level > 1 && head->forward[level - 1] == nullptr)
                level--;
        }
    }



    vector<int> getTopN(int n) override {
        vector<int> result;
        Node* curr = head->forward[0];

        while (curr && n--) {
            result.push_back(curr->playerID);
            curr = curr->forward[0];
        }
        return result;
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
    int totalSum = accumulate(coins.begin(), coins.end(), 0);
    int target = totalSum / 2;

    vector<bool> dp(target + 1, false);
    dp[0] = true;

    for (int coin : coins) {
        for (int s = target; s >= coin; s--) {
            dp[s] = dp[s] || dp[s - coin];
        }
    }

    for (int s = target; s >= 0; s--) {
        if (dp[s]) return totalSum - 2 * s;
    }

    return totalSum;
}

int InventorySystem::maximizeCarryValue(int capacity, vector<pair<int, int>>& items) {
    // TODO: Implement 0/1 Knapsack using DP
    // items = {weight, value} pairs
    // Return maximum value achievable within capacity
    vector<int> dp(capacity + 1, 0);

    for (auto& item : items) {
        int w = item.first;
        int v = item.second;
        for (int c = capacity; c >= w; c--) {
            dp[c] = max(dp[c], dp[c - w] + v);
        }
    }

    return dp[capacity];
}

long long InventorySystem::countStringPossibilities(string s) {
    // TODO: Implement string decoding DP
    // Rules: "uu" can be decoded as "w" or "uu"
    //        "nn" can be decoded as "m" or "nn"
    // Count total possible decodings
    const long long MOD = 1e9 + 7;
    int n = s.size();
    vector<long long> dp(n + 1, 0);
    dp[0] = 1;

    for (int i = 1; i <= n; i++) {
        dp[i] = dp[i - 1];

        if (i >= 2) {
            string pair = s.substr(i - 2, 2);
            if (pair == "uu" || pair == "nn") {
                dp[i] = (dp[i] + dp[i - 2]) % MOD;
            }
        }
    }

    return dp[n];
}

// =========================================================
// PART C: WORLD NAVIGATOR (Graphs)
// =========================================================


bool WorldNavigator::pathExists(int n, vector<vector<int>>& edges, int source, int dest) {
    // TODO: Implement path existence check using BFS or DFS
    // edges are bidirectional

    //case 1 src=dest
    if (source == dest) {
        return true;
    }

    //create the adjacency list(list of city neighbors)
    vector<vector<int>> adj(n);
    for (const auto& edge : edges) {
        int from = edge[0];
        int to = edge[1];
        adj[from].push_back(to);
        adj[to].push_back(from);
    }

    // BFS
    vector<bool> visited(n, false);
    queue<int> q;
    // Start from source city
    q.push(source);
    visited[source] = true;

    while (!q.empty()) {
        int current = q.front();
        q.pop();
        // Check if we found destination
        if (current == dest) {
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
    vector<int> parent(n);
    for (int i = 0; i < n; i++) parent[i] = i;

    function<int(int)> find = [&](int x) {
        if (parent[x] == x)
            return x;
        return find(parent[x]);
        };

    auto unite = [&](int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        parent[b] = a;
        return true;
        };

    // Compute cost of each road
    vector<pair<long long, pair<int, int>>> edges;

    for (auto& r : roadData) {
        int u = r[0];
        int v = r[1];
        long long gold = r[2];
        long long silver = r[3];
        long long cost = gold * goldRate + silver * silverRate;
        edges.push_back({ cost, {u, v} });
    }

    sort(edges.begin(), edges.end());

    long long totalCost = 0;
    int connectedEdges = 0;

    for (auto& e : edges) {
        if (unite(e.second.first, e.second.second)) {
            totalCost += e.first;
            connectedEdges++;
            if (connectedEdges == n - 1) break;
        }
    }

    if (connectedEdges != n - 1) return -1;
    return totalCost;

}

//string WorldNavigator::sumMinDistancesBinary(int n, vector<vector<int>>& roads) {
//    // TODO: Implement All-Pairs Shortest Path (Floyd-Warshall)
//    // Sum all shortest distances between unique pairs (i < j)
//    // Return the sum as a binary string
//    // Hint: Handle large numbers carefully
//    const long long INF = 1e15;
//
//    // Step 1: Initialize distance matrix
//    vector<vector<long long>> dist(n, vector<long long>(n, INF));
//
//    for (int i = 0; i < n; i++)
//        dist[i][i] = 0;
//
//    // roads[i] = {u, v, w}
//    for (auto& r : roads) {
//        int u = r[0], v = r[1];
//        long long w = r[2];
//
//        dist[u][v] = min(dist[u][v], w);
//        dist[v][u] = min(dist[v][u], w); // undirected
//    }
//
//    // Step 2: Floyd–Warshall
//    for (int k = 0; k < n; k++) {
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < n; j++) {
//                if (dist[i][k] + dist[k][j] < dist[i][j]) {
//                    dist[i][j] = dist[i][k] + dist[k][j];
//                }
//            }
//        }
//    }
//
//    // Step 3: Sum all distances
//    long long total = 0;
//
//    for (int i = 0; i < n; i++) {
//        for (int j = 0; j < n; j++) {
//            if (dist[i][j] < INF)
//                total += dist[i][j];
//        }
//    }
//
//    // Step 4: Convert to binary
//    string bin = "";
//    while (total > 0) {
//        bin.push_back((total % 2) + '0');
//        total /= 2;
//    }
//
//    if (bin.empty())
//        return "0";
//
//    reverse(bin.begin(), bin.end());
//    return bin;
//    //return "0";
//}

string WorldNavigator::sumMinDistancesBinary(int n, vector<vector<int>>& roads) {
    // TODO: Implement All-Pairs Shortest Path (Floyd-Warshall)
    // Sum all shortest distances between unique pairs (i < j)
    // Return the sum as a binary string
    // Hint: Handle large numbers carefully
    const long long INF = 1e15;

    vector<vector<long long>> dist(n, vector<long long>(n, INF));
    for (int i = 0; i < n; i++)
        dist[i][i] = 0;

    for (auto& r : roads) {
        int u = r[0], v = r[1];
        long long w = r[2];
        dist[u][v] = min(dist[u][v], w);
        dist[v][u] = min(dist[v][u], w);
    }

    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    long long total = 0;

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            total += dist[i][j];
        }
    }

    string bin = "";
    while (total > 0) {
        bin.push_back((total % 2) + '0');
        total /= 2;
    }

    if (bin.empty())
        return "0";

    reverse(bin.begin(), bin.end());
    return bin;
}



// =========================================================
// PART D: SERVER KERNEL (Greedy)
// =========================================================

int ServerKernel::minIntervals(vector<char>& tasks, int n) {
    // TODO: Implement task scheduler with cooling time
    // Same task must wait 'n' intervals before running again
    // Return minimum total intervals needed (including idle time)
    // Hint: Use greedy approach with frequency counting
    if (n == 0)
        return tasks.size();

    vector<int> freq(26, 0);
    for (char t : tasks) {
        freq[t - 'A']++;
    }

    int maxFreq = 0;
    for (int f : freq) {
        maxFreq = max(maxFreq, f);
    }

    int countMax = 0;
    for (int f : freq) {
        if (f == maxFreq)
            countMax++;
    }

    int intervals = (maxFreq - 1) * (n + 1) + countMax;

    return max((int)tasks.size(), intervals);
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
