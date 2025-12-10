#include <iostream>
#include "ArcadiaEngine.h"
#include "ArcadiaEngine.cpp"
using namespace std;

int main() {
    ConcretePlayerTable table;

    table.insert(10, "Alice");
    table.insert(21, "Bob");

    cout << table.search(10) << endl; // Alice
    cout << table.search(21) << endl; // Bob
    cout << table.search(99) << endl; // empty string

    return 0;
}
