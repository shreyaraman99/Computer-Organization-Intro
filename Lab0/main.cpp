#include <iostream>
#include <cassert>
using namespace std;

int ezThreeFourths(int x) {
    int temp = (x << 1) + x;
    int sign = temp >> 31;
    int temp2 = sign & 3;
    return (temp + temp2) >> 2;
}

int main() {
    assert(ezThreeFourths(11) == 8);
    assert(ezThreeFourths(-9) == -6);
    assert(ezThreeFourths(1073741824) == -268435456);
    cout << "All tests passed!" << endl;
}
