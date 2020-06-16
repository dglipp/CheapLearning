#include <iostream>
#include "../Utilities.h"
using namespace std;

int main()
{
    int * a = new int[12]{1,2,3,4,2,3,4,5,3,10,11,12};
    SoupVec<int> vec(12,a);
    SoupMat<int> mat(3,4,a);
    int nr = mat.nRow();
    for(int i=0;i<mat.nRow();i++) {
        for (int j=0; j<mat.nCol(); j++) cout << mat[i][j] << "\t";
        cout << "\n\n";
    }
    for(int i=0;i<vec.size(); i++) cout << vec[i] << "\t";
    cout << "\n\n";
}