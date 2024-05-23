#include <torch/extension.h>
#include <iostream>
#include <vector>
using namespace std;

void cpp_print(
    string input
)
{
    cout<< input << endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("print",&cpp_print,"cpp print");
}