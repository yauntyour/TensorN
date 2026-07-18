#include <stdfloat>
#include <iostream>
#include "Tensor.hpp"

int main(int argc, char const *argv[])
{
    TensorN::Tensor<std::float16_t> a({2, 5, 6});

    std::cout << a << std::endl;
    return 0;
}
