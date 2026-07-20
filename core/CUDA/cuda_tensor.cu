#include "cuda_tensor.hpp"

// Explicit instantiations for common types
template class TensorN::CudaTensor<float>;
template class TensorN::CudaTensor<double>;
template class TensorN::CudaTensor<int32_t>;
template class TensorN::CudaTensor<int64_t>;