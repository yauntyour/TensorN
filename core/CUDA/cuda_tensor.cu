#include "cuda_tensor.hpp"

// Explicit instantiations for common types
template class TensorN::CudaTensor<float>;
template class TensorN::CudaTensor<double>;
template class TensorN::CudaTensor<int32_t>;
template class TensorN::CudaTensor<int64_t>;
template class TensorN::CudaTensor<TensorN::half>;
template class TensorN::CudaTensor<TensorN::bfloat16>;
template class TensorN::CudaTensor<TensorN::tf32>;
#if CUDART_VERSION >= 12000
template class TensorN::CudaTensor<TensorN::fp8_e4m3>;
template class TensorN::CudaTensor<TensorN::fp8_e5m2>;
#endif