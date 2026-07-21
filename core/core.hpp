#pragma once
#ifndef __CORE__H__
#define __CORE__H__

#include "TensorN.hpp"
#include "einsum.hpp"
#include "operations.hpp"
#include "static.hpp"
#include "BLAS/blas_tensor.hpp"

#ifdef TENSORN_CUDA_AVAILABLE
#include "CUDA/cuda_tensor.hpp"
#include "CUDA/elementwise.hpp"
#include "CUDA/reduction.hpp"
#include "CUDA/matmul.hpp"
#include "CUDA/convolution.hpp"
#endif

#endif //!__CORE__H__
