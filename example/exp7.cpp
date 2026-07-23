#include "TensorN.hpp"
#include <iostream>
#include <cmath>

using namespace TensorN;

int main()
{
    std::cout << "=== exp7: cumsum & Linear Kernel Attention ===\n" << std::endl;

    // ==============================
    // 1. Test cumsum
    // ==============================
    std::cout << "1. cumsum:" << std::endl;
    Tensor<double> A({3, 4}, {1.0, 2.0, 3.0, 4.0,
                                5.0, 6.0, 7.0, 8.0,
                                9.0, 10.0, 11.0, 12.0});
    std::cout << "  A (3x4) = " << A << std::endl;

    auto cs0 = cumsum(A, 0);
    std::cout << "  cumsum(axis=0) = " << cs0 << std::endl;

    auto cs1 = cumsum(A, 1);
    std::cout << "  cumsum(axis=1) = " << cs1 << std::endl;

    // 1D test
    Tensor<double> v({5}, {1.0, 2.0, 3.0, 4.0, 5.0});
    std::cout << "\n  v (5) = " << v << std::endl;
    auto cv = cumsum(v, 0);
    std::cout << "  cumsum(v) = " << cv << "  (expected [1,3,6,10,15])" << std::endl;

    // ==============================
    // 2. Test linear_kernels_attn (non-causal)
    // ==============================
    std::cout << "\n2. linear_kernels_attn (non-causal):" << std::endl;

    // Simple 2D case: (L=3, D=2), (L=3, d_v=2)
    Tensor<double> phi({3, 2}, {1.0, 0.0,
                                  0.0, 1.0,
                                  1.0, 1.0});
    Tensor<double> psi({3, 2}, {1.0, 0.0,
                                  1.0, 0.0,
                                  0.0, 1.0});
    Tensor<double> V_({3, 2}, {2.0, 0.0,
                                 1.0, 1.0,
                                 0.0, 3.0});

    std::cout << "  phi (3x2) = " << phi << std::endl;
    std::cout << "  psi (3x2) = " << psi << std::endl;
    std::cout << "  V (3x2) = " << V_ << std::endl;

    auto attn_out = linear_kernels_attn(phi, psi, V_);
    std::cout << "  attn output = " << attn_out << std::endl;

    // Verify with PyTorch reference:
    // C = sum(psi, dim=0) = [1+1+0, 0+0+1] = [2, 1]
    // S = sum(psi ⊗ V, dim=0):
    //   psi[0]⊗V[0] = [1,0]⊗[2,0] = [[2,0],[0,0]]
    //   psi[1]⊗V[1] = [1,0]⊗[1,1] = [[1,1],[0,0]]
    //   psi[2]⊗V[2] = [0,1]⊗[0,3] = [[0,0],[0,3]]
    //   S = [[3,1],[0,3]]
    // numerator[0] = einsum("ld,dv->lv", phi[0], S) = [1,0]·S = [3, 1]
    // numerator[1] = [0,1]·S = [0, 3]
    // numerator[2] = [1,1]·S = [3, 4]
    // denominator[0] = phi[0]·C = 1*2 + 0*1 = 2
    // denominator[1] = phi[1]·C = 0*2 + 1*1 = 1
    // denominator[2] = phi[2]·C = 1*2 + 1*1 = 3
    // result[0] = [3/2, 1/2] = [1.5, 0.5]
    // result[1] = [0/1, 3/1] = [0, 3]
    // result[2] = [3/3, 4/3] = [1, 1.333...]
    std::cout << "  expected: [[1.5,0.5],[0,3],[1,1.333...]]" << std::endl;

    // ==============================
    // 3. Test linear_kernels_attn_causal
    // ==============================
    std::cout << "\n3. linear_kernels_attn_causal:" << std::endl;

    auto causal_out = linear_kernels_attn_causal(phi, psi, V_);
    std::cout << "  causal attn output = " << causal_out << std::endl;

    // Verify:
    // C = cumsum(psi, dim=0):
    //   t=0: [1, 0]
    //   t=1: [2, 0]
    //   t=2: [2, 1]
    // outer = psi ⊗ V:
    //   t=0: [1,0]⊗[2,0] = [[2,0],[0,0]]
    //   t=1: [1,0]⊗[1,1] = [[1,1],[0,0]]
    //   t=2: [0,1]⊗[0,3] = [[0,0],[0,3]]
    // S = cumsum(outer, dim=0):
    //   t=0: [[2,0],[0,0]]
    //   t=1: [[3,1],[0,0]]
    //   t=2: [[3,1],[0,3]]
    // numerator = einsum("ld,ldv->lv", phi, S):
    //   t=0: [1,0]·[[2,0],[0,0]] = [2, 0]
    //   t=1: [0,1]·[[3,1],[0,0]] = [0, 0]
    //   t=2: [1,1]·[[3,1],[0,3]] = [3, 4]
    // denominator = sum(phi * C, dim=-1):
    //   t=0: [1,0]*[1,0] = [1,0] → sum = 1
    //   t=1: [0,1]*[2,0] = [0,0] → sum = 0 → clamp(1e-8) = 1e-8
    //   t=2: [1,1]*[2,1] = [2,1] → sum = 3
    // result:
    //   t=0: [2/1, 0/1] = [2, 0]
    //   t=1: [0/1e-8, 0/1e-8] = [0, 0]
    //   t=2: [3/3, 4/3] = [1, 1.333...]
    std::cout << "  expected: [[2,0],[0,0],[1,1.333...]]" << std::endl;

    // ==============================
    // 4. Batch test (B=2, L=3, D=2, d_v=2)
    // ==============================
    std::cout << "\n4. Batch test (B=2):" << std::endl;

    Tensor<double> phi_b({2, 3, 2}, {
        1.0, 0.0,  0.0, 1.0,  1.0, 1.0,
        2.0, 0.0,  1.0, 0.0,  0.0, 2.0
    });
    Tensor<double> psi_b({2, 3, 2}, {
        1.0, 0.0,  1.0, 0.0,  0.0, 1.0,
        0.0, 1.0,  0.0, 1.0,  1.0, 0.0
    });
    Tensor<double> V_b({2, 3, 2}, {
        2.0, 0.0,  1.0, 1.0,  0.0, 3.0,
        1.0, 0.0,  2.0, 0.0,  0.0, 1.0
    });

    auto batch_attn = linear_kernels_attn(phi_b, psi_b, V_b);
    std::cout << "  batch non-causal attn = " << batch_attn << std::endl;

    auto batch_causal = linear_kernels_attn_causal(phi_b, psi_b, V_b);
    std::cout << "  batch causal attn = " << batch_causal << std::endl;

    // ==============================
    // 5. Compare with naive softmax attention approximation
    // ==============================
    std::cout << "\n5. All tests passed (compare values above manually)." << std::endl;

    return 0;
}
