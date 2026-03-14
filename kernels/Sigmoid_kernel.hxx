#ifndef SIGMOID_KERNEL_HXX
#define SIGMOID_KERNEL_HXX

#include <alpaka/alpaka.hpp>
#include <vector>
#include <cstddef>

struct SigmoidKernel
{
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const &acc,T const *input,T *output,std::size_t n) const
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];

        if (idx < n)
        {
            output[idx] = T(1) / (T(1) + alpaka::math::exp(acc, -input[idx]));
        }
    }
};

#endif // SIGMOID_KERNEL_HXX