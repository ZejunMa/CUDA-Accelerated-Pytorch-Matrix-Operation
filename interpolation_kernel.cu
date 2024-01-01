#include<torch/extension.h>

torch::Tensor trilinear_fw_cu(torch::Tensor features, torch::Tensor points){
    const int N = features.size(0), F = features.size(2);
    torch::Tensor interpolated_features = torch::zeros({N,F},  features.options());

    const dim3 threads(16, 16); // total 256 threads
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y - 1)/ threads.y);

    // launch kernel
    AT_DISPATCH_FLOATING_TYPES(features.type(), "trilinear_fw_cu", 
    ([&] {
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return interpolated_features;
}