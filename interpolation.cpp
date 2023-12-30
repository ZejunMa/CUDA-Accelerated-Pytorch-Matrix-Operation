#include<torch/extension.h>
#include<include/utils.h>

torch::Tensor trilinear_interpolation(torch::Tensor features, torch::Tensor points){
    return trilinear_fw_cu(features,points);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_interpolation", &trilinear_interpolation);
}  