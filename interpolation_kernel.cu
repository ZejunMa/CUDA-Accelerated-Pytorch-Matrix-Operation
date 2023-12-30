#include<torch/extension.h>

torch::Tensor trilinear_fw_cu(torch::Tensor features, torch::Tensor points){
    return features;
}