#include <src/kernels/rmsnorm.h>
#include <tests/unittests/data_gen.h>
#include <src/utils/check.h>
#include <memory>
#include <src/weights/norm_weights.h>

void cpuRMSNorm(float* out, float* weight, float eps, int num_token, int hidden_size)
{
    for(int row = 0; row < num_token; row ++)
    {
        float sum = 0.0f;
        for(int col = 0; col < hidden_size; col ++)
        {
            sum += out[row * hidden_size + col] * out[row * hidden_size + col];
        }
        float factor = rsqrtf(sum / hidden_size + eps);
        for(int col = 0; col < hidden_size; col ++) out[row * hidden_size + col] *= factor;
    }
}

int main(int argc, char** argv)
{
    float eps = 1e-5;
    int num_token = atoi(argv[1]);
    int hidden_size = atoi(argv[2]);
    // generate test data
    int ele_size = num_token * hidden_size;

    GeneRdData generatorD(0, 1, 0.0f, 2.0f);
    std::vector<int> decoder_shape {num_token, hidden_size};
    std::vector<int> weight_shape {hidden_size};
    float* h_decoder;
    float* h_weight;
    float* d_decoder;
    float* d_weight;
    float* d_decoder_rsd;
    int decoder_size = generatorD.gen_real_data(decoder_shape, (float**)&h_decoder);
    int weight_size = generatorD.gen_real_data(weight_shape, (float**)&d_decoder);
    CHECK(cudaMalloc((void**)&d_decoder, decoder_size));
    CHECK(cudaMalloc((void**)&d_weight, weight_size));
    CHECK(cudaMalloc((void**)&d_decoder_rsd, decoder_size));

    // memory copy
    CHECK(cudaMemcpy(d_decoder, h_decoder, decoder_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice));

    // TensorWrapper Init
    TensorWrapper<float>* decoderrsd_tensor = new TensorWrapper<float>(Device::GPU, DataType::FP32, decoder_shape, d_decoder_rsd);
    TensorWrapper<float>* decoder_tensor = new TensorWrapper<float>(Device::GPU, DataType::FP32, decoder_shape, d_decoder);
    // TensorWrapper<float>* weight_tensor = new TensorWrapper<float>(Device::GPU, DataType::FP32, weight_shape, d_weight);
    LayerNormWeight<float>* weight_tensor = new LayerNormWeight<float>();
    weight_tensor -> gamma = d_weight;
    launchRMSNorm(decoder_tensor, decoderrsd_tensor, weight_tensor, eps);
}