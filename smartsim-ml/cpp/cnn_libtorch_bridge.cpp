#include <torch/script.h>
#include <torch/cuda.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>

namespace
{

std::mutex g_mutex;
std::string g_loaded_path;
std::string g_loaded_device;
torch::jit::script::Module g_model;
bool g_has_model = false;

void set_error(char* error_buf, const std::size_t error_buf_len, const std::string& message)
{
    if (error_buf == nullptr || error_buf_len == 0)
    {
        return;
    }

    const std::size_t n = std::min(error_buf_len - 1, message.size());
    std::memcpy(error_buf, message.data(), n);
    error_buf[n] = '\0';
}

torch::Device resolve_device()
{
    const char* device_env = std::getenv("ML_CNN_DEVICE");
    const std::string requested =
        device_env == nullptr ? std::string("cpu") : std::string(device_env);

    if ((requested == "cuda" || requested == "gpu" || requested == "auto")
        && torch::cuda::is_available())
    {
        return torch::Device(torch::kCUDA);
    }
    return torch::Device(torch::kCPU);
}

}  // namespace

extern "C" int ml_cnn_predict_blocks(
    const char* model_path,
    const float* input,
    float* output,
    std::int64_t n_blocks,
    char* error_buf,
    std::size_t error_buf_len
)
{
    try
    {
        if (model_path == nullptr || model_path[0] == '\0')
        {
            set_error(error_buf, error_buf_len, "model_path is empty");
            return 2;
        }
        if (input == nullptr || output == nullptr)
        {
            set_error(error_buf, error_buf_len, "input or output pointer is null");
            return 2;
        }
        if (n_blocks <= 0)
        {
            set_error(error_buf, error_buf_len, "n_blocks must be positive");
            return 2;
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        const std::string path(model_path);
        const torch::Device device = resolve_device();
        const std::string device_key = device.str();
        if (!g_has_model || g_loaded_path != path || g_loaded_device != device_key)
        {
            g_model = torch::jit::load(path, device);
            g_model.to(device);
            g_model.eval();
            g_loaded_path = path;
            g_loaded_device = device_key;
            g_has_model = true;
        }

        torch::NoGradGuard no_grad;
        torch::Tensor x = torch::from_blob(
            const_cast<float*>(input),
            {n_blocks, 18, 20, 20, 20},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone().to(device);

        torch::Tensor y = g_model.forward({x}).toTensor().to(torch::kCPU).contiguous();
        const auto expected = std::array<std::int64_t, 5>{n_blocks, 6, 16, 16, 16};
        if (y.dim() != static_cast<std::int64_t>(expected.size()))
        {
            set_error(error_buf, error_buf_len, "unexpected model output rank");
            return 3;
        }
        for (std::size_t dim = 0; dim < expected.size(); ++dim)
        {
            if (y.sizes()[dim] != expected[dim])
            {
                set_error(error_buf, error_buf_len, "unexpected model output shape");
                return 3;
            }
        }

        std::memcpy(output, y.data_ptr<float>(), sizeof(float) * n_blocks * 6 * 16 * 16 * 16);
        return 0;
    }
    catch (const c10::Error& e)
    {
        set_error(error_buf, error_buf_len, std::string("LibTorch error: ") + e.what());
        return 1;
    }
    catch (const std::exception& e)
    {
        set_error(error_buf, error_buf_len, e.what());
        return 1;
    }
}

extern "C" int ml_cnn_predict_block(
    const char* model_path,
    const float* input,
    float* output,
    char* error_buf,
    std::size_t error_buf_len
)
{
    return ml_cnn_predict_blocks(model_path, input, output, 1, error_buf, error_buf_len);
}
