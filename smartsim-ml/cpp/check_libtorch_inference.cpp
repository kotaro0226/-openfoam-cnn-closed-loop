#include <torch/script.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

std::vector<float> readFloatFile(const std::string& path)
{
    std::ifstream is(path, std::ios::binary);
    if (!is)
    {
        throw std::runtime_error("Could not open " + path);
    }
    is.seekg(0, std::ios::end);
    const std::streamoff bytes = is.tellg();
    is.seekg(0, std::ios::beg);
    if (bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
    {
        throw std::runtime_error("File size is not a multiple of float size: " + path);
    }
    std::vector<float> out(static_cast<std::size_t>(bytes / sizeof(float)));
    is.read(reinterpret_cast<char*>(out.data()), bytes);
    if (!is)
    {
        throw std::runtime_error("Failed reading " + path);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr
            << "Usage: " << argv[0]
            << " MODEL_TS INPUT_FLOAT32_BIN EXPECTED_FLOAT32_BIN\n";
        return 2;
    }

    const std::string modelPath = argv[1];
    const std::string inputPath = argv[2];
    const std::string expectedPath = argv[3];

    try
    {
        torch::NoGradGuard noGrad;
        torch::jit::script::Module model = torch::jit::load(modelPath, torch::kCPU);
        model.eval();

        std::vector<float> inputValues = readFloatFile(inputPath);
        std::vector<float> expectedValues = readFloatFile(expectedPath);

        const std::int64_t inputCount = 1LL * 18 * 20 * 20 * 20;
        const std::int64_t outputCount = 1LL * 6 * 16 * 16 * 16;
        if (static_cast<std::int64_t>(inputValues.size()) != inputCount)
        {
            throw std::runtime_error("Unexpected input float count");
        }
        if (static_cast<std::int64_t>(expectedValues.size()) != outputCount)
        {
            throw std::runtime_error("Unexpected expected-output float count");
        }

        torch::Tensor input = torch::from_blob(
            inputValues.data(),
            {1, 18, 20, 20, 20},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone();

        torch::Tensor expected = torch::from_blob(
            expectedValues.data(),
            {1, 6, 16, 16, 16},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone();

        torch::Tensor output = model.forward({input}).toTensor();
        torch::Tensor diff = (output - expected).abs();
        const double maxAbs = diff.max().item<double>();
        const double meanAbs = diff.mean().item<double>();

        std::cout << "output shape: " << output.sizes() << "\n";
        std::cout << "max_abs: " << maxAbs << "\n";
        std::cout << "mean_abs: " << meanAbs << "\n";

        return maxAbs <= 1e-5 ? 0 : 1;
    }
    catch (const c10::Error& e)
    {
        std::cerr << "LibTorch error: " << e.what() << "\n";
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
