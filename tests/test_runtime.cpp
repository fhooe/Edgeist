#include "edgeist/edgeist.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

#ifndef EDGEIST_PYTHON_EXECUTABLE
#define EDGEIST_PYTHON_EXECUTABLE "python3"
#endif

#ifndef EDGEIST_PYTHON_GOLDEN_MODEL
#define EDGEIST_PYTHON_GOLDEN_MODEL "tools/python_golden_model.py"
#endif

int g_failures = 0;

void fail(const char* expr, const char* file, int line, const std::string& detail = {})
{
    ++g_failures;
    std::cerr << file << ':' << line << " assertion failed: " << expr;
    if (!detail.empty()) {
        std::cerr << " (" << detail << ')';
    }
    std::cerr << '\n';
}

#define EXPECT_TRUE(expr) do { if (!(expr)) fail(#expr, __FILE__, __LINE__); } while (false)
#define EXPECT_EQ(a, b) do { auto _a = (a); auto _b = (b); if (!(_a == _b)) fail(#a " == " #b, __FILE__, __LINE__, std::to_string(_a) + " vs " + std::to_string(_b)); } while (false)
#define EXPECT_NEAR(a, b, tol) do { auto _a = static_cast<double>(a); auto _b = static_cast<double>(b); if (std::fabs(_a - _b) > static_cast<double>(tol)) fail(#a " ~= " #b, __FILE__, __LINE__, std::to_string(_a) + " vs " + std::to_string(_b)); } while (false)
#define RUN_TEST(fn) do { std::cout << "[ RUN      ] " #fn "\n"; fn(); std::cout << "[     DONE ] " #fn "\n"; } while (false)

template <typename T>
void append_struct(std::vector<std::byte>& bytes, const T& value)
{
    const auto* raw = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), raw, raw + sizeof(T));
}

void append_floats(std::vector<std::byte>& bytes, const std::vector<float>& values)
{
    const auto* raw = reinterpret_cast<const std::byte*>(values.data());
    bytes.insert(bytes.end(), raw, raw + values.size() * sizeof(float));
}

void write_u32(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value)
{
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

std::vector<float> floats_from_bytes(const std::vector<std::byte>& bytes)
{
    std::vector<float> out(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), out.size() * sizeof(float));
    return out;
}


template <typename T, std::size_t N>
auto bytes_of(const std::array<T, N>& arr) -> edgeist::ConstByteSpan
{
    return edgeist::ConstByteSpan(reinterpret_cast<const std::byte*>(arr.data()), arr.size() * sizeof(T));
}

template <typename T, std::size_t N>
auto mutable_bytes_of(std::array<T, N>& arr) -> edgeist::ByteSpan
{
    return edgeist::ByteSpan(reinterpret_cast<std::byte*>(arr.data()), arr.size() * sizeof(T));
}

auto bytes_of_vector(const std::vector<std::byte>& vec) -> edgeist::ConstByteSpan
{
    return edgeist::ConstByteSpan(vec.data(), vec.size());
}

auto mutable_bytes_of_vector(std::vector<std::byte>& vec) -> edgeist::ByteSpan
{
    return edgeist::ByteSpan(vec.data(), vec.size());
}

template <std::size_t N>
auto float_tensor(const std::array<float, N>& arr, float scale = 1.0F) -> edgeist::ConstTypedTensorView
{
    return edgeist::ConstTypedTensorView { bytes_of(arr), edgeist::NumericDataType::Float32, arr.size(), scale };
}

template <std::size_t N>
auto mutable_float_tensor(std::array<float, N>& arr, float scale = 1.0F) -> edgeist::MutableTypedTensorView
{
    return edgeist::MutableTypedTensorView { mutable_bytes_of(arr), edgeist::NumericDataType::Float32, arr.size(), scale };
}

auto encode_values(const std::vector<float>& values, edgeist::NumericDataType type, float scale) -> std::vector<std::byte>
{
    std::vector<std::byte> bytes(values.size() * edgeist::numeric_data_type_bytes(type));
    edgeist::MutableTypedTensorView tensor { edgeist::ByteSpan(bytes.data(), bytes.size()), type, values.size(), scale };
    const auto status = edgeist::encode_float_span(edgeist::ConstSpan<float>(values), tensor);
    if (!status.ok()) {
        fail("encode_float_span", __FILE__, __LINE__, std::string(edgeist::to_string(status.code)) + ": " + std::string(status.message));
    }
    return bytes;
}

auto typed_const_from_bytes(const std::vector<std::byte>& bytes, edgeist::NumericDataType type, std::size_t elements, float scale) -> edgeist::ConstTypedTensorView
{
    return edgeist::ConstTypedTensorView { bytes_of_vector(bytes), type, elements, scale };
}

auto typed_mut_from_bytes(std::vector<std::byte>& bytes, edgeist::NumericDataType type, std::size_t elements, float scale) -> edgeist::MutableTypedTensorView
{
    return edgeist::MutableTypedTensorView { mutable_bytes_of_vector(bytes), type, elements, scale };
}

using GoldenMap = std::map<std::string, std::vector<double>>;

std::vector<double> parse_values(const std::string& text)
{
    std::vector<double> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(std::stod(item));
        }
    }
    return out;
}

GoldenMap parse_golden_output(const std::string& output)
{
    GoldenMap map;
    std::stringstream ss(output);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty()) {
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        map.emplace(line.substr(0, eq), parse_values(line.substr(eq + 1U)));
    }
    return map;
}

GoldenMap load_all_python_golden()
{
    const std::string output_path = std::string("/tmp/edgeist_golden_all_") + std::to_string(std::rand()) + ".txt";
    const std::string command = std::string("timeout 60s \"") + EDGEIST_PYTHON_EXECUTABLE + "\" \"" + EDGEIST_PYTHON_GOLDEN_MODEL
        + "\" --case all > \"" + output_path + "\" 2>&1";
    const int rc = std::system(command.c_str());
    std::ifstream in(output_path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string output = buffer.str();
    std::remove(output_path.c_str());
    if (rc != 0) {
        fail("python golden model returned success", __FILE__, __LINE__, command + " output=" + output);
    }
    return parse_golden_output(output);
}

GoldenMap run_python_case(const std::string& case_name)
{
    static const GoldenMap all_cases = load_all_python_golden();
    GoldenMap selected;
    const std::string prefix = case_name + ".";
    for (const auto& item : all_cases) {
        if (item.first.rfind(prefix, 0) == 0) {
            selected.emplace(item.first.substr(prefix.size()), item.second);
        }
    }
    if (selected.empty()) {
        fail("python golden case exists", __FILE__, __LINE__, case_name);
    }
    return selected;
}

const std::vector<double>& require_key(const GoldenMap& map, const std::string& key)
{
    const auto it = map.find(key);
    if (it == map.end()) {
        fail("golden key exists", __FILE__, __LINE__, key);
        static const std::vector<double> empty;
        return empty;
    }
    return it->second;
}

template <typename T>
void expect_vector_near(edgeist::ConstSpan<T> actual, const std::vector<double>& expected, double tolerance, const std::string& name)
{
    EXPECT_EQ(actual.size(), expected.size());
    const auto n = std::min(actual.size(), expected.size());
    for (std::size_t i = 0; i < n; ++i) {
        const double a = static_cast<double>(actual[i]);
        const double e = expected[i];
        if (std::fabs(a - e) > tolerance) {
            fail("vector element near", __FILE__, __LINE__, name + "[" + std::to_string(i) + "] " + std::to_string(a) + " vs " + std::to_string(e));
        }
    }
}

void expect_status_ok(edgeist::Status status)
{
    if (!status.ok()) {
        fail("status.ok()", __FILE__, __LINE__, std::string(edgeist::to_string(status.code)) + ": " + std::string(status.message));
    }
}

std::vector<std::byte> make_linear_model(edgeist::DataEncodingId encoding = edgeist::DataEncodingId::Float32)
{
    using namespace edgeist;
    ModelHeaderPrefix header {};
    header.headerSize = static_cast<std::uint16_t>(sizeof(ModelHeaderPrefix) + sizeof(std::uint32_t));
    header.magicNumber = kNmcfMagic;
    header.version = { '0', '2', '.', '0', '0', '.', '0', '0' };
    header.layerNrs = 1;
    header.channelsIn = 1;
    header.dimensionInputX = 3;
    header.dimensionInputY = 1;
    header.channelsOut = 1;
    header.dimensionOutputX = 2;
    header.dimensionOutputY = 1;

    std::vector<std::byte> model;
    append_struct(model, header);
    const std::uint32_t layer_offset = header.headerSize;
    append_struct(model, layer_offset);

    LinearHeader linear {};
    linear.id = static_cast<std::uint8_t>(LayerId::Linear);
    linear.layerNr = 0;
    linear.predecessorNr = 0;
    linear.dimensionInputX = 3;
    linear.dimensionOutputX = 2;
    const auto element_size = data_encoding_bytes(encoding);
    linear.dataEncoding = static_cast<std::uint8_t>(encoding);
    linear.weightsAmountFrozen = 0;
    linear.weightsAmountTrainable = 6;
    linear.biasAmountFrozen = 0;
    linear.biasAmountTrainable = 2;
    linear.offsets.weightsTrainableOffset = 0;
    linear.offsets.biasTrainableOffset = static_cast<std::uint32_t>(6 * element_size);
    append_struct(model, linear);
    return model;
}

std::vector<std::byte> make_linear_trainable(edgeist::NumericDataType type = edgeist::NumericDataType::Float32)
{
    const std::vector<float> values { 0.1F, -0.2F, 0.3F, 0.4F, 0.5F, -0.6F, 0.01F, -0.02F };
    if (type == edgeist::NumericDataType::Float32) {
        std::vector<std::byte> trainable;
        append_floats(trainable, values);
        return trainable;
    }
    return encode_values(values, type, 1.0F / 64.0F);
}

std::vector<std::byte> make_network_model()
{
    using namespace edgeist;
    constexpr std::uint16_t layer_count = 4;
    ModelHeaderPrefix header {};
    header.headerSize = static_cast<std::uint16_t>(sizeof(ModelHeaderPrefix) + layer_count * sizeof(std::uint32_t));
    header.magicNumber = kNmcfMagic;
    header.version = { '0', '2', '.', '0', '0', '.', '0', '0' };
    header.layerNrs = layer_count;
    header.channelsIn = 1;
    header.dimensionInputX = 2;
    header.dimensionInputY = 1;
    header.channelsOut = 1;
    header.dimensionOutputX = 2;
    header.dimensionOutputY = 1;

    std::vector<std::byte> model;
    append_struct(model, header);
    const std::size_t offset_table = model.size();
    model.resize(model.size() + layer_count * sizeof(std::uint32_t));

    std::array<std::uint32_t, layer_count> offsets {};
    offsets[0] = static_cast<std::uint32_t>(model.size());
    LinearHeader l0 {};
    l0.id = static_cast<std::uint8_t>(LayerId::Linear);
    l0.layerNr = 0;
    l0.dimensionInputX = 2;
    l0.dimensionOutputX = 3;
    l0.dataEncoding = static_cast<std::uint8_t>(DataEncodingId::Float32);
    l0.weightsAmountTrainable = 6;
    l0.biasAmountTrainable = 3;
    l0.offsets.weightsTrainableOffset = 0;
    l0.offsets.biasTrainableOffset = 6 * sizeof(float);
    append_struct(model, l0);

    offsets[1] = static_cast<std::uint32_t>(model.size());
    UnaryHeader relu {};
    relu.id = static_cast<std::uint8_t>(LayerId::ReLU);
    relu.layerNr = 1;
    relu.dimensionInputX = 3;
    relu.dimensionOutputX = 3;
    append_struct(model, relu);

    offsets[2] = static_cast<std::uint32_t>(model.size());
    LinearHeader l2 {};
    l2.id = static_cast<std::uint8_t>(LayerId::Linear);
    l2.layerNr = 2;
    l2.dimensionInputX = 3;
    l2.dimensionOutputX = 2;
    l2.dataEncoding = static_cast<std::uint8_t>(DataEncodingId::Float32);
    l2.weightsAmountTrainable = 6;
    l2.biasAmountTrainable = 2;
    l2.offsets.weightsTrainableOffset = (6 + 3) * sizeof(float);
    l2.offsets.biasTrainableOffset = (6 + 3 + 6) * sizeof(float);
    append_struct(model, l2);

    offsets[3] = static_cast<std::uint32_t>(model.size());
    UnaryHeader softmax {};
    softmax.id = static_cast<std::uint8_t>(LayerId::Softmax);
    softmax.layerNr = 3;
    softmax.dimensionInputX = 2;
    softmax.dimensionOutputX = 2;
    append_struct(model, softmax);

    for (std::size_t i = 0; i < offsets.size(); ++i) {
        write_u32(model, offset_table + i * sizeof(std::uint32_t), offsets[i]);
    }
    return model;
}

std::vector<std::byte> make_network_trainable()
{
    std::vector<std::byte> trainable;
    append_floats(trainable, {
        0.2F, -0.1F, -0.3F, 0.4F, 0.1F, 0.2F, 0.0F, 0.1F, -0.1F,
        0.3F, -0.2F, 0.1F, -0.4F, 0.2F, 0.5F, 0.0F, 0.0F,
    });
    return trainable;
}

void test_softmax_cross_entropy_python_golden()
{
    const auto golden = run_python_case("softmax");
    std::array<float, 3> logits { 1.0F, 2.0F, 3.0F };
    std::array<float, 3> output {};
    expect_status_ok(edgeist::layers::SoftmaxLayer::forward(float_tensor(logits), mutable_float_tensor(output)));
    expect_vector_near(edgeist::ConstSpan<float>(output), require_key(golden, "probabilities"), 1.0e-6, "softmax probabilities");

    std::array<float, 3> target { 0.0F, 0.0F, 1.0F };
    float loss = 0.0F;
    expect_status_ok(edgeist::layers::CrossEntropyLoss::loss(edgeist::ConstSpan<float>(output), edgeist::ConstSpan<float>(target), loss));
    EXPECT_NEAR(loss, require_key(golden, "loss").front(), 1.0e-6F);
    std::array<float, 3> grad {};
    expect_status_ok(edgeist::layers::CrossEntropyLoss::gradient(edgeist::ConstSpan<float>(output), edgeist::ConstSpan<float>(target), edgeist::Span<float>(grad)));
    expect_vector_near(edgeist::ConstSpan<float>(grad), require_key(golden, "gradient"), 1.0e-6, "cross entropy gradient");
}

void test_linear_backward_optimizer_python_golden()
{
    const auto golden = run_python_case("linear");
    std::array<float, 3> input { 1.0F, 2.0F, 3.0F };
    std::array<float, 6> weights { 0.1F, -0.2F, 0.3F, 0.4F, 0.5F, -0.6F };
    std::array<float, 2> bias { 0.01F, -0.02F };
    std::array<float, 2> output {};
    expect_status_ok(edgeist::layers::LinearLayer::forward(float_tensor(input), float_tensor(weights), float_tensor(bias),
        mutable_float_tensor(output), 3, 2));
    expect_vector_near(edgeist::ConstSpan<float>(output), require_key(golden, "output"), 1.0e-6, "linear output");

    std::array<float, 2> grad_out { 0.5F, -1.0F };
    std::array<float, 3> grad_input {};
    std::array<float, 6> grad_weights {};
    std::array<float, 2> grad_bias {};
    expect_status_ok(edgeist::layers::LinearLayer::backward(float_tensor(input), edgeist::ConstSpan<float>(grad_out),
        float_tensor(weights), edgeist::Span<float>(grad_input), edgeist::Span<float>(grad_weights),
        edgeist::Span<float>(grad_bias), 3, 2, false));
    expect_vector_near(edgeist::ConstSpan<float>(grad_input), require_key(golden, "grad_input"), 1.0e-6, "linear grad input");
    expect_vector_near(edgeist::ConstSpan<float>(grad_weights), require_key(golden, "grad_weights"), 1.0e-6, "linear grad weights");
    expect_vector_near(edgeist::ConstSpan<float>(grad_bias), require_key(golden, "grad_bias"), 1.0e-6, "linear grad bias");

    std::array<float, 2> params { 1.0F, 2.0F };
    std::array<float, 2> grads { 0.2F, -0.3F };
    expect_status_ok(edgeist::apply_sgd(edgeist::Span<float>(params), edgeist::ConstSpan<float>(grads), 0.1F));
    expect_vector_near(edgeist::ConstSpan<float>(params), require_key(golden, "sgd_params"), 1.0e-6, "sgd params");

    std::array<float, 1> adam_param { 1.0F };
    std::array<float, 1> adam_grad { 0.2F };
    std::array<float, 1> m { 0.0F };
    std::array<float, 1> v { 0.0F };
    expect_status_ok(edgeist::apply_adam(edgeist::Span<float>(adam_param), edgeist::ConstSpan<float>(adam_grad), edgeist::Span<float>(m), edgeist::Span<float>(v), 0.1F, 0.9F, 0.999F, 1.0e-8F, 1));
    expect_vector_near(edgeist::ConstSpan<float>(adam_param), require_key(golden, "adam_param"), 1.0e-5, "adam param");
}

void test_conv2d_python_golden()
{
    const auto golden = run_python_case("conv2d");
    std::array<float, 9> input { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F };
    std::array<float, 4> weights { 1.0F, 0.0F, 0.0F, -1.0F };
    std::array<float, 1> bias { 0.5F };
    std::array<float, 4> output {};
    edgeist::layers::Conv2DParams params { 1, 1, 3, 3, 2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1 };
    expect_status_ok(edgeist::layers::Conv2DLayer::forward(float_tensor(input), float_tensor(weights), float_tensor(bias),
        mutable_float_tensor(output), params));
    expect_vector_near(edgeist::ConstSpan<float>(output), require_key(golden, "output"), 1.0e-6, "conv output");

    std::array<float, 4> grad_out { 1.0F, 1.0F, 1.0F, 1.0F };
    std::array<float, 9> grad_input {};
    std::array<float, 4> grad_weights {};
    std::array<float, 1> grad_bias {};
    expect_status_ok(edgeist::layers::Conv2DLayer::backward(float_tensor(input), edgeist::ConstSpan<float>(grad_out),
        float_tensor(weights), edgeist::Span<float>(grad_input), edgeist::Span<float>(grad_weights),
        edgeist::Span<float>(grad_bias), params, false));
    expect_vector_near(edgeist::ConstSpan<float>(grad_input), require_key(golden, "grad_input"), 1.0e-6, "conv grad input");
    expect_vector_near(edgeist::ConstSpan<float>(grad_weights), require_key(golden, "grad_weights"), 1.0e-6, "conv grad weights");
    expect_vector_near(edgeist::ConstSpan<float>(grad_bias), require_key(golden, "grad_bias"), 1.0e-6, "conv grad bias");
}

void test_relu_pooling_batchnorm_python_golden()
{
    const auto golden = run_python_case("relu_pool_batchnorm");
    std::array<float, 4> relu_input { -1.0F, 0.0F, 2.0F, -3.0F };
    std::array<float, 4> relu_output {};
    expect_status_ok(edgeist::layers::ReLULayer::forward(float_tensor(relu_input), mutable_float_tensor(relu_output)));
    expect_vector_near(edgeist::ConstSpan<float>(relu_output), require_key(golden, "relu_output"), 1.0e-6, "relu output");
    std::array<float, 4> relu_grad_out { 1.0F, 1.0F, 1.0F, 1.0F };
    std::array<float, 4> relu_grad_in {};
    expect_status_ok(edgeist::layers::ReLULayer::backward(float_tensor(relu_input), edgeist::ConstSpan<float>(relu_grad_out), edgeist::Span<float>(relu_grad_in)));
    expect_vector_near(edgeist::ConstSpan<float>(relu_grad_in), require_key(golden, "relu_grad"), 1.0e-6, "relu gradient");

    std::array<float, 9> pool_input { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F };
    std::array<float, 4> pool_output {};
    std::array<std::uint32_t, 4> argmax {};
    edgeist::layers::Pool2DParams pool { 1, 3, 3, 2, 2, 2, 2, 0, 0, 1, 1 };
    expect_status_ok(edgeist::layers::MaxPool2DLayer::forward(float_tensor(pool_input), mutable_float_tensor(pool_output), edgeist::Span<std::uint32_t>(argmax), pool));
    expect_vector_near(edgeist::ConstSpan<float>(pool_output), require_key(golden, "pool_output"), 1.0e-6, "pool output");
    std::array<float, 9> pool_grad_in {};
    std::array<float, 4> pool_grad_out { 1.0F, 2.0F, 3.0F, 4.0F };
    expect_status_ok(edgeist::layers::MaxPool2DLayer::backward(edgeist::ConstSpan<float>(pool_grad_out), edgeist::ConstSpan<std::uint32_t>(argmax), edgeist::Span<float>(pool_grad_in), pool));
    expect_vector_near(edgeist::ConstSpan<float>(pool_grad_in), require_key(golden, "pool_grad_input"), 1.0e-6, "pool grad input");

    std::array<float, 4> bn_input { 1.0F, 2.0F, 3.0F, 4.0F };
    std::array<float, 2> bn_gamma { 2.0F, 3.0F };
    std::array<float, 2> bn_beta { 0.5F, -0.5F };
    std::array<float, 2> bn_mean { 1.5F, 3.5F };
    std::array<float, 2> bn_var { 0.25F, 0.25F };
    std::array<float, 4> bn_output {};
    auto gamma_t = float_tensor(bn_gamma);
    auto beta_t = float_tensor(bn_beta);
    auto mean_t = float_tensor(bn_mean);
    auto var_t = float_tensor(bn_var);
    expect_status_ok(edgeist::layers::BatchNormLayer::forward(float_tensor(bn_input), gamma_t, beta_t, mean_t, var_t, mutable_float_tensor(bn_output), 1.0e-5F));
    expect_vector_near(edgeist::ConstSpan<float>(bn_output), require_key(golden, "batchnorm_output"), 1.0e-5, "batchnorm output");
    std::array<float, 4> bn_grad_out { 1.0F, 1.0F, 1.0F, 1.0F };
    std::array<float, 4> bn_grad_in {};
    std::array<float, 2> bn_grad_gamma {};
    std::array<float, 2> bn_grad_beta {};
    expect_status_ok(edgeist::layers::BatchNormLayer::backward_affine(float_tensor(bn_input), edgeist::ConstSpan<float>(bn_grad_out), gamma_t, mean_t, var_t,
        edgeist::Span<float>(bn_grad_in), edgeist::Span<float>(bn_grad_gamma), edgeist::Span<float>(bn_grad_beta), 1.0e-5F, false));
    expect_vector_near(edgeist::ConstSpan<float>(bn_grad_in), require_key(golden, "batchnorm_grad_input"), 1.0e-4, "batchnorm grad input");
    expect_vector_near(edgeist::ConstSpan<float>(bn_grad_gamma), require_key(golden, "batchnorm_grad_gamma"), 1.0e-5, "batchnorm grad gamma");
    expect_vector_near(edgeist::ConstSpan<float>(bn_grad_beta), require_key(golden, "batchnorm_grad_beta"), 1.0e-6, "batchnorm grad beta");
}

void test_precision_selection_python_golden()
{
    const auto golden = run_python_case("precision");
    edgeist::QuantizationConfig q {};
    const std::vector<float> input_values { 1.0F, 2.0F, 3.0F };
    const std::vector<float> weight_values { 0.1F, -0.2F, 0.3F, 0.4F, 0.5F, -0.6F };
    const std::vector<float> bias_values { 0.01F, -0.02F };

    auto fp16_input = encode_values(input_values, edgeist::NumericDataType::Float16, q.activation_scale);
    auto fp16_weights = encode_values(weight_values, edgeist::NumericDataType::Float16, q.weight_scale);
    auto fp16_bias = encode_values(bias_values, edgeist::NumericDataType::Float16, q.bias_scale);
    std::vector<std::byte> fp16_output_bytes(2 * edgeist::numeric_data_type_bytes(edgeist::NumericDataType::Float16));
    expect_status_ok(edgeist::layers::LinearLayer::forward(
        typed_const_from_bytes(fp16_input, edgeist::NumericDataType::Float16, 3, q.activation_scale),
        typed_const_from_bytes(fp16_weights, edgeist::NumericDataType::Float16, 6, q.weight_scale),
        typed_const_from_bytes(fp16_bias, edgeist::NumericDataType::Float16, 2, q.bias_scale),
        typed_mut_from_bytes(fp16_output_bytes, edgeist::NumericDataType::Float16, 2, q.activation_scale), 3, 2));
    std::array<float, 2> fp16_output {};
    expect_status_ok(edgeist::decode_to_float_span(typed_const_from_bytes(fp16_output_bytes, edgeist::NumericDataType::Float16, 2, q.activation_scale), edgeist::Span<float>(fp16_output)));
    expect_vector_near(edgeist::ConstSpan<float>(fp16_output), require_key(golden, "fp16_linear_output"), 1.0e-4, "fp16 linear output");

    auto int8_input = encode_values(input_values, edgeist::NumericDataType::Int8, q.activation_scale);
    auto int8_weights = encode_values(weight_values, edgeist::NumericDataType::Int8, q.weight_scale);
    auto int8_bias = encode_values(bias_values, edgeist::NumericDataType::Int8, q.bias_scale);
    std::vector<std::byte> int8_output_bytes(2 * edgeist::numeric_data_type_bytes(edgeist::NumericDataType::Int8));
    expect_status_ok(edgeist::layers::LinearLayer::forward(
        typed_const_from_bytes(int8_input, edgeist::NumericDataType::Int8, 3, q.activation_scale),
        typed_const_from_bytes(int8_weights, edgeist::NumericDataType::Int8, 6, q.weight_scale),
        typed_const_from_bytes(int8_bias, edgeist::NumericDataType::Int8, 2, q.bias_scale),
        typed_mut_from_bytes(int8_output_bytes, edgeist::NumericDataType::Int8, 2, q.activation_scale), 3, 2));
    std::array<float, 2> int8_output {};
    expect_status_ok(edgeist::decode_to_float_span(typed_const_from_bytes(int8_output_bytes, edgeist::NumericDataType::Int8, 2, q.activation_scale), edgeist::Span<float>(int8_output)));
    expect_vector_near(edgeist::ConstSpan<float>(int8_output), require_key(golden, "int8_linear_output"), 1.0e-6, "int8 linear output");

    edgeist::NumericDataType parsed {};
    expect_status_ok(edgeist::parse_numeric_data_type("int8", parsed));
    EXPECT_TRUE(parsed == edgeist::NumericDataType::Int8);
    EXPECT_EQ(edgeist::numeric_data_type_bytes(edgeist::NumericDataType::Float16), 2U);
    EXPECT_EQ(edgeist::numeric_data_type_bytes(edgeist::NumericDataType::Int8), 1U);
}


void test_runtime_uses_exported_parameter_dtype_python_golden()
{
    const auto golden = run_python_case("precision");
    std::array<float, 3> input { 1.0F, 2.0F, 3.0F };

    {
        auto model = make_linear_model(edgeist::DataEncodingId::Int8);
        auto trainable = make_linear_trainable(edgeist::NumericDataType::Int8);
        edgeist::ModelInfo info;
        expect_status_ok(edgeist::validate_model(edgeist::ConstByteSpan(model), edgeist::ConstByteSpan(trainable), &info));
        EXPECT_EQ(info.trainable_bytes, 8U);

        std::array<std::byte, 4096> scratch {};
        edgeist::ScratchArena arena{edgeist::ByteSpan(scratch)};
        edgeist::SramStorage storage{edgeist::ByteSpan(trainable)};
        edgeist::ModelView view{edgeist::ConstByteSpan(model), edgeist::ConstByteSpan(trainable)};
        edgeist::TrainingConfig config;
        config.mode = edgeist::RuntimeMode::InferenceOnly;
        config.inference_data_type = edgeist::NumericDataType::Int8;
        edgeist::ModelRuntime runtime;
        expect_status_ok(runtime.init(view, storage, arena, config));

        std::array<float, 2> output {};
        expect_status_ok(runtime.inference(edgeist::ConstSpan<float>(input), edgeist::Span<float>(output)));
        expect_vector_near(edgeist::ConstSpan<float>(output), require_key(golden, "int8_linear_output"), 1.0e-6, "int8 exported runtime output");

        std::array<std::byte, 4096> mismatch_scratch {};
        edgeist::ScratchArena mismatch_arena{edgeist::ByteSpan(mismatch_scratch)};
        edgeist::ModelRuntime mismatch_runtime;
        edgeist::TrainingConfig mismatch_config;
        mismatch_config.inference_data_type = edgeist::NumericDataType::Float32;
        const auto mismatch_status = mismatch_runtime.init(view, storage, mismatch_arena, mismatch_config);
        EXPECT_TRUE(!mismatch_status.ok());
    }

    {
        auto model = make_linear_model(edgeist::DataEncodingId::Float16);
        auto trainable = make_linear_trainable(edgeist::NumericDataType::Float16);
        edgeist::ModelInfo info;
        expect_status_ok(edgeist::validate_model(edgeist::ConstByteSpan(model), edgeist::ConstByteSpan(trainable), &info));
        EXPECT_EQ(info.trainable_bytes, 16U);

        std::array<std::byte, 4096> scratch {};
        edgeist::ScratchArena arena{edgeist::ByteSpan(scratch)};
        edgeist::SramStorage storage{edgeist::ByteSpan(trainable)};
        edgeist::ModelView view{edgeist::ConstByteSpan(model), edgeist::ConstByteSpan(trainable)};
        edgeist::TrainingConfig config;
        config.mode = edgeist::RuntimeMode::InferenceOnly;
        config.inference_data_type = edgeist::NumericDataType::Float16;
        edgeist::ModelRuntime runtime;
        expect_status_ok(runtime.init(view, storage, arena, config));

        std::array<float, 2> output {};
        expect_status_ok(runtime.inference(edgeist::ConstSpan<float>(input), edgeist::Span<float>(output)));
        expect_vector_near(edgeist::ConstSpan<float>(output), require_key(golden, "fp16_linear_output"), 1.0e-4, "fp16 exported runtime output");
    }
}

void test_binary_validation_runtime_inference_and_trace_python_golden()
{
    const auto golden = run_python_case("linear");
    auto model = make_linear_model();
    auto trainable = make_linear_trainable();
    edgeist::ModelInfo info;
    expect_status_ok(edgeist::validate_model(edgeist::ConstByteSpan(model), edgeist::ConstByteSpan(trainable), &info));
    EXPECT_EQ(info.layers.size(), 1U);
    EXPECT_EQ(info.layers[0].weights_trainable, 6U);

    auto corrupted = model;
    corrupted[2] = std::byte { 0x00 };
    auto status = edgeist::validate_model(edgeist::ConstByteSpan(corrupted), edgeist::ConstByteSpan(trainable), nullptr);
    EXPECT_TRUE(!status.ok());

    auto truncated = std::vector<std::byte>(model.begin(), model.begin() + 8);
    status = edgeist::validate_model(edgeist::ConstByteSpan(truncated), edgeist::ConstByteSpan(trainable), nullptr);
    EXPECT_TRUE(!status.ok());

    std::array<std::byte, 4096> scratch {};
    edgeist::ScratchArena arena{edgeist::ByteSpan(scratch)};
    edgeist::SramStorage storage{edgeist::ByteSpan(trainable)};
    edgeist::ModelView view{edgeist::ConstByteSpan(model), edgeist::ConstByteSpan(trainable)};
    edgeist::TrainingConfig config;
    config.mode = edgeist::RuntimeMode::InferenceOnly;
    edgeist::ModelRuntime runtime;
    expect_status_ok(runtime.init(view, storage, arena, config));
    std::array<float, 3> input { 1.0F, 2.0F, 3.0F };
    std::array<float, 2> output {};
    std::array<float, 2> trace_values {};
    std::array<std::uint32_t, 2> trace_offsets {};
    edgeist::ActivationTraceBuffer trace { edgeist::Span<float>(trace_values), edgeist::Span<std::uint32_t>(trace_offsets) };
    expect_status_ok(runtime.inference_with_trace(edgeist::ConstSpan<float>(input), edgeist::Span<float>(output), trace));
    expect_vector_near(edgeist::ConstSpan<float>(output), require_key(golden, "output"), 1.0e-6, "runtime output");
    expect_vector_near(edgeist::ConstSpan<float>(trace_values), require_key(golden, "output"), 1.0e-6, "runtime trace layer0");
    EXPECT_EQ(trace.layer_count, 1U);
    EXPECT_EQ(trace.offsets[0], 0U);
    EXPECT_EQ(trace.offsets[1], 2U);
}

void test_network_activation_trace_python_golden()
{
    const auto golden = run_python_case("network_forward");
    auto model = make_network_model();
    auto trainable = make_network_trainable();
    std::array<std::byte, 8192> scratch {};
    edgeist::ScratchArena arena{edgeist::ByteSpan(scratch)};
    edgeist::SramStorage storage{edgeist::ByteSpan(trainable)};
    edgeist::ModelView view{edgeist::ConstByteSpan(model), edgeist::ConstByteSpan(trainable)};
    edgeist::TrainingConfig config;
    config.mode = edgeist::RuntimeMode::InferenceOnly;
    edgeist::ModelRuntime runtime;
    expect_status_ok(runtime.init(view, storage, arena, config));

    std::array<float, 2> input { 0.6F, -0.4F };
    std::array<float, 2> output {};
    std::array<float, 10> trace_values {};
    std::array<std::uint32_t, 5> trace_offsets {};
    edgeist::ActivationTraceBuffer trace { edgeist::Span<float>(trace_values), edgeist::Span<std::uint32_t>(trace_offsets) };
    expect_status_ok(runtime.inference_with_trace(edgeist::ConstSpan<float>(input), edgeist::Span<float>(output), trace));
    expect_vector_near(edgeist::ConstSpan<float>(output), require_key(golden, "output"), 1.0e-6, "network output");
    EXPECT_EQ(trace.layer_count, 4U);
    expect_vector_near(edgeist::ConstSpan<float>(trace_values.data() + trace_offsets[0], trace_offsets[1] - trace_offsets[0]), require_key(golden, "activation_0"), 1.0e-6, "activation 0");
    expect_vector_near(edgeist::ConstSpan<float>(trace_values.data() + trace_offsets[1], trace_offsets[2] - trace_offsets[1]), require_key(golden, "activation_1"), 1.0e-6, "activation 1");
    expect_vector_near(edgeist::ConstSpan<float>(trace_values.data() + trace_offsets[2], trace_offsets[3] - trace_offsets[2]), require_key(golden, "activation_2"), 1.0e-6, "activation 2");
    expect_vector_near(edgeist::ConstSpan<float>(trace_values.data() + trace_offsets[3], trace_offsets[4] - trace_offsets[3]), require_key(golden, "activation_3"), 1.0e-6, "activation 3");
}

void test_full_network_training_python_golden()
{
    const auto golden = run_python_case("full_training");
    auto model = make_network_model();
    auto trainable = make_network_trainable();
    std::array<std::byte, 16384> scratch {};
    edgeist::ScratchArena arena{edgeist::ByteSpan(scratch)};
    edgeist::SramStorage storage{edgeist::ByteSpan(trainable)};
    edgeist::ModelView view{edgeist::ConstByteSpan(model), edgeist::ConstByteSpan(trainable)};
    edgeist::TrainingConfig config;
    config.mode = edgeist::RuntimeMode::FullTraining;
    config.optimizer = edgeist::OptimizerKind::Sgd;
    config.learning_rate = 0.05F;
    config.training_data_type = edgeist::NumericDataType::Float32;
    edgeist::ModelRuntime runtime;
    expect_status_ok(runtime.init(view, storage, arena, config));

    std::array<float, 2> input { 0.6F, -0.4F };
    std::array<float, 2> target { 1.0F, 0.0F };
    std::array<float, 3> losses {};
    for (std::size_t step = 0; step < losses.size(); ++step) {
        expect_status_ok(runtime.begin_micro_batch());
        expect_status_ok(runtime.train_sample(edgeist::ConstSpan<float>(input), edgeist::ConstSpan<float>(target), &losses[step]));
        expect_status_ok(runtime.apply_updates(1));
    }
    expect_vector_near(edgeist::ConstSpan<float>(losses), require_key(golden, "losses"), 1.0e-6, "training losses");
    const auto trained = floats_from_bytes(trainable);
    expect_vector_near(edgeist::ConstSpan<float>(trained), require_key(golden, "final_trainable"), 1.0e-5, "final trainable params");

    std::array<float, 2> output {};
    expect_status_ok(runtime.inference(edgeist::ConstSpan<float>(input), edgeist::Span<float>(output)));
    expect_vector_near(edgeist::ConstSpan<float>(output), require_key(golden, "final_output"), 1.0e-6, "trained output");
}

void test_memory_planner_reports_strategy_and_dtype_changes()
{
    const auto golden = run_python_case("memory");
    auto model = make_linear_model();
    auto trainable = make_linear_trainable();
    edgeist::ModelInfo info;
    expect_status_ok(edgeist::validate_model(edgeist::ConstByteSpan(model), edgeist::ConstByteSpan(trainable), &info));
    edgeist::TrainingConfig inference;
    inference.mode = edgeist::RuntimeMode::InferenceOnly;
    inference.inference_data_type = edgeist::NumericDataType::Int8;
    edgeist::TrainingConfig training;
    training.mode = edgeist::RuntimeMode::FullTraining;
    training.training_data_type = edgeist::NumericDataType::Float16;
    edgeist::MemoryReport infer_report;
    edgeist::MemoryReport train_report;
    expect_status_ok(edgeist::MemoryPlanner::estimate(info, inference, infer_report));
    expect_status_ok(edgeist::MemoryPlanner::estimate(info, training, train_report));
    EXPECT_TRUE(train_report.gradient_bytes > infer_report.gradient_bytes);
    EXPECT_TRUE(train_report.total_peak_sram_bytes > infer_report.total_peak_sram_bytes);
    EXPECT_EQ(infer_report.inference_element_bytes, static_cast<std::size_t>(require_key(golden, "int8_bytes").front()));
    EXPECT_EQ(train_report.training_element_bytes, static_cast<std::size_t>(require_key(golden, "fp16_bytes").front()));
}

} // namespace

int main()
{
    try {
        RUN_TEST(test_softmax_cross_entropy_python_golden);
        RUN_TEST(test_linear_backward_optimizer_python_golden);
        RUN_TEST(test_conv2d_python_golden);
        RUN_TEST(test_relu_pooling_batchnorm_python_golden);
        RUN_TEST(test_precision_selection_python_golden);
        RUN_TEST(test_runtime_uses_exported_parameter_dtype_python_golden);
        RUN_TEST(test_binary_validation_runtime_inference_and_trace_python_golden);
        RUN_TEST(test_network_activation_trace_python_golden);
        RUN_TEST(test_full_network_training_python_golden);
        RUN_TEST(test_memory_planner_reports_strategy_and_dtype_changes);
    } catch (const std::exception& ex) {
        ++g_failures;
        std::cerr << "Unhandled exception: " << ex.what() << '\n';
    }

    if (g_failures != 0) {
        std::cerr << g_failures << " test assertion(s) failed\n";
        return 1;
    }
    std::cout << "All Edgeist runtime tests passed\n";
    return 0;
}
