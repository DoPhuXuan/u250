#include "bert_full_model_interfaces.h"
#include "bert_math.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

static const std::size_t TB_TOKEN_WORDS = SEQ_LEN * PACKS;
static const std::size_t ALL_ATTN_WEIGHT_WORDS = NUM_LAYERS * ATTN_W_PACKS;
static const std::size_t ALL_ATTN_VECTOR_WORDS = NUM_LAYERS * ATTN_B_PACKS;
static const std::size_t FFN_WEIGHT_WORDS =
    (std::size_t)HIDDEN_SIZE * 3072 / PACK_SIZE;
static const std::size_t FFN_B1_WORDS = 3072 / PACK_SIZE;

std::string join_path(const std::string &base, const std::string &leaf)
{
    return base.empty() || base[base.size() - 1] == '/'
        ? base + leaf : base + "/" + leaf;
}

std::vector<bus_t> read_packed(
    const std::string &path, std::size_t expected_words)
{
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open " + path);
    const std::streamoff expected =
        (std::streamoff)(expected_words * PACK_SIZE * sizeof(float));
    if (input.tellg() != expected) {
        throw std::runtime_error("Wrong size for " + path);
    }
    input.seekg(0);
    std::vector<bus_t> words(expected_words);
    float lane[PACK_SIZE];
    for (std::size_t w = 0; w < expected_words; ++w) {
        input.read(reinterpret_cast<char *>(lane), sizeof(lane));
        if (!input) throw std::runtime_error("Short read from " + path);
        words[w] = pack_bus16(lane);
    }
    return words;
}

std::vector<int> read_i32(const std::string &path, std::size_t count)
{
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open " + path);
    if (input.tellg() != (std::streamoff)(count * sizeof(std::int32_t))) {
        throw std::runtime_error("Wrong size for " + path);
    }
    input.seekg(0);
    std::vector<std::int32_t> raw(count);
    input.read(reinterpret_cast<char *>(raw.data()),
               count * sizeof(std::int32_t));
    if (!input) throw std::runtime_error("Short read from " + path);
    return std::vector<int>(raw.begin(), raw.end());
}

std::string layer_directory(const std::string &weights, int layer)
{
    std::ostringstream name;
    name << "layer_" << std::setw(2) << std::setfill('0') << layer;
    return join_path(weights, name.str());
}

std::vector<bus_t> read_all_ffn_layers(
    const std::string &weights,
    const std::string &filename,
    std::size_t words_per_layer)
{
    std::vector<bus_t> result;
    result.reserve(NUM_LAYERS * words_per_layer);
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        std::vector<bus_t> current = read_packed(
            join_path(layer_directory(weights, layer), filename),
            words_per_layer);
        result.insert(result.end(), current.begin(), current.end());
    }
    return result;
}

void write_packed(const std::string &path, const std::vector<bus_t> &words)
{
    std::ofstream output(path.c_str(), std::ios::binary);
    if (!output) throw std::runtime_error("Cannot create " + path);
    float lane[PACK_SIZE];
    for (std::size_t w = 0; w < words.size(); ++w) {
        unpack_bus16(words[w], lane);
        output.write(reinterpret_cast<const char *>(lane), sizeof(lane));
    }
    if (!output) throw std::runtime_error("Write failed for " + path);
}

void require_empty(hidden_stream_t &stream, const char *name)
{
    if (!stream.empty()) {
        throw std::runtime_error(std::string(name) + " contains leftover data");
    }
}

} // namespace

int main(int argc, char **argv)
{
    try {
        if (argc != 3) {
            std::cerr << "Usage: persistent_csim DATA_DIR OUTPUT_DIR\n";
            return 2;
        }
        const std::string data = argv[1];
        const std::string output = argv[2];
        const std::string input_dir = join_path(data, "input");
        const std::string weights = join_path(data, "weights");

        std::cout << "Loading full-model input and all 12 weight sets...\n";
        std::vector<bus_t> initial_hidden = read_packed(
            join_path(input_dir, "embedding_output.f32"), TB_TOKEN_WORDS);
        std::vector<int> mask = read_i32(
            join_path(input_dir, "attention_mask.i32"), SEQ_LEN);
        std::vector<bus_t> qw = read_packed(
            join_path(weights, "attention_q_weight.packed.f32"), ALL_ATTN_WEIGHT_WORDS);
        std::vector<bus_t> kw = read_packed(
            join_path(weights, "attention_k_weight.packed.f32"), ALL_ATTN_WEIGHT_WORDS);
        std::vector<bus_t> vw = read_packed(
            join_path(weights, "attention_v_weight.packed.f32"), ALL_ATTN_WEIGHT_WORDS);
        std::vector<bus_t> ow = read_packed(
            join_path(weights, "attention_o_weight.packed.f32"), ALL_ATTN_WEIGHT_WORDS);
        std::vector<bus_t> qb = read_packed(
            join_path(weights, "attention_q_bias.packed.f32"), ALL_ATTN_VECTOR_WORDS);
        std::vector<bus_t> kb = read_packed(
            join_path(weights, "attention_k_bias.packed.f32"), ALL_ATTN_VECTOR_WORDS);
        std::vector<bus_t> vb = read_packed(
            join_path(weights, "attention_v_bias.packed.f32"), ALL_ATTN_VECTOR_WORDS);
        std::vector<bus_t> ob = read_packed(
            join_path(weights, "attention_o_bias.packed.f32"), ALL_ATTN_VECTOR_WORDS);
        std::vector<bus_t> attn_gamma = read_packed(
            join_path(weights, "attention_norm_gamma.packed.f32"), ALL_ATTN_VECTOR_WORDS);
        std::vector<bus_t> attn_beta = read_packed(
            join_path(weights, "attention_norm_beta.packed.f32"), ALL_ATTN_VECTOR_WORDS);
        std::vector<bus_t> w1 = read_all_ffn_layers(
            weights, "ffn_w1.packed.f32", FFN_WEIGHT_WORDS);
        std::vector<bus_t> b1 = read_all_ffn_layers(
            weights, "ffn_b1.packed.f32", FFN_B1_WORDS);
        std::vector<bus_t> w2 = read_all_ffn_layers(
            weights, "ffn_w2.packed.f32", FFN_WEIGHT_WORDS);
        std::vector<bus_t> b2 = read_all_ffn_layers(
            weights, "ffn_b2.packed.f32", ATTN_B_PACKS);
        std::vector<bus_t> ffn_gamma = read_all_ffn_layers(
            weights, "ffn_norm_gamma.packed.f32", ATTN_B_PACKS);
        std::vector<bus_t> ffn_beta = read_all_ffn_layers(
            weights, "ffn_norm_beta.packed.f32", ATTN_B_PACKS);
        std::vector<bus_t> final_output(TB_TOKEN_WORDS);
        unsigned int ready = 1;

        // Capacities match config/system_12layer.cfg and make C-sim exercise
        // bounded backpressure instead of relying on an infinite software FIFO.
        hidden_stream_t q_stream("q", 256);
        hidden_stream_t k_stream("k", 256);
        hidden_stream_t v_stream("v", 256);
        hidden_stream_t context_stream("context", 512);
        hidden_stream_t attention_residual_stream("attention_residual", 512);
        hidden_stream_t attn_mid_stream("attn_mid", 256);
        hidden_stream_t ffn_residual_stream("ffn_residual", 512);
        hidden_stream_t gelu_stream("gelu", 512);
        hidden_stream_t next_hidden_stream("next_hidden", 512);

        std::cout << "Starting five persistent CUs once..." << std::endl;
        const std::chrono::steady_clock::time_point begin =
            std::chrono::steady_clock::now();

        std::thread down_thread([&] {
            bert_ffn_down_norm_feedback_12layer_kernel(
                gelu_stream, ffn_residual_stream,
                w2.data(), b2.data(), ffn_gamma.data(), ffn_beta.data(),
                next_hidden_stream, final_output.data());
        });
        std::thread up_thread([&] {
            bert_ffn_up_gelu_12layer_kernel(
                attn_mid_stream, w1.data(), b1.data(), gelu_stream);
        });
        std::thread out_thread([&] {
            bert_attn_out_norm_12layer_kernel(
                context_stream, attention_residual_stream,
                ow.data(), ob.data(), attn_gamma.data(), attn_beta.data(),
                attn_mid_stream, ffn_residual_stream);
        });
        std::thread core_thread([&] {
            bert_attn_core_12layer_kernel(
                q_stream, k_stream, v_stream, mask.data(), context_stream);
        });
        std::thread qkv_thread([&] {
            bert_qkv_12layer_kernel(
                initial_hidden.data(), &ready,
                qw.data(), qb.data(), kw.data(), kb.data(), vw.data(), vb.data(),
                next_hidden_stream, attention_residual_stream,
                q_stream, k_stream, v_stream);
        });

        qkv_thread.join();
        core_thread.join();
        out_thread.join();
        up_thread.join();
        down_thread.join();
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();

        require_empty(q_stream, "q_stream");
        require_empty(k_stream, "k_stream");
        require_empty(v_stream, "v_stream");
        require_empty(context_stream, "context_stream");
        require_empty(attention_residual_stream, "attention_residual_stream");
        require_empty(attn_mid_stream, "attn_mid_stream");
        require_empty(ffn_residual_stream, "ffn_residual_stream");
        require_empty(gelu_stream, "gelu_stream");
        require_empty(next_hidden_stream, "next_hidden_stream");

        const std::string output_file =
            join_path(output, "final_encoder_output.f32");
        write_packed(output_file, final_output);
        std::ofstream summary(
            join_path(output, "persistent_csim_summary.txt").c_str());
        summary << "layers=12\n"
                << "kernel_starts=5\n"
                << "host_layer_loop=0\n"
                << "software_seconds=" << std::setprecision(9) << seconds << "\n"
                << "output=" << output_file << "\n";
        std::cout << "Persistent 12-layer C-sim completed in "
                  << seconds << " seconds." << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "PERSISTENT C-SIM ERROR: " << error.what() << '\n';
        return 1;
    }
}
