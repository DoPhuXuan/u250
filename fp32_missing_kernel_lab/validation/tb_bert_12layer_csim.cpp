#include "bert_kernel_interfaces.h"
#include "bert_math.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

static const std::size_t TOKEN_WORD_COUNT = SEQ_LEN * PACKS;
static const std::size_t ALL_ATTN_WEIGHT_WORDS = NUM_LAYERS * ATTN_W_PACKS;
static const std::size_t ALL_ATTN_VECTOR_WORDS = NUM_LAYERS * ATTN_B_PACKS;
static const std::size_t FFN_WEIGHT_WORDS =
    (std::size_t)HIDDEN_SIZE * 3072 / PACK_SIZE;

std::string join_path(const std::string &base, const std::string &leaf)
{
    if (base.empty() || base[base.size() - 1] == '/') {
        return base + leaf;
    }
    return base + "/" + leaf;
}

std::vector<bus_t> read_packed_f32(
    const std::string &path, std::size_t expected_words)
{
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Cannot open " + path);
    }
    const std::streamoff bytes = input.tellg();
    const std::streamoff expected =
        (std::streamoff)(expected_words * PACK_SIZE * sizeof(float));
    if (bytes != expected) {
        throw std::runtime_error(
            "Wrong size for " + path + ": got " + std::to_string((long long)bytes)
            + ", expected " + std::to_string((long long)expected));
    }
    input.seekg(0);

    std::vector<bus_t> words(expected_words);
    float lane[PACK_SIZE];
    for (std::size_t word_index = 0; word_index < expected_words; ++word_index) {
        input.read(reinterpret_cast<char *>(lane), sizeof(lane));
        if (!input) {
            throw std::runtime_error("Short read from " + path);
        }
        words[word_index] = pack_bus16(lane);
    }
    return words;
}

std::vector<int> read_i32(const std::string &path, std::size_t expected_count)
{
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Cannot open " + path);
    }
    const std::streamoff expected =
        (std::streamoff)(expected_count * sizeof(std::int32_t));
    if (input.tellg() != expected) {
        throw std::runtime_error("Wrong size for " + path);
    }
    input.seekg(0);
    std::vector<std::int32_t> raw(expected_count);
    input.read(reinterpret_cast<char *>(raw.data()), expected);
    if (!input) {
        throw std::runtime_error("Short read from " + path);
    }
    return std::vector<int>(raw.begin(), raw.end());
}

void write_packed_f32(const std::string &path, const std::vector<bus_t> &words)
{
    std::ofstream output(path.c_str(), std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot create " + path);
    }
    float lane[PACK_SIZE];
    for (std::size_t word_index = 0; word_index < words.size(); ++word_index) {
        unpack_bus16(words[word_index], lane);
        output.write(reinterpret_cast<const char *>(lane), sizeof(lane));
    }
    if (!output) {
        throw std::runtime_error("Write failed for " + path);
    }
}

struct Statistics {
    double minimum;
    double maximum;
    double mean;
    std::size_t nonfinite;
};

Statistics statistics(const std::vector<bus_t> &words)
{
    Statistics result;
    result.minimum = std::numeric_limits<double>::infinity();
    result.maximum = -std::numeric_limits<double>::infinity();
    result.mean = 0.0;
    result.nonfinite = 0;
    std::size_t finite_count = 0;
    float lane[PACK_SIZE];
    for (std::size_t w = 0; w < words.size(); ++w) {
        unpack_bus16(words[w], lane);
        for (int i = 0; i < PACK_SIZE; ++i) {
            const double value = lane[i];
            if (!std::isfinite(value)) {
                ++result.nonfinite;
                continue;
            }
            if (value < result.minimum) result.minimum = value;
            if (value > result.maximum) result.maximum = value;
            result.mean += value;
            ++finite_count;
        }
    }
    if (finite_count != 0) result.mean /= finite_count;
    return result;
}

std::string layer_name(int layer)
{
    std::ostringstream name;
    name << "layer_" << std::setw(2) << std::setfill('0') << layer;
    return name.str();
}

void require_stream_empty(hidden_stream_t &stream, const char *name)
{
    if (!stream.empty()) {
        throw std::runtime_error(std::string(name) + " was not fully consumed");
    }
}

} // namespace

int main(int argc, char **argv)
{
    try {
        if (argc != 3) {
            std::cerr << "Usage: tb_bert_12layer_csim DATA_DIR OUTPUT_DIR\n";
            return 2;
        }
        const std::string data_dir = argv[1];
        const std::string output_dir = argv[2];
        const std::string input_dir = join_path(data_dir, "input");
        const std::string weights_dir = join_path(data_dir, "weights");

        std::cout << "Loading input and all 12 Attention parameter sets...\n";
        std::vector<bus_t> hidden = read_packed_f32(
            join_path(input_dir, "embedding_output.f32"), TOKEN_WORD_COUNT);
        std::vector<int> attention_mask = read_i32(
            join_path(input_dir, "attention_mask.i32"), SEQ_LEN);

        std::vector<bus_t> qw = read_packed_f32(
            join_path(weights_dir, "attention_q_weight.packed.f32"),
            ALL_ATTN_WEIGHT_WORDS);
        std::vector<bus_t> kw = read_packed_f32(
            join_path(weights_dir, "attention_k_weight.packed.f32"),
            ALL_ATTN_WEIGHT_WORDS);
        std::vector<bus_t> vw = read_packed_f32(
            join_path(weights_dir, "attention_v_weight.packed.f32"),
            ALL_ATTN_WEIGHT_WORDS);
        std::vector<bus_t> ow = read_packed_f32(
            join_path(weights_dir, "attention_o_weight.packed.f32"),
            ALL_ATTN_WEIGHT_WORDS);
        std::vector<bus_t> qb = read_packed_f32(
            join_path(weights_dir, "attention_q_bias.packed.f32"),
            ALL_ATTN_VECTOR_WORDS);
        std::vector<bus_t> kb = read_packed_f32(
            join_path(weights_dir, "attention_k_bias.packed.f32"),
            ALL_ATTN_VECTOR_WORDS);
        std::vector<bus_t> vb = read_packed_f32(
            join_path(weights_dir, "attention_v_bias.packed.f32"),
            ALL_ATTN_VECTOR_WORDS);
        std::vector<bus_t> ob = read_packed_f32(
            join_path(weights_dir, "attention_o_bias.packed.f32"),
            ALL_ATTN_VECTOR_WORDS);
        std::vector<bus_t> attn_gamma = read_packed_f32(
            join_path(weights_dir, "attention_norm_gamma.packed.f32"),
            ALL_ATTN_VECTOR_WORDS);
        std::vector<bus_t> attn_beta = read_packed_f32(
            join_path(weights_dir, "attention_norm_beta.packed.f32"),
            ALL_ATTN_VECTOR_WORDS);

        std::ofstream summary(join_path(output_dir, "csim_layer_outputs.csv").c_str());
        if (!summary) throw std::runtime_error("Cannot create C-sim summary CSV");
        summary << "layer,stage,seconds,min,max,mean,nonfinite,file\n";

        unsigned int hidden_ready = 1;
        unsigned int attn_mid_done = 0;
        for (int layer = 0; layer < NUM_LAYERS; ++layer) {
            const std::string layer_dir =
                join_path(weights_dir, layer_name(layer));
            std::cout << "[layer " << std::setw(2) << std::setfill('0') << layer
                      << "] loading FFN parameters..." << std::endl;
            std::vector<bus_t> w1 = read_packed_f32(
                join_path(layer_dir, "ffn_w1.packed.f32"), FFN_WEIGHT_WORDS);
            std::vector<bus_t> b1 = read_packed_f32(
                join_path(layer_dir, "ffn_b1.packed.f32"), 3072 / PACK_SIZE);
            std::vector<bus_t> w2 = read_packed_f32(
                join_path(layer_dir, "ffn_w2.packed.f32"), FFN_WEIGHT_WORDS);
            std::vector<bus_t> b2 = read_packed_f32(
                join_path(layer_dir, "ffn_b2.packed.f32"), ATTN_B_PACKS);
            std::vector<bus_t> ffn_gamma = read_packed_f32(
                join_path(layer_dir, "ffn_norm_gamma.packed.f32"), ATTN_B_PACKS);
            std::vector<bus_t> ffn_beta = read_packed_f32(
                join_path(layer_dir, "ffn_norm_beta.packed.f32"), ATTN_B_PACKS);

            hidden_stream_t q_stream("tb_q_stream");
            hidden_stream_t k_stream("tb_k_stream");
            hidden_stream_t v_stream("tb_v_stream");
            hidden_stream_t context_stream("tb_context_stream");
            hidden_stream_t attn_mid_stream("tb_attn_mid_stream");
            hidden_stream_t ffn_residual_stream("tb_ffn_residual_stream");
            hidden_stream_t gelu_stream("tb_gelu_stream");
            std::vector<bus_t> attn_mid(TOKEN_WORD_COUNT);
            std::vector<bus_t> next_hidden(TOKEN_WORD_COUNT);

            const std::chrono::steady_clock::time_point begin =
                std::chrono::steady_clock::now();
            bert_qkv_kernel(
                hidden.data(), &hidden_ready,
                qw.data(), qb.data(), kw.data(), kb.data(), vw.data(), vb.data(),
                layer, q_stream, k_stream, v_stream);
            bert_attn_core_kernel(
                q_stream, k_stream, v_stream, attention_mask.data(), context_stream);
            bert_attn_out_norm_residual_kernel(
                context_stream, hidden.data(), &hidden_ready,
                ow.data(), ob.data(), attn_gamma.data(), attn_beta.data(),
                attn_mid.data(), &attn_mid_done, layer,
                attn_mid_stream, ffn_residual_stream);
            const std::chrono::steady_clock::time_point attention_end =
                std::chrono::steady_clock::now();
            bert_ffn_up_gelu_v21_dotpipe_kernel(
                attn_mid_stream, w1.data(), b1.data(), gelu_stream);
            bert_ffn_down_residual_norm_fp32_kernel(
                gelu_stream, ffn_residual_stream,
                w2.data(), b2.data(), ffn_gamma.data(), ffn_beta.data(),
                next_hidden.data());
            const std::chrono::steady_clock::time_point end =
                std::chrono::steady_clock::now();

            require_stream_empty(q_stream, "q_stream");
            require_stream_empty(k_stream, "k_stream");
            require_stream_empty(v_stream, "v_stream");
            require_stream_empty(context_stream, "context_stream");
            require_stream_empty(attn_mid_stream, "attn_mid_stream");
            require_stream_empty(ffn_residual_stream, "ffn_residual_stream");
            require_stream_empty(gelu_stream, "gelu_stream");
            if (attn_mid_done != (unsigned int)(layer + 1)) {
                throw std::runtime_error("Attention completion token mismatch");
            }

            const double attention_seconds =
                std::chrono::duration<double>(attention_end - begin).count();
            const double encoder_seconds =
                std::chrono::duration<double>(end - begin).count();
            const Statistics attn_stats = statistics(attn_mid);
            const Statistics encoder_stats = statistics(next_hidden);
            if (attn_stats.nonfinite != 0 || encoder_stats.nonfinite != 0) {
                throw std::runtime_error("Non-finite value produced at " + layer_name(layer));
            }

            const std::string attn_file =
                "attention_" + layer_name(layer) + ".f32";
            const std::string encoder_file =
                "encoder_" + layer_name(layer) + ".f32";
            write_packed_f32(join_path(output_dir, attn_file), attn_mid);
            write_packed_f32(join_path(output_dir, encoder_file), next_hidden);
            summary << layer << ",attention," << std::setprecision(9)
                    << attention_seconds << ',' << attn_stats.minimum << ','
                    << attn_stats.maximum << ',' << attn_stats.mean << ','
                    << attn_stats.nonfinite << ',' << attn_file << '\n';
            summary << layer << ",encoder," << encoder_seconds << ','
                    << encoder_stats.minimum << ',' << encoder_stats.maximum << ','
                    << encoder_stats.mean << ',' << encoder_stats.nonfinite << ','
                    << encoder_file << '\n';
            summary.flush();

            std::cout << "[layer " << std::setw(2) << std::setfill('0') << layer
                      << "] attention=" << attention_seconds
                      << " s, complete encoder=" << encoder_seconds << " s"
                      << std::endl;
            hidden.swap(next_hidden);
        }
        std::cout << "C-sim completed all 12 encoder layers successfully.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "C-SIM ERROR: " << error.what() << '\n';
        return 1;
    }
}
