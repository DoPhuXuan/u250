#include "bert_ffn_kernel_lab.h"

#include <ap_int.h>
#include <hls_stream.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

typedef ap_uint<512> packed_float16_t;
typedef hls::stream<packed_float16_t> packed_float_stream_t;

extern "C" {
void bert_ffn_up_gelu_v21_dotpipe_kernel(
    packed_float_stream_t &, packed_float16_t *, packed_float16_t *, packed_float_stream_t &);
void bert_ffn_down_v21_dotpipe_kernel(
    packed_float_stream_t &, packed_float16_t *, packed_float16_t *, packed_float16_t *);
}

static ap_uint<32> float_bits(float value) {
    union { float f; unsigned int u; } conversion;
    conversion.f = value;
    return conversion.u;
}

static float bits_float(ap_uint<32> bits) {
    union { float f; unsigned int u; } conversion;
    conversion.u = (unsigned int)bits;
    return conversion.f;
}

static std::vector<float> read_f32(const std::string &path, size_t count) {
    std::vector<float> values(count);
    FILE *file = std::fopen(path.c_str(), "rb");
    if (!file || std::fread(values.data(), sizeof(float), count, file) != count) {
        std::fprintf(stderr, "Cannot read %s (%zu floats)\n", path.c_str(), count);
        std::exit(2);
    }
    std::fclose(file);
    return values;
}

static std::vector<packed_float16_t> pack16(const std::vector<float> &values) {
    std::vector<packed_float16_t> packed(values.size() / 16);
    for (size_t word = 0; word < packed.size(); ++word) {
        packed[word] = 0;
        for (int lane = 0; lane < 16; ++lane) {
            packed[word].range(lane * 32 + 31, lane * 32) = float_bits(values[word * 16 + lane]);
        }
    }
    return packed;
}

struct Metrics { double cosine; double max_error; };

static Metrics metrics(const std::vector<float> &reference, const std::vector<float> &candidate) {
    double dot = 0.0, nr = 0.0, nc = 0.0, maximum = 0.0;
    for (size_t index = 0; index < reference.size(); ++index) {
        double r = reference[index], c = candidate[index];
        dot += r * c; nr += r * r; nc += c * c;
        maximum = std::fmax(maximum, std::fabs(c - r));
    }
    Metrics result = {dot / std::sqrt(nr * nc), maximum};
    return result;
}

int main() {
    const char *root_env = std::getenv("FFN_FP32_CSIM_DATA");
    if (!root_env) {
        std::fprintf(stderr, "FFN_FP32_CSIM_DATA is not set\n");
        return 2;
    }
    std::string root(root_env);
    std::vector<float> input = read_f32(root + "/input.bin", HIDDEN_SIZE);
    std::vector<float> w1 = read_f32(root + "/w1_tilemajor.bin", (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE);
    std::vector<float> b1 = read_f32(root + "/b1.bin", INTERMEDIATE_SIZE);
    std::vector<float> w2 = read_f32(root + "/w2_tilemajor.bin", (size_t)HIDDEN_SIZE * INTERMEDIATE_SIZE);
    std::vector<float> b2 = read_f32(root + "/b2.bin", HIDDEN_SIZE);
    std::vector<float> exact = read_f32(root + "/exact_output.bin", HIDDEN_SIZE);
    std::vector<float> pq8 = read_f32(root + "/pq8_output.bin", HIDDEN_SIZE);
    std::vector<packed_float16_t> w1p = pack16(w1), b1p = pack16(b1), w2p = pack16(w2), b2p = pack16(b2);

    packed_float_stream_t input_stream, gelu_stream;
    for (int pack = 0; pack < HIDDEN_SIZE / 16; ++pack) {
        packed_float16_t word = 0;
        for (int lane = 0; lane < 16; ++lane)
            word.range(lane * 32 + 31, lane * 32) = float_bits(input[pack * 16 + lane]);
        input_stream.write(word);
    }
    bert_ffn_up_gelu_v21_dotpipe_kernel(input_stream, w1p.data(), b1p.data(), gelu_stream);
    std::vector<packed_float16_t> output_packed(HIDDEN_SIZE / 16);
    bert_ffn_down_v21_dotpipe_kernel(gelu_stream, w2p.data(), b2p.data(), output_packed.data());

    std::vector<float> actual(HIDDEN_SIZE);
    for (int pack = 0; pack < HIDDEN_SIZE / 16; ++pack)
        for (int lane = 0; lane < 16; ++lane)
            actual[pack * 16 + lane] = bits_float(output_packed[pack].range(lane * 32 + 31, lane * 32));
    FILE *text_output = std::fopen((root + "/csim_output.txt").c_str(), "w");
    if (!text_output) {
        std::fprintf(stderr, "Cannot create csim_output.txt\n");
        return 2;
    }
    for (int hidden = 0; hidden < HIDDEN_SIZE; ++hidden)
        std::fprintf(text_output, "%.9g\n", actual[hidden]);
    std::fclose(text_output);
    Metrics versus_exact = metrics(exact, actual);
    Metrics versus_pq8 = metrics(pq8, actual);
    std::printf("CSIM real BERT layer0 row: cosine vs exact erf = %.12f, max error = %.9g\n",
                versus_exact.cosine, versus_exact.max_error);
    std::printf("CSIM real BERT layer0 row: cosine vs PQ8 golden = %.12f, max error = %.9g\n",
                versus_pq8.cosine, versus_pq8.max_error);
    if (versus_exact.cosine < 0.99 || versus_pq8.cosine < 0.999999) return 1;
    std::printf("PASS: HLS CSim matches PQ8 golden and exceeds 0.99 cosine versus original FFN.\n");
    return 0;
}
