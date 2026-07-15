#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string join_path(const std::string &base, const std::string &leaf)
{
    return base.empty() || base[base.size() - 1] == '/'
        ? base + leaf : base + "/" + leaf;
}

std::vector<std::uint8_t> read_binary(const std::string &path)
{
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open " + path);
    const std::streamoff size = input.tellg();
    input.seekg(0);
    std::vector<std::uint8_t> data((std::size_t)size);
    input.read(reinterpret_cast<char *>(data.data()), size);
    if (!input) throw std::runtime_error("Short read from " + path);
    return data;
}

std::vector<std::uint8_t> read_ffn_layers(
    const std::string &weights, const std::string &filename)
{
    std::vector<std::uint8_t> result;
    for (int layer = 0; layer < 12; ++layer) {
        std::ostringstream directory;
        directory << "layer_" << std::setw(2) << std::setfill('0') << layer;
        std::vector<std::uint8_t> current = read_binary(
            join_path(join_path(weights, directory.str()), filename));
        result.insert(result.end(), current.begin(), current.end());
    }
    return result;
}

xrt::bo upload(
    const xrt::device &device,
    int group_id,
    const std::vector<std::uint8_t> &data)
{
    xrt::bo buffer(device, data.size(), group_id);
    std::uint8_t *mapped = buffer.map<std::uint8_t *>();
    std::memcpy(mapped, data.data(), data.size());
    buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return buffer;
}

void write_binary(const std::string &path, const void *data, std::size_t size)
{
    std::ofstream output(path.c_str(), std::ios::binary);
    if (!output) throw std::runtime_error("Cannot create " + path);
    output.write(reinterpret_cast<const char *>(data), size);
    if (!output) throw std::runtime_error("Write failed for " + path);
}

} // namespace

int main(int argc, char **argv)
{
    try {
        if (argc < 4 || argc > 5) {
            std::cerr << "Usage: run_persistent_12layer MODEL.xclbin DATA_DIR OUTPUT.f32 [device_index]\n";
            return 2;
        }
        const std::string xclbin_path = argv[1];
        const std::string data_dir = argv[2];
        const std::string output_path = argv[3];
        const unsigned int device_index = argc == 5
            ? (unsigned int)std::stoul(argv[4]) : 0;
        const std::string input = join_path(data_dir, "input");
        const std::string weights = join_path(data_dir, "weights");

        xrt::device device(device_index);
        const xrt::uuid uuid = device.load_xclbin(xclbin_path);
        xrt::kernel qkv(device, uuid, "bert_qkv_12layer_kernel:{qkv12}");
        xrt::kernel core(device, uuid, "bert_attn_core_12layer_kernel:{core12}");
        xrt::kernel out(device, uuid, "bert_attn_out_norm_12layer_kernel:{out12}");
        xrt::kernel up(device, uuid, "bert_ffn_up_gelu_12layer_kernel:{up12}");
        xrt::kernel down(device, uuid, "bert_ffn_down_norm_feedback_12layer_kernel:{down12}");

        std::cout << "Loading and uploading immutable model data..." << std::endl;
        xrt::bo initial_hidden = upload(device, qkv.group_id(0),
            read_binary(join_path(input, "embedding_output.f32")));
        std::vector<std::uint8_t> ready_data(sizeof(std::uint32_t));
        const std::uint32_t ready_value = 1;
        std::memcpy(ready_data.data(), &ready_value, sizeof(ready_value));
        xrt::bo ready = upload(device, qkv.group_id(1), ready_data);
        xrt::bo qw = upload(device, qkv.group_id(2),
            read_binary(join_path(weights, "attention_q_weight.packed.f32")));
        xrt::bo qb = upload(device, qkv.group_id(3),
            read_binary(join_path(weights, "attention_q_bias.packed.f32")));
        xrt::bo kw = upload(device, qkv.group_id(4),
            read_binary(join_path(weights, "attention_k_weight.packed.f32")));
        xrt::bo kb = upload(device, qkv.group_id(5),
            read_binary(join_path(weights, "attention_k_bias.packed.f32")));
        xrt::bo vw = upload(device, qkv.group_id(6),
            read_binary(join_path(weights, "attention_v_weight.packed.f32")));
        xrt::bo vb = upload(device, qkv.group_id(7),
            read_binary(join_path(weights, "attention_v_bias.packed.f32")));

        xrt::bo mask = upload(device, core.group_id(3),
            read_binary(join_path(input, "attention_mask.i32")));

        xrt::bo ow = upload(device, out.group_id(2),
            read_binary(join_path(weights, "attention_o_weight.packed.f32")));
        xrt::bo ob = upload(device, out.group_id(3),
            read_binary(join_path(weights, "attention_o_bias.packed.f32")));
        xrt::bo attn_gamma = upload(device, out.group_id(4),
            read_binary(join_path(weights, "attention_norm_gamma.packed.f32")));
        xrt::bo attn_beta = upload(device, out.group_id(5),
            read_binary(join_path(weights, "attention_norm_beta.packed.f32")));

        xrt::bo w1 = upload(device, up.group_id(1),
            read_ffn_layers(weights, "ffn_w1.packed.f32"));
        xrt::bo b1 = upload(device, up.group_id(2),
            read_ffn_layers(weights, "ffn_b1.packed.f32"));

        xrt::bo w2 = upload(device, down.group_id(2),
            read_ffn_layers(weights, "ffn_w2.packed.f32"));
        xrt::bo b2 = upload(device, down.group_id(3),
            read_ffn_layers(weights, "ffn_b2.packed.f32"));
        xrt::bo ffn_gamma = upload(device, down.group_id(4),
            read_ffn_layers(weights, "ffn_norm_gamma.packed.f32"));
        xrt::bo ffn_beta = upload(device, down.group_id(5),
            read_ffn_layers(weights, "ffn_norm_beta.packed.f32"));
        const std::size_t output_bytes = 128u * 768u * sizeof(float);
        xrt::bo final_output(device, output_bytes, down.group_id(7));

        xrt::run qkv_run(qkv);
        qkv_run.set_arg(0, initial_hidden);
        qkv_run.set_arg(1, ready);
        qkv_run.set_arg(2, qw); qkv_run.set_arg(3, qb);
        qkv_run.set_arg(4, kw); qkv_run.set_arg(5, kb);
        qkv_run.set_arg(6, vw); qkv_run.set_arg(7, vb);
        xrt::run core_run(core);
        core_run.set_arg(3, mask);
        xrt::run out_run(out);
        out_run.set_arg(2, ow); out_run.set_arg(3, ob);
        out_run.set_arg(4, attn_gamma); out_run.set_arg(5, attn_beta);
        xrt::run up_run(up);
        up_run.set_arg(1, w1); up_run.set_arg(2, b1);
        xrt::run down_run(down);
        down_run.set_arg(2, w2); down_run.set_arg(3, b2);
        down_run.set_arg(4, ffn_gamma); down_run.set_arg(5, ffn_beta);
        down_run.set_arg(7, final_output);

        std::cout << "Starting five CUs once (no host layer loop)..." << std::endl;
        const std::chrono::steady_clock::time_point begin =
            std::chrono::steady_clock::now();
        // Start consumers before producers so finite AXI-stream FIFOs cannot
        // fill during the few microseconds of host-side launch skew.
        down_run.start();
        up_run.start();
        out_run.start();
        core_run.start();
        qkv_run.start();
        down_run.wait();
        qkv_run.wait(); core_run.wait(); out_run.wait(); up_run.wait();
        const double device_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();

        const std::chrono::steady_clock::time_point sync_begin =
            std::chrono::steady_clock::now();
        final_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const double output_sync_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - sync_begin).count();
        write_binary(
            output_path, final_output.map<std::uint8_t *>(), output_bytes);

        std::ofstream timing((output_path + ".timing.txt").c_str());
        timing << std::fixed << std::setprecision(6)
               << "device_full_12layer_ms=" << device_ms << '\n'
               << "output_d2h_ms=" << output_sync_ms << '\n'
               << "host_layer_loop=0\n"
               << "kernel_starts=5\n";
        std::cout << std::fixed << std::setprecision(3)
                  << "Full 12-layer device latency: " << device_ms << " ms\n"
                  << "Output transfer: " << output_sync_ms << " ms\n";
        return device_ms <= 200.0 ? 0 : 3;
    } catch (const std::exception &error) {
        std::cerr << "HOST ERROR: " << error.what() << '\n';
        return 1;
    }
}
