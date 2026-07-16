#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
constexpr std::size_t kNumLayers = 12;
constexpr std::size_t kSeqLen = 128;
constexpr std::size_t kHidden = 768;
constexpr std::size_t kFfnDim = 3072;
constexpr std::size_t kVocab = 30522;

constexpr std::size_t bytes_f32(std::size_t count) { return count * sizeof(float); }
constexpr std::size_t bytes_i32(std::size_t count) { return count * sizeof(std::int32_t); }

struct Options {
  std::string xclbin = "bert_pipeline_hw_200.xclbin";
  std::string data_dir = "bert_base_uncased_u250";
  std::string output = "last_hidden_state.bin";
  unsigned int device_index = 0;
};

[[noreturn]] void usage(const char* program, const std::string& error = {}) {
  if (!error.empty()) std::cerr << "error: " << error << "\n\n";
  std::cerr << "Usage: " << program
            << " [--xclbin FILE] [--data-dir DIR] [--output FILE] [--device N]\n";
  std::exit(error.empty() ? 0 : 2);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const char* name) -> std::string {
      if (++i >= argc) usage(argv[0], std::string("missing value for ") + name);
      return argv[i];
    };
    if (arg == "--xclbin") options.xclbin = value("--xclbin");
    else if (arg == "--data-dir") options.data_dir = value("--data-dir");
    else if (arg == "--output") options.output = value("--output");
    else if (arg == "--device") options.device_index = std::stoul(value("--device"));
    else if (arg == "-h" || arg == "--help") usage(argv[0]);
    else usage(argv[0], "unknown argument: " + arg);
  }
  return options;
}

void require_regular_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("file not found: " + path);
}

std::string join_path(const std::string& directory, const std::string& name) {
  if (directory.empty()) return name;
  const char last = directory.back();
  return directory + ((last == '/' || last == '\\') ? "" : "/") + name;
}

void load_bo(xrt::bo& bo, const std::string& path, std::size_t expected_bytes) {
  require_regular_file(path);
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  const auto actual = static_cast<std::size_t>(input.tellg());
  if (actual != expected_bytes)
    throw std::runtime_error(path + ": expected " + std::to_string(expected_bytes) +
                             " bytes, got " + std::to_string(actual));
  input.seekg(0);
  auto* destination = bo.map<char*>();
  input.read(destination, static_cast<std::streamsize>(expected_bytes));
  if (!input) throw std::runtime_error("failed to read: " + path);
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

void zero_bo(xrt::bo& bo, std::size_t bytes) {
  std::fill_n(bo.map<unsigned char*>(), bytes, static_cast<unsigned char>(0));
  bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

void save_output(xrt::bo& bo, const std::string& path) {
  const std::size_t bytes = bytes_f32(kSeqLen * kHidden);
  bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const auto* values = bo.map<float*>();
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(values), static_cast<std::streamsize>(bytes));
  if (!output) throw std::runtime_error("failed to write: " + path);

  std::cout << "First token, first 16 values:";
  for (std::size_t i = 0; i < 16; ++i)
    std::cout << ' ' << std::fixed << std::setprecision(6) << values[i];
  std::cout << '\n';
}
}  // namespace

int main(int argc, char** argv) try {
  const Options options = parse_options(argc, argv);
  require_regular_file(options.xclbin);

  std::cout << "Opening XRT device " << options.device_index << "\n";
  xrt::device device(options.device_index);
  const auto uuid = device.load_xclbin(options.xclbin);

  xrt::kernel k_emb(device, uuid, "bert_embedding_prep_kernel");
  xrt::kernel k_qkv(device, uuid, "bert_qkv_kernel");
  xrt::kernel k_core(device, uuid, "bert_attn_core_kernel");
  xrt::kernel k_attn_out(device, uuid, "bert_attn_out_norm_kernel");
  xrt::kernel k_ffn_up(device, uuid, "bert_ffn_up_gelu_kernel");
  xrt::kernel k_ffn_down(device, uuid, "bert_ffn_down_norm_kernel");

  // BO placement is obtained from the actual xclbin connectivity metadata.
  // Shared BOs use ports which resolve to the same physical DDR bank.
  xrt::bo input_ids(device, bytes_i32(kSeqLen), k_emb.group_id(0));
  xrt::bo token_type_ids(device, bytes_i32(kSeqLen), k_emb.group_id(1));
  xrt::bo attention_mask(device, bytes_i32(kSeqLen), k_core.group_id(1));
  xrt::bo hidden_ping(device, bytes_f32(kSeqLen * kHidden), k_emb.group_id(7));
  xrt::bo hidden_pong(device, bytes_f32(kSeqLen * kHidden), k_ffn_down.group_id(7));
  xrt::bo hidden_ready(device, sizeof(std::uint32_t), k_emb.group_id(8));
  xrt::bo attn_mid(device, bytes_f32(kSeqLen * kHidden), k_attn_out.group_id(7));
  xrt::bo attn_mid_done(device, sizeof(std::uint32_t), k_attn_out.group_id(8));

  xrt::bo token_emb(device, bytes_f32(kVocab * kHidden), k_emb.group_id(2));
  xrt::bo pos_emb(device, bytes_f32(kSeqLen * kHidden), k_emb.group_id(3));
  xrt::bo seg_emb(device, bytes_f32(2 * kHidden), k_emb.group_id(4));
  xrt::bo emb_gamma(device, bytes_f32(kHidden), k_emb.group_id(5));
  xrt::bo emb_beta(device, bytes_f32(kHidden), k_emb.group_id(6));

  const std::size_t attn_w_bytes = bytes_f32(kNumLayers * kHidden * kHidden);
  const std::size_t attn_b_bytes = bytes_f32(kNumLayers * kHidden);
  const std::size_t ffn_w_bytes = bytes_f32(kNumLayers * kHidden * kFfnDim);
  const std::size_t ffn_up_b_bytes = bytes_f32(kNumLayers * kFfnDim);
  const std::size_t norm_bytes = bytes_f32(kNumLayers * kHidden);

  xrt::bo q_w(device, attn_w_bytes, k_qkv.group_id(2));
  xrt::bo q_b(device, attn_b_bytes, k_qkv.group_id(3));
  xrt::bo k_w(device, attn_w_bytes, k_qkv.group_id(4));
  xrt::bo k_b(device, attn_b_bytes, k_qkv.group_id(5));
  xrt::bo v_w(device, attn_w_bytes, k_qkv.group_id(6));
  xrt::bo v_b(device, attn_b_bytes, k_qkv.group_id(7));
  xrt::bo o_w(device, attn_w_bytes, k_attn_out.group_id(3));
  xrt::bo o_b(device, attn_b_bytes, k_attn_out.group_id(4));
  xrt::bo attn_gamma(device, norm_bytes, k_attn_out.group_id(5));
  xrt::bo attn_beta(device, norm_bytes, k_attn_out.group_id(6));
  xrt::bo ffn_up_w(device, ffn_w_bytes, k_ffn_up.group_id(1));
  xrt::bo ffn_up_b(device, ffn_up_b_bytes, k_ffn_up.group_id(2));
  xrt::bo ffn_down_w(device, ffn_w_bytes, k_ffn_down.group_id(3));
  xrt::bo ffn_down_b(device, attn_b_bytes, k_ffn_down.group_id(4));
  xrt::bo ffn_gamma(device, norm_bytes, k_ffn_down.group_id(5));
  xrt::bo ffn_beta(device, norm_bytes, k_ffn_down.group_id(6));

  const auto file = [&](const char* name) { return join_path(options.data_dir, name); };
  std::cout << "Loading model/input BOs from " << options.data_dir << "\n";
  load_bo(input_ids, file("input_ids.bin"), bytes_i32(kSeqLen));
  load_bo(token_type_ids, file("token_type_ids.bin"), bytes_i32(kSeqLen));
  load_bo(attention_mask, file("attention_mask.bin"), bytes_i32(kSeqLen));
  load_bo(token_emb, file("token_emb.bin"), bytes_f32(kVocab * kHidden));
  load_bo(pos_emb, file("pos_emb.bin"), bytes_f32(kSeqLen * kHidden));
  load_bo(seg_emb, file("seg_emb.bin"), bytes_f32(2 * kHidden));
  load_bo(emb_gamma, file("emb_gamma.bin"), bytes_f32(kHidden));
  load_bo(emb_beta, file("emb_beta.bin"), bytes_f32(kHidden));
  load_bo(q_w, file("attn_q_w.bin"), attn_w_bytes);
  load_bo(q_b, file("attn_q_b.bin"), attn_b_bytes);
  load_bo(k_w, file("attn_k_w.bin"), attn_w_bytes);
  load_bo(k_b, file("attn_k_b.bin"), attn_b_bytes);
  load_bo(v_w, file("attn_v_w.bin"), attn_w_bytes);
  load_bo(v_b, file("attn_v_b.bin"), attn_b_bytes);
  load_bo(o_w, file("attn_o_w.bin"), attn_w_bytes);
  load_bo(o_b, file("attn_o_b.bin"), attn_b_bytes);
  load_bo(attn_gamma, file("attn_norm_gamma.bin"), norm_bytes);
  load_bo(attn_beta, file("attn_norm_beta.bin"), norm_bytes);
  load_bo(ffn_up_w, file("ffn_up_w.bin"), ffn_w_bytes);
  load_bo(ffn_up_b, file("ffn_up_b.bin"), ffn_up_b_bytes);
  load_bo(ffn_down_w, file("ffn_down_w.bin"), ffn_w_bytes);
  load_bo(ffn_down_b, file("ffn_down_b.bin"), attn_b_bytes);
  load_bo(ffn_gamma, file("ffn_norm_gamma.bin"), norm_bytes);
  load_bo(ffn_beta, file("ffn_norm_beta.bin"), norm_bytes);
  zero_bo(hidden_ping, bytes_f32(kSeqLen * kHidden));
  zero_bo(hidden_pong, bytes_f32(kSeqLen * kHidden));
  zero_bo(hidden_ready, sizeof(std::uint32_t));
  zero_bo(attn_mid, bytes_f32(kSeqLen * kHidden));
  zero_bo(attn_mid_done, sizeof(std::uint32_t));

  const auto begin = std::chrono::steady_clock::now();
  for (std::uint32_t layer = 0; layer < kNumLayers; ++layer) {
    const bool even = (layer & 1U) == 0;
    xrt::bo& hidden_in = even ? hidden_ping : hidden_pong;
    xrt::bo& hidden_out = even ? hidden_pong : hidden_ping;

    xrt::run r_emb(k_emb);
    r_emb.set_arg(0, input_ids); r_emb.set_arg(1, token_type_ids);
    r_emb.set_arg(2, token_emb); r_emb.set_arg(3, pos_emb); r_emb.set_arg(4, seg_emb);
    r_emb.set_arg(5, emb_gamma); r_emb.set_arg(6, emb_beta);
    r_emb.set_arg(7, hidden_ping); r_emb.set_arg(8, hidden_ready); r_emb.set_arg(9, layer);

    xrt::run r_qkv(k_qkv);
    r_qkv.set_arg(0, hidden_in); r_qkv.set_arg(1, hidden_ready);
    r_qkv.set_arg(2, q_w); r_qkv.set_arg(3, q_b); r_qkv.set_arg(4, k_w);
    r_qkv.set_arg(5, k_b); r_qkv.set_arg(6, v_w); r_qkv.set_arg(7, v_b);
    r_qkv.set_arg(8, layer);  // arg 9 is the internally connected AXI stream.

    xrt::run r_core(k_core);
    r_core.set_arg(1, attention_mask);  // args 0 and 2 are connected streams.

    xrt::run r_attn_out(k_attn_out);
    r_attn_out.set_arg(1, hidden_in); r_attn_out.set_arg(2, hidden_ready);
    r_attn_out.set_arg(3, o_w); r_attn_out.set_arg(4, o_b);
    r_attn_out.set_arg(5, attn_gamma); r_attn_out.set_arg(6, attn_beta);
    r_attn_out.set_arg(7, attn_mid); r_attn_out.set_arg(8, attn_mid_done);
    r_attn_out.set_arg(9, layer);  // args 0 and 10 are connected streams.

    xrt::run r_ffn_up(k_ffn_up);
    r_ffn_up.set_arg(1, ffn_up_w); r_ffn_up.set_arg(2, ffn_up_b);
    r_ffn_up.set_arg(3, layer);  // args 0 and 4 are connected streams.

    xrt::run r_ffn_down(k_ffn_down);
    r_ffn_down.set_arg(1, attn_mid); r_ffn_down.set_arg(2, attn_mid_done);
    r_ffn_down.set_arg(3, ffn_down_w); r_ffn_down.set_arg(4, ffn_down_b);
    r_ffn_down.set_arg(5, ffn_gamma); r_ffn_down.set_arg(6, ffn_beta);
    r_ffn_down.set_arg(7, hidden_out); r_ffn_down.set_arg(8, layer);

    // Start sink-to-source. This lets every connected stream have a live consumer.
    r_ffn_down.start(); r_ffn_up.start(); r_attn_out.start();
    r_core.start(); r_qkv.start(); r_emb.start();

    r_ffn_down.wait(); r_ffn_up.wait(); r_attn_out.wait();
    r_core.wait(); r_qkv.wait(); r_emb.wait();
    std::cout << "Layer " << layer << " complete\n";
  }

  const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin);
  save_output(hidden_ping, options.output);  // Layer 11 writes ping (12 is even).
  std::cout << "Saved [128,768] FP32 last_hidden_state to " << options.output << "\n"
            << "Kernel pipeline time: " << elapsed.count() << " s\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "fatal: " << error.what() << '\n';
  return 1;
}
