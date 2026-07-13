#ifndef BERT_TYPES_H
#define BERT_TYPES_H

#include "bert_config.h"
#include <ap_int.h>
#include <hls_stream.h>

typedef ap_uint<512> bus_t;
typedef bus_t qkv_weight_bus_t;
typedef hls::stream<bus_t> hidden_stream_t;

#endif
