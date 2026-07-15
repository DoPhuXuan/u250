#ifndef BERT_TYPES_H
#define BERT_TYPES_H

#include "bert_config.h"
#include "bert_stream_compat.h"
#include <ap_int.h>

typedef ap_uint<512> bus_t;
typedef bus_t qkv_weight_bus_t;
typedef bert_stream_t<bus_t> hidden_stream_t;

#endif
