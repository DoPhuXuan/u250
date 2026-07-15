#ifndef BERT_CONFIG_H
#define BERT_CONFIG_H

// Fixed BERT-base geometry used by the three split Attention kernels.
static const int NUM_LAYERS = 12;
static const int SEQ_LEN = 128;
static const int HIDDEN_SIZE = 768;
static const int HIDDEN = HIDDEN_SIZE;
static const int NUM_HEADS = 12;
static const int HEAD_DIM = 64;
static const int HEAD_PAR = 2;
static const int HEAD_GROUPS = NUM_HEADS / HEAD_PAR;
static const int GROUP_DIM = HEAD_PAR * HEAD_DIM;

static const int PACK_SIZE = 16;
static const int PACKS = HIDDEN_SIZE / PACK_SIZE;
static const int GROUP_PACKS = GROUP_DIM / PACK_SIZE;
static const int TOKEN_WORDS = SEQ_LEN * PACKS;

// Route-friendly compute profiles retained from the proven split baseline.
static const int QKV_TILE_O = 64;
static const int QKV_K_PAR = 8;
static const int QKV_O_PAR = 8;
static const int QKV_ACC_BANKS = 8;
static const int ATTN_D_PAR = 8;
static const int ATTN_CONTEXT_BANKS = 16;
static const int ATTN_ACC_BANKS = 8;
static const int OUT_TILE_O = 32;
static const int OUT_K_PAR = 8;
static const int OUT_O_PAR = 8;

static const int ATTN_W_FLOATS = HIDDEN_SIZE * HIDDEN_SIZE;
static const int ATTN_W_PACKS = ATTN_W_FLOATS / PACK_SIZE;
static const int ATTN_B_PACKS = HIDDEN_SIZE / PACK_SIZE;

static_assert(HIDDEN_SIZE == NUM_HEADS * HEAD_DIM,
              "hidden size must equal heads times head dimension");
static_assert((HIDDEN_SIZE % PACK_SIZE) == 0, "hidden size must be packed");
static_assert((GROUP_DIM % QKV_TILE_O) == 0, "group must contain whole QKV tiles");
static_assert((QKV_TILE_O % QKV_O_PAR) == 0,
              "QKV output tile must contain whole output-lane blocks");
static_assert((QKV_ACC_BANKS & (QKV_ACC_BANKS - 1)) == 0,
              "QKV accumulator banks must be a power of two");
static_assert(QKV_ACC_BANKS == 8,
              "named QKV accumulator lanes require exactly eight banks");
static_assert((HEAD_DIM % ATTN_D_PAR) == 0, "head dimension must match MAC lanes");
static_assert(ATTN_D_PAR == 8,
              "Core rotating softmax sums require exactly eight lanes");
static_assert((HEAD_DIM % ATTN_CONTEXT_BANKS) == 0,
              "head dimension must contain whole context chunks");
static_assert((GROUP_DIM % ATTN_CONTEXT_BANKS) == 0,
              "context buffer must match context banking");
static_assert((ATTN_ACC_BANKS & (ATTN_ACC_BANKS - 1)) == 0,
              "attention accumulator banks must be a power of two");
static_assert((OUT_TILE_O % PACK_SIZE) == 0, "output tile must be packed");

#endif
