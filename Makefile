CXX ?= g++
XILINX_XRT ?= /opt/xilinx/xrt

CXXFLAGS += -std=c++14 -O2 -Wall -Wextra -I$(XILINX_XRT)/include
LDFLAGS += -L$(XILINX_XRT)/lib -lxrt_coreutil -pthread

.PHONY: all clean export run

all: host_bert_u250

host_bert_u250: host_bert_u250.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

export:
	python3 export_bert_base_uncased.py --output-dir bert_base_uncased_u250

run: host_bert_u250
	./host_bert_u250 --xclbin bert_pipeline_hw_200.xclbin --data-dir bert_base_uncased_u250

clean:
	rm -f host_bert_u250 last_hidden_state.bin
