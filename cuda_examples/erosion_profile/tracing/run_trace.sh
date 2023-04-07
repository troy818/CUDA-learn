#!/usr/bin/bash

# kairos.cudashim: https://github.com/intel-sandbox/kairos.cudashim

export CUDASHIM_OUTPUT_FORMAT=chrometrace|csv
export CUDA_INJECTION64_PATH=/path/to/kairos.cudashim/build/libcudashim.so
export LD_LIBRARY_PATH="/path/to/kairos.cudashim/build:$LD_LIBRARY_PATH"

../build/ErosionCase