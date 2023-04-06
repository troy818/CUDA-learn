#!/bin/bash

# profile metric according to "metric", use cuda profile api
DATE=`date "+%Y%m%d_%H%M%S"`
sudo /usr/local/cuda/bin/ncu \
    --profile-from-start off -f -o nc-%h-$DATE \
    --metrics dram__bytes.sum,lts__t_bytes.sum,l1tex__t_bytes.sum,sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,sm__instruction_throughput.avg.pct_of_peak_sustained_active,sm__inst_executed.avg.per_cycle_elapsed,dram__bytes.sum.per_second,lts__t_sector_hit_rate.pct,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,lts__t_bytes.sum,sm__sass_thread_inst_executed_op_hadd_pred_on.sum,sm__sass_thread_inst_executed_op_hfma_pred_on.sum,sm__sass_thread_inst_executed_op_hmul_pred_on.sum,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
    --replay-mode kernel --target-processes all \
    ../build/ErosionCase

# print profile result and code optimization suggestions in terminal
sudo /usr/local/cuda/bin/ncu --profile-from-start on ../build/ErosionCase
