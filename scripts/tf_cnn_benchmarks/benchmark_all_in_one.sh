#!/bin/bash

model=$1
batch_size=$2
gpu_name=$3 # RTX, V100
run_mode=$4
num_jobs=$5
output_filename=$6

num_gpus=1
num_warmup_batches=0
use_unified_memory=True
allow_growth=True

if [ ${gpu_name} == V100 ]; then
  gpu_id=0
elif [ ${gpu_name} == RTX ]; then
  gpu_id=1
else
  echo "GANGMUK_ERROR: not supported gpu_name: ${gpu_name}"
  exit
fi

if [ ${model} == resnet110 -o ${model} == resnet20 -o ${model} == nasnet_cifar -o ${model} == densenet40_k12 -o ${model} == densenet100_k12 -o ${model} == densenet100_k24 ]; then
  data_name=cifar10
elif [ ${model} == resnet50 -o ${model} == resnet152 -o ${model} == vgg16 -o ${model} == vgg19 -o ${model} == inception3  -o ${model} == inception4 -o ${model} == nasnet -o ${model} == nasnetlarge -o ${model} == ncf ]; then
  data_name=imagenet
elif [ ${model} == deepspeech2 ]; then
  data_name=librispeech

else
  echo "GANGMUK_ERROR: not supported model: ${model}"
  exit
fi

num_batches=50
# if [ ${model} == resnet110 -o ${model} == resnet50 ]; then
#   num_batches=200
# fi

if [ ${run_mode} == SOLO ]; then
  run_mode=SOLO
  # CUDA_VISIBLE_DEVICES=0,1 python3 tf_cnn_benchmarks.py --num_gpus=${num_gpus} --batch_size=${batch_size} \
  #                           --num_batches=${num_batches} --model=${model} \
  #                           --variable_update=parameter_server --local_parameter_device=CPU \
  #                           --use_unified_memory=${use_unified_memory} --allow_growth=${allow_growth} \
  #                           --run_mode=${run_mode} \
  #                           --eval=False \
  #                           --data_name=${data_name} &> ${output_filename}
  CUDA_VISIBLE_DEVICES=${gpu_id} python3 tf_cnn_benchmarks.py --num_gpus=${num_gpus} --batch_size=${batch_size} \
                            --num_batches=${num_batches} --model=${model} \
                            --use_unified_memory=${use_unified_memory} --allow_growth=${allow_growth} \
                            --run_mode=${run_mode} --data_format=NHWC \
                            --eval=False \
                            --data_name=${data_name} &> ${output_filename}
                            # --use_tf_layers=False \

                            # --xla=True \
                            # --xla_compile=True \
                            # --compute_lr_on_cpu=True \

                            # --use_fp16=True \
                            # --fp16_vars=True \
                            # --fp16_loss_scale=1 \
                            # --fp16_enable_auto_loss_scale=True \
                            # --fp16_inc_loss_scale_every_n=1000 \
                            # --distortions=False \

                            # --trace_file=/home/gangmuk/trace_file.json \
                            # --graph_file=/home/gangmuk/${model}_computational_graph.txt \
                            # --tfprof_file=True \
elif [ ${run_mode} == MPS ]; then
  run_mode=SOLO
  CUDA_VISIBLE_DEVICES=${gpu_id} python3 tf_cnn_benchmarks.py --num_gpus=${num_gpus} --batch_size=${batch_size} \
                            --num_batches=${num_batches} --model=${model} \
                            -- variable_update=parameter_server --local_parameter_device=CPU
                            --use_unified_memory=${use_unified_memory} --allow_growth=${allow_growth} \
                            --run_mode=${run_mode} \
                            --data_name=${data_name} &> ${output_filename}
elif [ ${run_mode} == ZICO ]; then
  # if [ ${num_jobs} == 2jobs ]; then
  if [ ${num_jobs} == 2 ]; then
    CUDA_VISIBLE_DEVICES=${gpu_id} python3 tf_cnn_benchmarks.py --num_gpus=${num_gpus} --num_batches=${num_batches} \
                              --use_unified_memory=${use_unified_memory} --allow_growth=${allow_growth} \
                              --model=${model} --batch_size=${batch_size} \
                              --model2=${model} --batch_size2=${batch_size} \
                              --data_name=${data_name} --data_name2=${data_name} \
                              --run_mode=${run_mode} &> ${output_filename}
    # CUDA_VISIBLE_DEVICES=0,1 python3 tf_cnn_benchmarks.py --num_gpus=${num_gpus} --num_batches=${num_batches} \
    #                           --use_unified_memory=${use_unified_memory} --allow_growth=${allow_growth} \
    #                           --data_format=NCHW --variable_update=replicated \
    #                           --model=${model} --batch_size=${batch_size} \
    #                           --model2=${model} --batch_size2=${batch_size} \
    #                           --data_name=${data_name} --data_name2=${data_name} \
    #                           --run_mode=${run_mode} &> ${output_filename}  
  # elif [ ${num_jobs} == 4jobs ]; then
  # elif [ ${num_jobs} == 4 ]; then
  #   CUDA_VISIBLE_DEVICES=${gpu_id} python3 tf_cnn_benchmarks_four_jobs.py --num_gpus=${num_gpus} --num_batches=${num_batches} \
  #                             --use_unified_memory=${use_unified_memory} --allow_growth=${allow_growth} \
  #                             --model=${model} --batch_size=${batch_size} \
  #                             --model2=${model} --batch_size2=${batch_size} \
  #                             --model3=${model} --batch_size3=${batch_size} \
  #                             --model4=${model} --batch_size4=${batch_size} \
  #                             --data_name=${data_name} --data_name2=${data_name} \
  #                             --data_name3=${data_name} --data_name4=${data_name} \
  #                             --run_mode=${num_jobs} &> ${output_filename}
  else
    echo SHELL: [ERROR] Unsupported number of jobs: ${num_jobs}
    exit
  fi
# elif [ ${run_mode} == MPS ]; then
#   run_mode=Solo
#   if [ ${num_jobs} == 2jobs ]; then
#   CUDA_VISIBLE_DEVICES=${gpu_id} python tf_cnn_benchmarks.py --num_gpus=${num_gpus} --batch_size=${batch_size} \
#                           --num_batches=${num_batches} --model=${model} \
#                           --use_unified_memory=${use_unified_memory} --allow_growth=${allow_growth} \
#                           --run_mode=${run_mode} \
#                           --data_name=${data_name} &> ${output_filename}_JobA &

#   CUDA_VISIBLE_DEVICES=${gpu_id} python tf_cnn_benchmarks.py --num_gpus=${num_gpus} --batch_size=${batch_size} \
#                         --num_batches=${num_batches} --model=${model} \
#                         --use_unified_memory=${use_unified_memory} --allow_growth=${allow_growth} \
#                         --run_mode=${run_mode} \
#                         --data_name=${data_name} &> ${output_filename}_JobB
#   elif [ ${num_jobs} == 4jobs ]; then
#     CUDA_VISIBLE_DEVICES=${gpu_id} python tf_cnn_benchmarks.py --num_gpus=${num_gpus} --batch_size=${batch_size} \
#                           --num_batches=${num_batches} --model=${model} \
#                           --use_unified_memory=${use_unified_memory} --allow_growth=${allow_growth} \
#                           --run_mode=${run_mode} \
#                           --data_name=${data_name} &> ${output_filename}_JobA &

#     CUDA_VISIBLE_DEVICES=${gpu_id} python tf_cnn_benchmarks.py --num_gpus=${num_gpus} --batch_size=${batch_size} \
#                           --num_batches=${num_batches} --model=${model} \
#                           --use_unified_memory=${use_unified_memory} --allow_growth=${allow_growth} \
#                           --run_mode=${run_mode} \
#                           --data_name=${data_name} &> ${output_filename}_JobB &
#     CUDA_VISIBLE_DEVICES=${gpu_id} python tf_cnn_benchmarks.py --num_gpus=${num_gpus} --batch_size=${batch_size} \
#                           --num_batches=${num_batches} --model=${model} \
#                           --use_unified_memory=${use_unified_memory} --allow_growth=${allow_growth} \
#                           --run_mode=${run_mode} \
#                           --data_name=${data_name} &> ${output_filename}_JobC &

#     CUDA_VISIBLE_DEVICES=${gpu_id} python tf_cnn_benchmarks.py --num_gpus=${num_gpus} --batch_size=${batch_size} \
#                           --num_batches=${num_batches} --model=${model} \
#                           --use_unified_memory=${use_unified_memory} --allow_growth=${allow_growth} \
#                           --run_mode=${run_mode} \
#                           --data_name=${data_name} &> ${output_filename}_JobD
#   else
#     echo SHELL: [ERROR] Unsupported number of jobs: ${num_jobs}
#     exit
#   fi
else
  echo GANGMUK_ERROR: unsupported run_mode: ${run_mode}
  exit
fi

# if [ ${run_mode} == MPS ]; then
#   cat temp_gpustat >> ${output_filename}_JobA
# else
#   cat temp_gpustat >> ${output_filename}
# fi
# rm temp_gpustat
# # pkill $GPUSTAT_PID
# # killall gpustat

# # pkill $GPUSTAT_PID
# # killall gpustat
# # pkill $MPSTAT_PID
# # killall mpstat
