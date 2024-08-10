export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --mixed_precision fp16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file deepspeed/stage3_no_offloading_accelerate.conf \
    finetune.py \
    --model_name_or_path SMART_Short_${MODEL_SIZE} \
    --tokenizer_name SMART_Short_${MODEL_SIZE}  \
    --use_slow_tokenizer \
    --train_file /dataset/long-trajectory.json \
    --max_seq_length 3076 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-4 \
    --lr_scheduler_type linear \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --warmup_ratio 0.03 \
    --output_dir output/SMART_${MODEL_SIZE}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_lora \
    --use_special_tokens \


