#!/bin/bash
MAX_JOBS=4
GPUS=(0 1 2 3 4 5 6 7)
TOTAL_GPUS=${#GPUS[@]}

get_gpu_allocation(){
    local job_number=$1
    # Calculate which GPU to allocate based on the job number
    local gpu_id=${GPUS[$((job_number % TOTAL_GPUS))]}
    echo $gpu_id
}

check_jobs(){
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

job_number=0

DATA_ROOT=./dataset
EXP_NAME=long_term
seed=2023
des='Fredformer'

model_name=Fredformer
datasets=(ETTh1)


normalize=1
auxi_loss=None
ot_type=upper_bound
mask_factor=0.0
auxi_mode=fft_ot

# hyper-parameters
dst=ETTh1

lambda=1.0
lr=0.0001
lradj=type3
train_epochs=100
patience=10
batch_size=128
test_batch_size=1
joint_forecast=1
distance=wasserstein_empirical_per_dim
eps=1e-9
reg_sk=0.005
rerun=1


pl_list=(96 192 336 720)
# NOTE: ETTh1 settings


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.0001 batch_size=128 lradj=type3 alpha=0.1 var_weight=0.002;;
        192) lr=0.0003 batch_size=128 lradj=type3 alpha=0.42 var_weight=0.002;;
        336) lr=0.0006 batch_size=128 lradj=type3 alpha=0.3 var_weight=0.002;;
        720) lr=0.0003 batch_size=128 lradj=type3 alpha=0.1 var_weight=0.002;;
    esac

    case $pl in
        96)  cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        192) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        336) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        720) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
    esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh1.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTh1 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --d_model $d_model \
            --d_ff 128 \
            --dropout 0.3 \
            --fc_dropout 0.3 \
            --patch_len 4 \
            --stride 4 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim\
            --use_nys 0 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done




# hyper-parameters
dst=ETTh2

lambda=1.0
lr=0.0001
lradj=type3
train_epochs=100
patience=10
batch_size=128
test_batch_size=1
joint_forecast=1
distance=wasserstein_empirical_per_dim
eps=1e-9
reg_sk=0.005
rerun=0

pl_list=(96 192 336 720)
# NOTE: ETTh2 settings


for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.001 batch_size=128 lradj=type3 alpha=0.02 var_weight=0.002;;
        192) lr=0.0005 batch_size=128 lradj=type3 alpha=0.1 var_weight=0.002;;
        336) lr=0.001 batch_size=128 lradj=type3 alpha=0.001 var_weight=0.002;;
        720) lr=0.0005 batch_size=128 lradj=type3 alpha=0.01 var_weight=0.002;;
    esac

    case $pl in
        96)  cf_dim=164 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=48;;
        192) cf_dim=164 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=48;;
        336) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        720) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
    esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh2.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTh2 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers 3 \
            --n_heads 4 \
            --d_model $d_model \
            --d_ff 128 \
            --dropout 0.3 \
            --fc_dropout 0.3 \
            --head_dropout 0 \
            --patch_len 4 \
            --stride 4 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 0 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






# hyper-parameters
dst=ETTm1

lambda=1.0
lr=0.0001
lradj=TST
train_epochs=100
patience=10
batch_size=128
test_batch_size=1
joint_forecast=1
distance=wasserstein_empirical_per_dim
eps=1e-9
reg_sk=0.005
rerun=0

pl_list=(96 192 336 720)
# NOTE: ETTm1 settings



for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.0001 batch_size=128 lradj=TST alpha=0.5 var_weight=0.002;;
        192) lr=0.0001 batch_size=128 lradj=TST alpha=0.4 var_weight=0.002;;
        336) lr=0.0006 batch_size=128 lradj=TST alpha=0.2 var_weight=0.002;;
        720) lr=0.0001 batch_size=128 lradj=TST alpha=0.2 var_weight=0.002;;
    esac

    case $pl in
        96)  cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        192) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        336) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        720) cf_dim=164 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=48;;
    esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTm1.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTm1 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers 3 \
            --n_heads 16 \
            --d_model $d_model \
            --d_ff 256 \
            --dropout 0.2 \
            --fc_dropout 0.2 \
            --head_dropout 0 \
            --patch_len 4 \
            --stride 4 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --pct_start 0.4 \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 0 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






# hyper-parameters
dst=ETTm2

lambda=1.0
lr=0.0001
lradj=TST
train_epochs=100
patience=10
batch_size=128
test_batch_size=1
joint_forecast=1
distance=wasserstein_empirical_per_dim
eps=1e-9
reg_sk=0.005
rerun=0

pl_list=(96 192 336 720)
# NOTE: ETTm2 settings



for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.0005 batch_size=128 lradj=TST alpha=0.2 var_weight=0.002;;
        192) lr=0.0002 batch_size=128 lradj=TST alpha=0.2 var_weight=0.002;;
        336) lr=0.0002 batch_size=128 lradj=TST alpha=0.01 var_weight=0.002;;
        720) lr=0.0001 batch_size=128 lradj=TST alpha=0.2 var_weight=0.002;;
    esac

    case $pl in
        96)  cf_dim=164 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=48;;
        192) cf_dim=164 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=48;;
        336) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        720) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
    esac

    rl=$(echo "1 - $alpha" | bc)
    decimal_places=$(echo "$alpha" | awk -F. '{print length($2)}')
    rl=$(printf "%.${decimal_places}f" $rl)
    ax=$alpha

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${eps}_${normalize}_${reg_sk}_${auxi_loss}_${mask_factor}_${distance}_${ot_type}_${joint_forecast}_${auxi_mode}_${var_weight}
    OUTPUT_DIR="./results/${EXP_NAME}/${JOB_NAME}"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.npy" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi


    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTm2.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTm2 \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers 3 \
            --n_heads 16 \
            --d_model $d_model \
            --d_ff 256 \
            --dropout 0.2 \
            --fc_dropout 0.2 \
            --head_dropout 0 \
            --patch_len 4 \
            --stride 4 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --pct_start 0.4 \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 0 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






wait