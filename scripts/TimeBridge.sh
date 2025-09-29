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
des='TimeBridge'

model_name=TimeBridge
auxi_mode=fft_ot
datasets=(ECL Traffic Weather PEMS03 PEMS08)



# hyper-parameters
dst=ECL

normalize=1
auxi_loss=None
ot_type=upper_bound
train_epochs=10
patience=3
test_batch_size=1
joint_forecast=1
distance=wasserstein_empirical_per_dim
eps=1e-9
reg_sk=0.005
mask_factor=0.0
alp=0.2
ca_layers=2
pd_layers=1
ia_layers=1
d_model=512
d_ff=512
n_heads=32
period=24
num_p=4
dropout=0.0
attn_dropout=0.1
stable_len=4
rerun=0


pl_list=(96 192 336 720)
# NOTE: ECL settings



for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.001 batch_size=16 lradj=type1 alpha=0.005 var_weight=0.002;;
        192) lr=0.001 batch_size=16 lradj=type1 alpha=0.05 var_weight=0.005;;
        336) lr=0.0002 batch_size=16 lradj=type1 alpha=0.001 var_weight=0.002;;
        720) lr=0.0002 batch_size=16 lradj=type1 alpha=0.002 var_weight=0.002;;
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
            --root_path $DATA_ROOT/electricity/ \
            --data_path electricity.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --factor 3 \
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
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --alpha $alp \
            --d_model $d_model \
            --d_ff $d_ff \
            --ca_layers $ca_layers \
            --pd_layers $pd_layers \
            --ia_layers $ia_layers \
            --num_p $num_p \
            --n_heads $n_heads \
            --period $period \
            --attn_dropout $attn_dropout \
            --stable_len $stable_len \
            --dropout $dropout \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






# hyper-parameters
dst=Traffic

normalize=1
auxi_loss=None
ot_type=upper_bound
train_epochs=100
patience=5
test_batch_size=1
joint_forecast=1
distance=wasserstein_empirical_per_dim
eps=1e-9
reg_sk=0.005
mask_factor=0.0
alp=0.35
ca_layers=3
pd_layers=1
ia_layers=1
d_model=512
d_ff=512
n_heads=64
period=24
num_p=8
dropout=0.0
attn_dropout=0.15
stable_len=2
rerun=0

pl_list=(96 192 336 720)
# NOTE: Traffic settings



for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.0005 batch_size=8 lradj=type1 alpha=0.005 var_weight=0.002;;
        192) lr=0.0005 batch_size=16 lradj=type1 alpha=0.01 var_weight=0.002;;
        336) lr=0.0005 batch_size=16 lradj=type1 alpha=0.01 var_weight=0.005;;
        720) lr=0.001 batch_size=16 lradj=type1 alpha=0.01 var_weight=0.01;;
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
            --root_path $DATA_ROOT/traffic/ \
            --data_path traffic.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 862 \
            --dec_in 862 \
            --c_out 862 \
            --factor 3 \
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
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --alpha $alp \
            --d_model $d_model \
            --d_ff $d_ff \
            --ca_layers $ca_layers \
            --pd_layers $pd_layers \
            --ia_layers $ia_layers \
            --num_p $num_p \
            --n_heads $n_heads \
            --period $period \
            --attn_dropout $attn_dropout \
            --stable_len $stable_len \
            --dropout $dropout \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






# hyper-parameters
dst=Weather

normalize=1
auxi_loss=None
ot_type=upper_bound
train_epochs=100
patience=15
test_batch_size=1
joint_forecast=1
distance=wasserstein_empirical_per_dim
eps=1e-9
reg_sk=0.005
mask_factor=0.0
alp=0.1
ca_layers=1
pd_layers=1
ia_layers=1
d_model=128
d_ff=128
n_heads=8
period=48
num_p=12
dropout=0.0
attn_dropout=0.15
stable_len=6
rerun=0

pl_list=(96 192 336 720)
# NOTE: Weather settings




for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96) lr=0.001 batch_size=64 lradj=type1 alpha=0.01 var_weight=0.005;;
        192) lr=0.001 batch_size=32 lradj=type1 alpha=0.015 var_weight=0.002;;
        336) lr=0.001 batch_size=32 lradj=type1 alpha=0.002 var_weight=0.002;;
        720) lr=0.0008 batch_size=64 lradj=type1 alpha=0.012 var_weight=0.002;;
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
            --root_path $DATA_ROOT/weather/ \
            --data_path weather.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 21 \
            --dec_in 21 \
            --c_out 21 \
            --factor 3 \
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
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --alpha $alp \
            --d_model $d_model \
            --d_ff $d_ff \
            --ca_layers $ca_layers \
            --pd_layers $pd_layers \
            --ia_layers $ia_layers \
            --num_p $num_p \
            --n_heads $n_heads \
            --period $period \
            --attn_dropout $attn_dropout \
            --stable_len $stable_len \
            --dropout $dropout \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






# hyper-parameters
dst=PEMS03

normalize=1
auxi_loss=None
ot_type=upper_bound
train_epochs=100
patience=15
test_batch_size=1
joint_forecast=1
distance=wasserstein_empirical_per_dim
eps=1e-9
reg_sk=0.005
mask_factor=0.0
alp=0.2
ca_layers=2
pd_layers=1
ia_layers=1
d_model=512
d_ff=512
n_heads=32
period=24
num_p=4
dropout=0.0
attn_dropout=0.1
stable_len=4
rerun=0

pl_list=(12 24 36 48)
# NOTE: PEMS03 settings




for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        12) lr=0.001 batch_size=32 lradj=type1 alpha=0.01 var_weight=0.01;;
        24) lr=0.001 batch_size=32 lradj=type1 alpha=0.01 var_weight=0.01;;
        36) lr=0.0008 batch_size=16 lradj=type1 alpha=0.005 var_weight=0.002;;
        48) lr=0.0008 batch_size=16 lradj=type1 alpha=0.01 var_weight=0.002;;
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
            --root_path $DATA_ROOT/PEMS/ \
            --data_path PEMS03.npz \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data PEMS \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 358 \
            --dec_in 358 \
            --c_out 358 \
            --factor 3 \
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
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --alpha $alp \
            --d_model $d_model \
            --d_ff $d_ff \
            --ca_layers $ca_layers \
            --pd_layers $pd_layers \
            --ia_layers $ia_layers \
            --num_p $num_p \
            --n_heads $n_heads \
            --period $period \
            --attn_dropout $attn_dropout \
            --stable_len $stable_len \
            --dropout $dropout \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done








# hyper-parameters
dst=PEMS08

normalize=1
auxi_loss=None
ot_type=upper_bound
train_epochs=100
patience=15
test_batch_size=1
joint_forecast=1
distance=wasserstein_empirical_per_dim
eps=1e-9
reg_sk=0.005
mask_factor=0.0
alp=0.05
ca_layers=1
pd_layers=1
ia_layers=1
d_model=128
d_ff=128
n_heads=8
period=48
num_p=12
dropout=0.0
attn_dropout=0.15
stable_len=6
rerun=0

pl_list=(12 24 36 48)
# NOTE: PEMS08 settings





for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        12) lr=0.002 batch_size=8 lradj=type1 alpha=0.005 var_weight=0.01;;
        24) lr=0.002 batch_size=8 lradj=type1 alpha=0.001 var_weight=0.01;;
        36) lr=0.002 batch_size=8 lradj=type1 alpha=0.001 var_weight=0.01;;
        48) lr=0.002 batch_size=8 lradj=type1 alpha=0.002 var_weight=0.02;;
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
            --root_path $DATA_ROOT/PEMS/ \
            --data_path PEMS08.npz \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data PEMS \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 170 \
            --dec_in 170 \
            --c_out 170 \
            --factor 3 \
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
            --joint_forecast ${joint_forecast} \
            --auxi_mode ${auxi_mode} \
            --ot_type ${ot_type} \
            --normalize ${normalize} \
            --distance ${distance} \
            --mask_factor ${mask_factor} \
            --reg_sk ${reg_sk} \
            --auxi_loss ${auxi_loss} \
            --eps ${eps} \
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --alpha $alp \
            --d_model $d_model \
            --d_ff $d_ff \
            --ca_layers $ca_layers \
            --pd_layers $pd_layers \
            --ia_layers $ia_layers \
            --num_p $num_p \
            --n_heads $n_heads \
            --period $period \
            --attn_dropout $attn_dropout \
            --stable_len $stable_len \
            --dropout $dropout \
            --var_weight $var_weight

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done






wait