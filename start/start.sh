#!/usr/bin/env bash
export DATA_PATH=/apdcephfs/private_jiyingzhang/CCA-SSG
export OUT_PATH=/apdcephfs/share_1364275/jiyingzhang/taiji_results/CCA-SSG/
log_dir=${OUT_PATH}/${TJ_TASK_ID}/${METRICS_TRIAL_NAME}
ln -s ${DATA_PATH}/data data

mkdir -p $log_dir
data_path='/apdcephfs/private_jiyingzhang/CCA-SSG'
save_file='/apdcephfs/share_1364275/jiyingzhang/taiji_results/CCA-SSG/automl_results.csv'


echo $log_dir
echo $tag

echo 'start python command!' >> $log_dir/train_out.log

python ${data_path}/main.py --dataname cora --epochs 50 --lambd 1e-3 --dfr 0.1 --der 0.4 --lr2 1e-2 --wd2 1e-4 >> $log_dir/train_out.log

echo 'finished python command' >> $log_dir/train_out.log
