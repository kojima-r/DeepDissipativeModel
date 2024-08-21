seed=$1
dataset=$2
mode=$3

ddm-opt --config config.yaml --data_train ./dataset/${dataset}nlink.train \
	--study_name study${dataset}_${mode}_${seed} \
    --dissipative_mode ${mode} --seed ${seed}
ddm-opt-post --study_name study${dataset}_${mode}_${seed}
ddm-train --config config_study${dataset}_${mode}_${seed}.retrain.yaml --epoch 5000
#--resume ./result1/model/model.4930.checkpoint

