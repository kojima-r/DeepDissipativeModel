ddm-opt --config config.yaml --data_train './dataset/020linear.train' \
	--data_test './dataset/020linear.train' \
	--study_name study020_naive --gpu 3
ddm-opt-post --study_name study020_naive

