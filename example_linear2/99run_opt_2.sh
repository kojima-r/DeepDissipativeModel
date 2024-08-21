ddm-opt --config config.yaml --data_train './dataset/020linear.train' \
	--data_test './dataset/020linear.train' \
	--study_name study020_dissipative --gpu 2 --dissipative_mode dissipative
ddm-opt-post --study_name study020_dissipative


