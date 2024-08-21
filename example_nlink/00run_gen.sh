
#ddm-dataset nlink --prefix 001 --num 100 --train_num 90
#ddm-dataset nlink --prefix 002 --num 1000 --train_num 900
#ddm-dataset nlink --prefix 003 --num 100 --train_num 90 --nlink_n 1 --nlink_q0 --T 5 --dh 0.05 --input_type_id 5 --nlink_old_input

#ddm-dataset nlink --prefix 003 --num 100 --train_num 90 --nlink_n 1 --nlink_q0 --T 10 --dh 0.05 --input_type_id 5 --without_normalization

ddm-dataset nlink --prefix 004 --num 100 --train_num 90 --nlink_n 1 --nlink_q0u0 --T 10 --dh 0.05 --input_type_id 5 --without_normalization
