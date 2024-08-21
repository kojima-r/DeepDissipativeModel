#ddm-dataset linear --prefix 000 --num 100  --train_num 90 --input_type_id 5 --without_normalization
#ddm-dataset linear --prefix 002 --num 100  --train_num 90 --input_type_id 5 --without_normalization --linear_x1
#ddm-dataset linear --prefix 003 --num 1000  --train_num 900 --input_type_id 5 --without_normalization --linear_x1

#ddm-dataset linear --prefix 010 --num 100  --train_num 90 --input_type_id 5 --without_normalization
#ddm-dataset linear --prefix 020 --num 1000  --train_num 900 --input_type_id 5 --without_normalization
#ddm-dataset linear --prefix 030 --num 10000  --train_num 9000 --input_type_id 5 --without_normalization


## random walk
ddm-dataset linear --prefix ex1 --num 11  --train_num 1 --input_type_id 2 --without_normalization
## step
#ddm-dataset linear --prefix ex2 --num 11  --train_num 1 --input_type_id 7 --without_normalization
