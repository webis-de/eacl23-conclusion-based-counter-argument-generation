# Training the model:

CUDA_VISIBLE_DEVICES=1  python train_baseline_1.py --output_dir ../multitask-counter-arg-generation/data/output/ca-final-models/mt-v4.baseline_1/ --eval_steps 500 --train_batch_size=8 > baseline_1.log

CUDA_VISIBLE_DEVICES=0 python train_baseline_2.py --output_dir ../multitask-counter-arg-generation/data/output/ca-final-models/mt-v4.baseline_2/ --eval_steps 500 --train_batch_size=8 > baseline_2.log


### Fine tuning parameters
CUDA_VISIBLE_DEVICES=0 python fine_tune_parameters_baseline_2.py --output_dir ../multitask-counter-arg-generation/data/output/ca-final-models/mt-v4.baseline_2_fine_tune/ --eval_steps 200 --train_batch_size 8 --train_size 5000 --eval_size 1000 --num_epoch 1

CUDA_VISIBLE_DEVICES=0 python fine_tune_parameters_baseline_1.py --output_dir ../data/output/ca-final-models/mt-v4.baseline_1_fine_tune/ --eval_steps 200 --train_batch_size 8 --train_size 5000 --eval_size 1000 --num_epoch 3


CUDA_VISIBLE_DEVICES=0 python train_baseline_2.py --output_dir ../data/output/ca-final-models/mt-v4.baseline_2/ --eval_steps 1000 --train_batch_size=4 --valid_batch_size=4 --max_input_length 256 --max_argument_target 256 --max_claim_target 32 > baseline_2.log


CUDA_VISIBLE_DEVICES=0  python train_baseline_1.py --output_dir ../data/output/ca-final-models/mt-v4.baseline_1/ --eval_steps 500 --train_batch_size 4 --valid_batch_size 4 --max_input_length 512 --max_claim_target 64 --max_argument_target 256 --learning_rate 3e-5 > baseline_1.log