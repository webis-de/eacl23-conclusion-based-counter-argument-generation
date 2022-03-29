### Training model on Reddit dataset:

#### Baseline version of Jointly generating conclusion then counter:
CUDA_VISIBLE_DEVICES=1 python training_conclusion_and_ca_generation.py --train_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/train_conclusion_comp_remove_75sem_perc.pkl --valid_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/valid_conclusion_comp_remove_75sem_perc_sample.pkl --output_dir ../data/output/pred-conclusion-bart-model --train_bs=32 --valid_bs=32 --train_epochs=6 --premises_clm masked_premises --conclusion_clm title --counter_clm counter --conclusion_and_counter_generation --max_source_length 512 --max_target_length 512

##### Fine-tuning:
CUDA_VISIBLE_DEVICES=0 python training_conclusion_and_ca_generation.py --train_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/train_conclusion_comp_remove_75sem_perc.pkl --valid_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/valid_conclusion_comp_remove_75sem_perc_sample.pkl --output_dir ../data/output/valid-ft/pred-conc --train_bs=16 --valid_bs=16 --train_epochs=3 --premises_clm masked_premises --conclusion_clm title --counter_clm counter --conclusion_and_counter_generation --max_source_length 512 --max_target_length 300 --logging_dir /var/tmp/sile280/multitask-counter-arg-generation/valid-ft-all/pred-conc --fine_tune


#### Baseline version of Jointly generating conclusion then counter Trained on all data:
CUDA_VISIBLE_DEVICES=0 python training_conclusion_and_ca_generation.py --train_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/train_conclusion_all.pkl --valid_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/valid_conclusion_all_sample.pkl --output_dir ../data/output/pred-conclusion-bart-model-on-all-data --train_bs=32 --valid_bs=32 --train_epochs=6 --premises_clm post --conclusion_clm title --counter_clm counter --conclusion_and_counter_generation --max_source_length 512 --max_target_length 512


#### Baseline trained on data with known conclusion:
CUDA_VISIBLE_DEVICES=0 python training_conclusion_and_ca_generation.py --train_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/train_conclusion_comp_remove_75sem_perc.pkl --valid_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/valid_conclusion_comp_remove_75sem_perc_sample.pkl --output_dir ../data/output/known-conclusion-bart-model --train_bs=32 --valid_bs=32 --train_epochs=6 --premises_clm masked_premises --conclusion_clm title --counter_clm counter --max_source_length 512 --max_target_length 200

##### Fine-tuning:
CUDA_VISIBLE_DEVICES=1 python training_conclusion_and_ca_generation.py --train_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/train_conclusion_comp_remove_75sem_perc.pkl --valid_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/valid_conclusion_comp_remove_75sem_perc_sample.pkl --output_dir ../data/output/valid-ft/known-conc --train_bs=16 --valid_bs=16 --train_epochs=3 --premises_clm masked_premises --conclusion_clm title --counter_clm counter --max_source_length 512 --max_target_length 200 --logging_dir /var/tmp/sile280/multitask-counter-arg-generation/valid-ft-all/known-conc --fine_tune


#### Baseline trained on data with known conclusion. Trained on all data:
CUDA_VISIBLE_DEVICES=1 python training_conclusion_and_ca_generation.py --train_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/train_conclusion_all.pkl --valid_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/valid_conclusion_all_sample.pkl --output_dir ../data/output/known-conclusion-bart-model-on-all-data --train_bs=32 --valid_bs=32 --train_epochs=6 --premises_clm post --conclusion_clm title --counter_clm counter --max_source_length 512 --max_target_length 200




### Training model on Kialo dataset:

#### Baseline trained on data with known conclusion:
CUDA_VISIBLE_DEVICES=0 python training_conclusion_and_ca_generation.py --train_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/kialo_data/kialo_train_df.pkl --valid_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/kialo_data/kialo_valid_df.pkl --output_dir ../data/output/known-conclusion-kialo-model --downsample_valid=0.1 --train_bs=16 --valid_bs=16 --train_epochs=3 --premises_clm premises --conclusion_clm conclusion_text --counter_clm counter --max_source_length 200 --max_target_length 80

#### Baseline trained on data with masked conclusion:
CUDA_VISIBLE_DEVICES=1 python training_conclusion_and_ca_generation.py --train_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/kialo_data/kialo_train_df.pkl --valid_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/kialo_data/kialo_valid_df.pkl --output_dir ../data/output/masked-conclusion-kialo-model --downsample_valid=0.1 --train_bs=16 --valid_bs=16 --train_epochs=3 --premises_clm premises --conclusion_clm conclusion_text --counter_clm counter --masked_conclusion --max_source_length 200 --max_target_length 80


