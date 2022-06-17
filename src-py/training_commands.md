### Training model on Reddit dataset:

#### Training basic model with argument conclusion in the input:

CUDA_VISIBLE_DEVICES=0 python training_conclusion_and_ca_generation.py --train_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/preprocessed_train_conclusion_all.pkl --valid_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/sample_valid_conclusion_all.pkl --output_dir /var/tmp/sile2804/ca-final-models/known-conc-model --train_bs=8 --valid_bs=8 --train_epochs=6 --premises_clm post --conclusion_clm title --counter_clm counter --max_source_length 512 --max_target_length 200 --unique_targets

#### Training basic model with no conclusion in the input:

CUDA_VISIBLE_DEVICES=0 python training_conclusion_and_ca_generation.py --train_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/preprocessed_train_conclusion_all.pkl --valid_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/sample_valid_conclusion_all.pkl --output_dir /var/tmp/sile2804/ca-final-models/masked-conc-model --train_bs=8 --valid_bs=8 --train_epochs=6 --premises_clm post --conclusion_clm title --counter_clm counter --max_source_length 512 --max_target_length 200 --unique_targets --masked_conclusion


#### Training a model to generate the argument's conclusion and then counter:

CUDA_VISIBLE_DEVICES=0 python training_conclusion_and_ca_generation.py --train_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/preprocessed_train_conclusion_all.pkl --valid_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/sample_valid_conclusion_all.pkl --output_dir /var/tmp/sile2804/ca-final-models/pred-conc-model --train_bs=8 --valid_bs=8 --train_epochs=6 --premises_clm post --conclusion_clm title --counter_clm counter --conclusion_and_counter_generation --max_source_length 512 --max_target_length 512 --unique_targets


#### Training a model to generate counter's conclusion as a planning step and then counter:

CUDA_VISIBLE_DEVICES=0 python training_conclusion_and_ca_generation.py --train_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/preprocessed_train_conclusion_all.pkl --valid_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/sample_valid_conclusion_all.pkl --output_dir /var/tmp/sile2804/ca-final-models/planning-model --train_bs=8 --valid_bs=8 --train_epochs=10 --premises_clm post --conclusion_clm counter_conclusion --counter_clm counter --counter_conclusion_clm counter_conclusion --conclusion_and_counter_conclusion_in_generation --max_source_length 512 --max_target_length 512 --unique_targets


#### Training basic model to generate conclusions:

CUDA_VISIBLE_DEVICES=0 python training_conclusion_and_ca_generation.py --train_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/preprocessed_train_conclusion_all.pkl --valid_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/sample_valid_conclusion_all.pkl --output_dir /var/tmp/sile2804/conc-gen-model --train_bs=8 --valid_bs=8 --train_epochs=6 --premises_clm post --counter_clm title --max_source_length 512 --max_target_length 200 --unique_targets --masked_conclusion