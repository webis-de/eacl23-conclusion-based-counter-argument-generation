# Conclusion-based Counter Argument Generation:

## Data Preparation:

The code in src-ipynb/data_prep.ipynb contains data preparation for the training split including removing the pot comments, and selecting for each post one comment that has the highest argumentative score according to the work of Gretz et al 2019.

## Training:

### Training the basline:

To train the baseline model. Execute the following command under the src-py folder:

``CUDA_VISIBLE_DEVICES=0 python training_conclusion_and_ca_generation.py --train_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/preprocessed_train_conclusion_all.pkl --valid_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/sample_valid_conclusion_all.pkl --output_dir /var/tmp/sile2804/ca-final-models/known-conc-model --train_bs=8 --valid_bs=8 --train_epochs=6 --premises_clm post --conclusion_clm title --counter_clm counter --max_source_length 512 --max_target_length 200 --unique_targets``

- To train the conclusion generation model, execute the following command under the src-py folder:

`` CUDA_VISIBLE_DEVICES=0 python training_conclusion_and_ca_generation.py --train_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/preprocessed_train_conclusion_all.pkl --valid_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/sample_valid_conclusion_all.pkl --output_dir /var/tmp/sile2804/conc-gen-model --train_bs=8 --valid_bs=8 --train_epochs=6 --premises_clm post --counter_clm title --max_source_length 512 --max_target_length 200 --unique_targets --masked_conclusion``

- To generate conclusions for the posts for the pipeline-based approach, execute the code in ``notebooks/bart-generate-conclusions.ipynb``


### Training approach:

#### Fully Shared Encoder and Decoder:

To train the approach on generating the conclusion and counter in one sequence, execute the following command under the src-py folder:

``CUDA_VISIBLE_DEVICES=0 python training_conclusion_and_ca_generation.py --train_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/preprocessed_train_conclusion_all.pkl --valid_data ../../../data-ceph/arguana/arg-generation/multi-taks-counter-argument-generation/reddit_data/conclusion_and_ca_generation/sample_valid_conclusion_all.pkl --output_dir /var/tmp/sile2804/ca-final-models/pred-conc-model --train_bs=8 --valid_bs=8 --train_epochs=6 --premises_clm post --conclusion_clm title --counter_clm counter --conclusion_and_counter_generation --max_source_length 512 --max_target_length 512 --unique_targets
``

#### Joint Encoder and Two Decoders:

To train the model, run the following command under joint-model-two-decoders folder:

``CUDA_VISIBLE_DEVICES=0 python train.py --output_dir ../data/output/ca-final-models/mt-v4.baseline_2/ --eval_steps 1000 --train_batch_size=4 --valid_batch_size=4 --max_input_length 256 --max_argument_target 256 --max_claim_target 32 > logging.log
``

For fine-tuning the alpha1 and alpha2 parameters, run the following command:

``CUDA_VISIBLE_DEVICES=0 python fine_tune_parameters.py --output_dir ../multitask-counter-arg-generation/data/output/ca-final-models/mt-v4.fine_tune/ --eval_steps 200 --train_batch_size 8 --train_size 5000 --eval_size 1000 --num_epoch 1``

## Inference:

### Baselines:

- To generate the baselines predictions, follow the instructions in the ``src-ipynb/baseline_predictions.ipynb``. 

### Fully shared encoder decoder:

- To generate predictions for the model, follow instructions in the ``prompted-conclusion/jointly-prompted-conclusion-generation.ipynb`` notebook.

- To generate predictions for the pipeline-based model with the stance-based ranking component, follow the instructions in ``prompted-conclusion/pipelined-prompted-conclusion-generation.ipynb``

### Joint Encoder and Two Decoders:

- To generate the predictions from this model, follow the instructions in ``joint-model-two-decoders/predicting_counters.ipynb`` notebook

## Training and Evaluating the stance classifier:

Code for training and evaluating the stance classifier on Kialo dataset can be found in ``notebooks/stance-classification.ipynb``

## Training and extracting targets from the dataset:

To train and extract targets from the conclusions, follow the instructions in ``notebooks/claim-target-extraction.ipynb``