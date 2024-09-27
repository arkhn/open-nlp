CUDA_VISIBLE_DEVICES=0 nohup python style_transfer/dstream_tasks/icd/train.py -m \
                                                --config-name=icd/100 dataset.percentile=70 \
                                                dataset.name=combined-4 seed=0 \
                                                training_args.learning_rate=5e-5,2e-5 \
                                                dataset.random_sampling=true \
                                                training_args.num_train_epochs=32 &
