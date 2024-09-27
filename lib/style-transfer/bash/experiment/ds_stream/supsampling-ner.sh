CUDA_VISIBLE_DEVICES=0 nohup python style_transfer/dstream_tasks/ner/train.py -m --config-name=gen-2 dataset.name=0.06-2-ofzh3aqu training_args.num_train_epochs=32 seed=0 &
CUDA_VISIBLE_DEVICES=1 nohup python style_transfer/dstream_tasks/ner/train.py -m --config-name=gen-2 dataset.name=0.06-2-ofzh3aqu training_args.num_train_epochs=32 seed=1 &
CUDA_VISIBLE_DEVICES=2 nohup python style_transfer/dstream_tasks/ner/train.py -m --config-name=gen-2 dataset.name=0.06-2-ofzh3aqu training_args.num_train_epochs=32 seed=2 &
CUDA_VISIBLE_DEVICES=3 nohup python style_transfer/dstream_tasks/ner/train.py -m --config-name=gen-2 dataset.name=0.06-2-ofzh3aqu training_args.num_train_epochs=32 seed=3,4 &
