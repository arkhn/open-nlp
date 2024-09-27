CUDA_VISIBLE_DEVICES=0 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=icd/20 dataset.name=0.06-2-ofzh3aqu seed=0,1,2 training_args.learning_rate=5e-5 training_args.num_train_epochs=32 &
CUDA_VISIBLE_DEVICES=0 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=icd/20 dataset.name=supsampling seed=0,1,2 training_args.learning_rate=5e-5 training_args.num_train_epochs=32 &

CUDA_VISIBLE_DEVICES=1 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=icd/50 dataset.name=0.06-2-ofzh3aqu seed=0,1,2 training_args.learning_rate=5e-5 training_args.num_train_epochs=32 &
CUDA_VISIBLE_DEVICES=1 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=icd/50 dataset.name=supsampling seed=0,1,2 training_args.learning_rate=5e-5 training_args.num_train_epochs=32 &

CUDA_VISIBLE_DEVICES=2 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=icd/100 dataset.name=0.06-2-ofzh3aqu seed=0,1,2 training_args.learning_rate=5e-5 training_args.num_train_epochs=32 &
CUDA_VISIBLE_DEVICES=2 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=icd/100 dataset.name=supsampling seed=0,1,2 training_args.learning_rate=5e-5 training_args.num_train_epochs=32 &

CUDA_VISIBLE_DEVICES=3 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=icd/400 dataset.name=0.06-2-ofzh3aqu seed=0,1,2 training_args.learning_rate=5e-5 training_args.num_train_epochs=32 &
CUDA_VISIBLE_DEVICES=3 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=icd/400 dataset.name=supsampling seed=0,1,2 training_args.learning_rate=5e-5 training_args.num_train_epochs=32 &
