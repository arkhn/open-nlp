CUDA_VISIBLE_DEVICES=0 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=icd/50 dataset.name=supsampling seed=0 &
CUDA_VISIBLE_DEVICES=1 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=icd/50 dataset.name=supsampling seed=1 &
CUDA_VISIBLE_DEVICES=2 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=icd/50 dataset.name=supsampling seed=2 &
CUDA_VISIBLE_DEVICES=3 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=icd/50 dataset.name=supsampling seed=3,4 &
