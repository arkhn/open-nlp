CUDA_VISIBLE_DEVICES=0 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=ablation_icd/20 dataset.name=0.04-2-mru97w7c &
CUDA_VISIBLE_DEVICES=1 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=ablation_icd/40 dataset.name=0.04-2-mru97w7c &
CUDA_VISIBLE_DEVICES=2 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=ablation_icd/60 dataset.name=0.04-2-mru97w7c &
CUDA_VISIBLE_DEVICES=3 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=ablation_icd/80 dataset.name=0.04-2-mru97w7c &
