CUDA_VISIBLE_DEVICES=0 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=combined_icd/20 dataset.name=combined-4 &
CUDA_VISIBLE_DEVICES=1 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=combined_icd/50 dataset.name=combined-4 &
CUDA_VISIBLE_DEVICES=2 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=combined_icd/100 dataset.name=combined-4 &
CUDA_VISIBLE_DEVICES=3 nohup python style_transfer/dstream_tasks/icd/train.py -m --config-name=combined_icd/400 dataset.name=combined-4 &
