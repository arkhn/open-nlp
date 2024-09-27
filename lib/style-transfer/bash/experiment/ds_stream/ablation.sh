CUDA_VISIBLE_DEVICES=1 nohup python style_transfer/dstream_tasks/ner/train.py -m --config-name=ablation_ner/40 &
CUDA_VISIBLE_DEVICES=2 nohup python style_transfer/dstream_tasks/ner/train.py -m --config-name=ablation_ner/60 &
CUDA_VISIBLE_DEVICES=3 nohup python style_transfer/dstream_tasks/ner/train.py -m --config-name=ablation_ner/80 &
