python train.py -m model.optimizer.lr=2e-5,3e-5,5e-5 \
                   datamodule.batch_size=8,16,32 \
                   datamodule.c_fold=0,1,2,3,4 \
                   datamodule.datasets=["webanno797399823581181383export.zip","webanno797399823581181383export-without-deft2021.zip"],["webanno797399823581181383export-without-deft2021-and-label-studio-export.zip"] \
                   trainer=gpu \
                   trainer.max_epochs=30 \
                   logger=wandb
