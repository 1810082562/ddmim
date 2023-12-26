import os 
# from classifier import Classifier
# from models.ddmimv8 import DDMIMV8
# from models.ddmimv7 import DDMIMV7
# from models.ddmimv6 import DDMIMV6
# from models.ddmim import DDMIM
# from models.ddmimv2 import DDMIMV2
# from models.ddmimv4 import DDMIMV4
# from models.ddmimv5 import DDMIMV5
from models.lightning_mae import Lightning_mae
from models.ddmimv9 import DDMIMV9
import pytorch_lightning as pl 
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.datamodules import ImagenetDataModule
from config.option import parse_args

from models.simmimforddmimv4 import SimMimForDDMIMV4

def main(args):

    pl.seed_everything(args.seed)
    

    '''
    model=DDMIM(
        image_size=224,
        patch_size=32,
        dim=1024,
        stages_depth=[1,1,3,1],
        heads=8,
        mlp_dim=2048,
        channels=3,
        dim_head=128,
        masking_ratio=0.5,
        stagefactor=[1e-4,1e-3,0.01,0.1],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        )
    
    model=DDMIMV2(
        image_size=224,
        patch_size=16,
        tsfm_dim=768,
        stages_depth=[2,2,6,2],
        heads=12,
        mlp_dim=3096,
        channels=3,
        dim_head=64,
        masking_ratio=0.75,
        stagefactor=[1e-3,1e-2,0.1,1],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        tau=0.3,
        alpha=1
        )
    '''
    # model=SimMimForDDMIMV4(
    #     masking_ratio=0.75
    # )
    # model=DDMIMV6(
    #     # num_classes=2048,
    #     masking_ratio=[0.5,0.5,0.5,0.5],
    #     stagefactor=[1.0,1.0,1.0,1.0],
    #     # tau=0.1,
    #     #alpha=0.5,
    # )
    # model=DDMIMV8(
    #     # num_classes=2048,
    #     masking_ratio=[0.4,0.4,0.5,0.4],
    #     stagefactor=[1.0,1.0,1.0,1.0],
    #     tau=0.1,
    #     #alpha=0.5,
    # )
    # model=DDMIMV4(
    #     # stagefactor=[0.,0.,0.,0.]
    #     # alpha=0,
    #     lr=args.lr,
    # )
    # model=Lightning_mae()
    model=DDMIMV9()
    # model=Classifier(resume=args.resume)
    
    datamodule=ImagenetDataModule(args.data,num_workers=8,batch_size=args.batch_size)
    
    os.makedirs(args.log_dir,exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_last=True,
        # every_n_train_steps=5000,
        filename='{epoch}_{val_loss:.2f}_'+model._get_name()
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks=[checkpoint_callback,lr_callback]

    logger=pl.loggers.TensorBoardLogger(
        args.log_dir,
        name=f'seed{args.seed}',
    )
    
    trainer=pl.Trainer(
        resume_from_checkpoint=args.resume,
        #stagefactor=[1e-3,1e-2,0.1,1],
        gpus=args.gpu,
        # gpus=0,
        #amp_backend='apex',
        # precision=16,
        #auto_scale_batch_size=True,
        max_epochs=args.epochs if args.epochs else 1000,
        #max_steps=None,
        callbacks=callbacks,
        logger=logger,
        #log_every_n_steps=10,
        # limit_train_batches=100, 
        
        # limit_val_batches=20,
        
    )
    
    if not args.evaluate:
        trainer.fit(model=model,datamodule=datamodule)
    else:
        trainer.test(model=model,datamodule=datamodule)
    



if __name__ == '__main__':
    
    args= parse_args()
 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #print(parser.parse_args().gpu)
    main(args)