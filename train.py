'''
author : Youngmin Seo
'''
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Callback
from pytorch_lightning.strategies import DDPStrategy

from models.attn_unet import get_model
from finance_dataset import FinaceDataset
from loss import CombLossFn
from metric import ValMetric



parser = argparse.ArgumentParser(description='PyTorch Lightning Finance')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate ')
parser.add_argument('--save_folder', default='checkpoint', type=str, help='checkpoint save folder')
parser.add_argument('--log_dir', default='logs', type=str, help='tensorboard log folder')
parser.add_argument('--epoch', default=400, type=int, help='max epoch')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--gpus', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--finetune', type=bool, default=False, help='pre-train or fine-tuning [random,last]')
parser.add_argument('--optim', type=str, default='adam', help='opitmizer')
parser.add_argument('--dropout', type=str, default='dropblock', help='')
parser.add_argument('--drop_p', type=float, default=0.1, help='dropout Probability')
parser.add_argument('--loss', type=str, default='ssim', help='sup loss : ref losses folder')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='5e4')
parser.add_argument('--post_fix', type=str, default='', help='post fix of finetune checkpoint path')
parser.add_argument('--scheduler', type=str, default='cosineanealing', help='lr scheduler')
parser.add_argument('--img_size', type=int, default=256, help='input img_size')
parser.add_argument('--mixed_precision', type=bool, default=False, help='mixed_precision')
parser.add_argument('--act', type=str, default='', help='')
parser.add_argument('--arch', type=str, default='attn', help='')
parser.add_argument('--replace_type', type=str, default='instance', help='')
parser.add_argument('--monitor_metric', type=str, default='ssim', help='')
parser.add_argument('--train_mode', type=str, default='last', help='')
parser.add_argument('--all_mask', type=bool, default=False, help='')
parser.add_argument('--alpha', type=float, default=0.5, help='')


args = parser.parse_args()



class PLModule(LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.arch = args.arch
        self.model = get_model(3,3,args.dropout,args.drop_p,act=None,replace_mode='instance',v=args.arch)
        if args.finetune:
            self.model = init_weights_chkpt(self.model,args.save_folder)

        self.criterion = CombLossFn(args.loss,args.alpha,reg_loss=True if self.arch == 'attnv4' else False)
        self.metrics = ValMetric()
        self.num_workers = 4
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets, targets_mask = batch['image'],batch['target_image'],batch['mask']
        if self.arch != 'attnv4':
            outputs,outputs_mask = self(inputs)
            loss = self.criterion(outputs, targets, outputs_mask, targets_mask)
            metrics = self.metrics.compute_metrics(outputs, targets,outputs_mask, targets_mask)
        else:
            reg_target = batch['change_rate']
            outputs,outputs_mask,out_reg = self(inputs)
            loss = self.criterion(outputs, targets, outputs_mask, targets_mask,out_reg,reg_target)
            metrics = self.metrics.compute_metrics(outputs, targets,outputs_mask, targets_mask)
        # outputs = outputs.view(args.batch_size, -1)
        # targets = targets.view(args.batch_size, -1)
        # mae = torch.mean(torch.abs(outputs - targets))
        # loss = loss+mae

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_mae', metrics['mae'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_ssim', metrics['ssim'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_dice', metrics['dice'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, targets_mask = batch['image'],batch['target_image'],batch['mask']
        if self.arch != 'attnv4':
            outputs,outputs_mask = self(inputs)
            loss = self.criterion(outputs, targets, outputs_mask, targets_mask)
            metrics = self.metrics.compute_metrics(outputs, targets,outputs_mask, targets_mask)
        else:
            reg_target = batch['change_rate']
            outputs,outputs_mask,out_reg = self(inputs)
            loss = self.criterion(outputs, targets, outputs_mask, targets_mask,out_reg,reg_target)
            metrics = self.metrics.compute_metrics(outputs, targets,outputs_mask, targets_mask)
        self.log('val_ssim', metrics['ssim'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_mae', metrics['mae'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_dice', metrics['dice'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if args.optim == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr,weight_decay=args.weight_decay)
        elif args.optim == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
        else:
            raise NameError(f'not support {args.optim}')
        if args.scheduler == 'cosineanealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epoch,eta_min=1e-7)
        elif args.scheduler == 'cosineanealingwarmup':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2, eta_min=0.000001)
        else:
            raise NameError(f'not support yet {args.scheduler}')
        
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epoch,eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_mae"}

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset= FinaceDataset(csv_path='/workspace/fi_homework/data/splits/train.csv',
                                                mode=args.train_mode,
                                                train=True,
                                                all_mask=args.all_mask
                                                    )
            self.val_dataset = FinaceDataset(csv_path='/workspace/fi_homework/data/splits/val.csv',
                                                mode=args.train_mode,
                                                train=False,
                                                all_mask=args.all_mask
                                                    )
        def test_dt(dt):
            inp = dt.__getitem__(0)[0]
            assert args.in_ch == inp.size(0), f"{args.in_ch} is not matching {inp.size(0)}"
        # test_dt(self.train_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=self.num_workers)

def init_weights_chkpt(model, save_folder):
    # List all files in the save folder
    checkpoint_files = [f for f in os.listdir(save_folder) if f.endswith('.pth')]

    # Assuming you want to load the latest checkpoint based on the file naming scheme
    # Sort files based on the version (assuming the format of the files is consistent)
    # checkpoint_files.sort(key=lambda x: int(x.split('_')[3].split('.')[0]), reverse=True)

    # Construct full checkpoint path
    checkpoint_path = os.path.join(save_folder, checkpoint_files[0])
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint into the model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'])

    return model

class CustomCheckpoint(Callback):
    def __init__(self, checkpoint_dir, repeat_idx, metric_name='val_mae', mode='min',post_fix=''):
        super().__init__()
        self.checkpoint_dir = f'{checkpoint_dir}/{args.loss}' if args.post_fix == '' else f'{checkpoint_dir}/{args.loss}/{args.post_fix}'
        if args.finetune:
            self.checkpoint_dir = os.path.join(self.checkpoint_dir,'finetune',args.loss,post_fix)
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_model_file_name = ""
        self.repeat_idx = repeat_idx
        self.metric_name = metric_name
        self.mode = mode

    def on_validation_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            val_metric = trainer.callback_metrics.get(self.metric_name, None)
            if val_metric is not None:
                is_best = (self.mode == 'min' and val_metric <= self.best_metric) or \
                          (self.mode == 'max' and val_metric >= self.best_metric)
                if is_best:
                    self.best_metric = val_metric
                    state = {
                        'net': pl_module.model.state_dict(),
                        'epoch': trainer.current_epoch,
                        'mae': trainer.callback_metrics.get('val_mae', None),
                        'ssim': trainer.callback_metrics.get('val_ssim', None),
                    }
                    if self.best_model_file_name:
                        old_path = os.path.join(self.checkpoint_dir, self.best_model_file_name)
                        os.remove(old_path)
                    
                    self.best_model_file_name = f'{self.repeat_idx}_{trainer.current_epoch}_{val_metric:.4f}.pth'
                    new_path = os.path.join(self.checkpoint_dir, self.best_model_file_name)
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    torch.save(state, new_path)
                    print(f'Saved new best model to: {new_path}')

def make_logdir():
    
    if args.finetune:           
        pre_train_loss = args.save_folder.split('/')[-1]
        logdir = os.path.join(args.log_dir,f'{args.arch}/{pre_train_loss}')
        logdir = os.path.join(logdir,f'finetune/{args.loss}')
    else:
        logdir = os.path.join(args.log_dir,f'{args.arch}/{args.loss}')
    
    logdir = f'{logdir}' if args.post_fix =='' else f'{logdir}/{args.post_fix}'

    return logdir

def main(i):
    model = PLModule(lr=args.lr)
    logdir = make_logdir()
    logger = TensorBoardLogger(save_dir=logdir, name=f'repeat_{i}')
    
    if args.monitor_metric == 'mae':
        monitor_metric = 'val_mae'  
    elif args.monitor_metric == 'ssim':
        monitor_metric ='val_ssim'
    elif args.monitor_metric == 'dice':
        monitor_metric ='val_dice'
    if monitor_metric in ['val_ssim','val_dice']:
        mode = 'max'
    else: mode = 'min'

    print(monitor_metric,'mode : ', )
    checkpoint_callback = CustomCheckpoint(
        checkpoint_dir=f'{args.save_folder}/{args.arch}' if not args.finetune else args.arch,
        repeat_idx=i,
        metric_name=monitor_metric, 
        mode = mode,
        post_fix=args.post_fix
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        max_epochs=args.epoch,
        devices=args.gpus,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        strategy=DDPStrategy(find_unused_parameters=False) if args.gpus > 1 else 'auto',
        enable_checkpointing=False,
        precision='16-mixed' if args.mixed_precision else '32-true'
    )

    trainer.fit(model)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main(0)