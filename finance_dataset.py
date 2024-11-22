from torch.utils.data import Dataset
from torchvision.transforms import transforms
import pandas as pd
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

'''
columns = ['symbol', 'input_path', 'target_path', 'mode', 'up_or_down',
       'change_rate']
'''
class FinaceDataset(Dataset):
    def __init__(self,csv_path,mode='random',transform=None,img_size=256,train=True,all_mask=False) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.mode = mode
        self.df = None
        self.trasform = transform
        self.all_mask = all_mask

        self.init_csv()
        self.norm = lambda x : x/255
        # self.norm = lambda x : ((x/255)-0.5)/0.2
        self.tr = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.2),
            # A.OneOf([
            #     # A.HorizontalFlip(p=1),
            #     # A.VerticalFlip(p=1),
            #     # A.Rotate(limit=(90, 90), p=1),
            #     # A.Rotate(limit=(180, 180), p=1),
            #     # A.Rotate(limit=(270, 270), p=1),
            #     # A.NoOp(p=1)
            # ], p=0.0),
            # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            ToTensorV2()
        ],
        additional_targets={
            "additional_mask": "mask"
        })
        if not train:
            self.tr = A.Compose([
                A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
                ToTensorV2()
            ],
            additional_targets={
            "additional_mask": "mask"
        })


    def decode_value(self,x):
        # random mode up_or_down, change_rate
        return [float(_) for _ in x.split(';')]
    
    def init_csv(self):
        df = pd.read_csv(self.csv_path)
        self.df = df[df['mode']==self.mode]
        print(f'data load.\nlen : {self.df.__len__()}')

    def __getitem__(self, idx):
        df = self.df.iloc[idx, :]
        image = cv2.imread(df.input_path)[:, :, ::-1]  # BGR to RGB
        target_image = cv2.imread(df.target_path)[:, :, ::-1]
        mask = self.create_binary_mask(target_image)
        
        up_or_down = self.decode_value(df.up_or_down)
        change_rate = self.decode_value(df.change_rate)

        image = self.norm(image)
        target_image = self.norm(target_image)

        transformed = self.tr(image=image, mask=target_image,additional_mask=mask[...,np.newaxis])
        input_tensor = transformed['image']
        target_tensor = transformed['mask']
        target_tensor = target_tensor.permute(2, 0, 1)  # HWC to CHW
        
        mask_tensor = transformed['additional_mask'].permute(2,0,1).float()
        if not self.all_mask:
            mask_tensor[:,:,:192]= 0.
            mask_tensor[:,175:,:]= 0.

        target_tensor = target_tensor.float() 
        input_tensor = input_tensor.float()   

        data = {
            'image': input_tensor,
            'target_image': target_tensor,
            'mask':mask_tensor,
            # future works
            'up_or_down': torch.tensor(up_or_down, dtype=torch.long),
            'change_rate': torch.tensor(change_rate),
            # For analysis
            'mode': self.mode,
            'symbol': df.symbol
        }

        return data

    def __len__(self):
        return self.df.__len__()
    
    @staticmethod
    def create_binary_mask(target_image, start_column=191):
        hsv = cv2.cvtColor(target_image, cv2.COLOR_RGB2HSV)
        hsv = (hsv[...,1] > 220).astype(np.uint8)
        return hsv

if __name__ == '__main__':
    dt = FinaceDataset('data/candlestick_data_lnx.csv',mode='last')
    data = dt.__getitem__(0)
    print(data['image'].shape,data['mask'].shape)

