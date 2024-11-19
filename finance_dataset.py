from torch.utils.data import Dataset
from torchvision.transforms import transforms
import pandas as pd
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

'''
columns = ['symbol', 'input_path', 'target_path', 'mode', 'up_or_down',
       'change_rate']
'''
class FinaceDataset(Dataset):
    def __init__(self,csv_path,mode='random',transform=None,img_size=256) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.mode = mode
        self.df = None
        self.trasform = transform
        
        self.init_csv()
        self.norm = lambda x : x/255
        # self.norm = lambda x : ((x/255)-0.5)/0.2
        self.tr = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.Rotate(limit=(90, 90), p=1),
                A.Rotate(limit=(180, 180), p=1),
                A.Rotate(limit=(270, 270), p=1),
                A.NoOp(p=1)  # 아무 변환도 적용하지 않을 확률
            ], p=0.5),   # OneOf 내의 변환 중 하나를 선택
            A.RandomBrightnessContrast(p=0.2),  # 밝기 및 대비 조정
            A.GaussianBlur(blur_limit=(3,7), p=0.2),  # 가우시안 블러
            # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            ToTensorV2()
        ])
    def decode_value(self,x):
        # random mode up_or_down, change_rate
        return [float(_) for _ in x.split(';')]
    
    def init_csv(self):
        df = pd.read_csv(self.csv_path)
        self.df = df[df['mode']==self.mode]
        print(f'data load.\nlen : {self.df.__len__()}')

    def __getitem__(self, idx) :
        df = self.df.iloc[idx,:]
        image = cv2.imread(df.input_path)[:,:,::-1]# bgr 2 rgb
        target_image = cv2.imread(df.target_path)[:,:,::-1]
        up_or_down = self.decode_value(df.up_or_down)
        change_rate = self.decode_value(df.change_rate)

        transformed = self.tr(image=image, mask=target_image)
        input_tensor = transformed['image']
        target_tensor = transformed['mask']
        target_tensor = target_tensor.permute(2,0,1)
        target_tensor = self.norm(target_tensor)

        data = {
            'image':input_tensor,
            'target_image':target_tensor,
            'up_or_down':torch.tensor(up_or_down,dtype=torch.long),
            'change_rate':torch.tensor(change_rate),
            # for analyze
            'mode':self.mode,
            'symbol':df.symbol
        }

        return data

    def __len__(self):
        return self.df.__len__()
    
if __name__ == '__main__':
    dt = FinaceDataset('data/candlestick_data_lnx.csv',mode='last')
    data = dt.__getitem__(0)
    print(data['image'].shape,data['target_image'].shape)

