from dataset.ZeroEGGS import ZEGGS as ZeroEGGS 
from torch.utils.data import Dataset, DataLoader 
from dataset.ZeroEGGS.ZEGGS import dataset 

import torch 
import os 
import io 


class CustomDataset(Dataset): 
    def __init__(self, root_dir, transform=None): 
        self.root_dir = root_dir 
        self.transform = transform 
        self.dataset = dataset(root_dir) 
        
        
    def __len__(self): 
        return len(self.dataset) 
    
    
    def __getitem__(self, idx): 
        if torch.is_tensor(idx): 
            idx = idx.tolist() 
        img_name = os.path.join(self.root_dir, self.dataset[idx][0]) 
        image = io.imread(img_name) 
        label = self.dataset[idx][1] 
        sample = {'image': image, 'label': label} 
        if self.transform: 
            sample = self.transform(sample) 
        return sample
    


