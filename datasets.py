import torch
import numpy as np
from torch.utils.data import Dataset
import glob
import os
from torch.utils.data.dataloader import DataLoader


class TrainDataset(Dataset):
    def __init__(self, LR_phiG_folder, HR_phi_folder):
        # super(TrainDataset, self).__init__()
        self.LR_phiG_list = glob.glob(LR_phiG_folder+'*')
        self.HR_phi_folder = HR_phi_folder
        
        self.data_len = len(self.LR_phiG_list)
        
        
    def __getitem__(self, idx):
        single_LR_phiG_path = self.LR_phiG_list[idx]
        LR_phiG_as_np = np.load(single_LR_phiG_path)
        LR_phiG_as_ten = torch.from_numpy(LR_phiG_as_np).float()
        
        file_name = os.path.split(single_LR_phiG_path)[1]
        
        single_HR_phi_path = self.HR_phi_folder + file_name
        HR_phi_as_np = np.load(single_HR_phi_path)
        HR_phi_as_np = HR_phi_as_np/(500/(2*np.pi))
        HR_phi_as_np = np.expand_dims(HR_phi_as_np, 0)
        HR_phi_as_ten = torch.from_numpy(HR_phi_as_np).float()
        
        return LR_phiG_as_ten, HR_phi_as_ten
        
    def __len__(self):
        return self.data_len


class EvalDataset(Dataset):
    def __init__(self, LR_phiG_folder, HR_phi_folder):
        # super(TrainDataset, self).__init__()
        self.LR_phiG_list = glob.glob(LR_phiG_folder+'*')
        self.HR_phi_folder = HR_phi_folder
        
        self.data_len = len(self.LR_phiG_list)
        
        
    def __getitem__(self, idx):
        single_LR_phiG_path = self.LR_phiG_list[idx]
        LR_phiG_as_np = np.load(single_LR_phiG_path)
        LR_phiG_as_ten = torch.from_numpy(LR_phiG_as_np).float()
        
        file_name = os.path.split(single_LR_phiG_path)[1]
        
        single_HR_phi_path = self.HR_phi_folder + file_name
        HR_phi_as_np = np.load(single_HR_phi_path)
        HR_phi_as_np = HR_phi_as_np/(500/(2*np.pi))
        HR_phi_as_np = np.expand_dims(HR_phi_as_np, 0)
        HR_phi_as_ten = torch.from_numpy(HR_phi_as_np).float()
        
        return LR_phiG_as_ten, HR_phi_as_ten
        
    def __len__(self):
        return self.data_len

# if __name__ == "__main__":
#     LR_phiG_folder = '/home/vicky/soapy/Dataset_254x254/Train/LR_phigrad/'
#     HR_phi_folder = '/home/vicky/soapy/Dataset_254x254/Train/HR_phi/'
#     train_dataset = TrainDataset(LR_phiG_folder, HR_phi_folder)
#     train_dataloader = DataLoader(dataset=train_dataset,
#                                   batch_size=2,
#                                   shuffle=True,
#                                   num_workers=2,
#                                   pin_memory=True)

    
#     for images, labels in train_dataloader:
#         break
