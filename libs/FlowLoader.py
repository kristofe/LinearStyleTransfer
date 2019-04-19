import os
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

def is_npz_file(filename):
    return any(filename.endswith(extension) for extension in [".npz", ".npy"])

class FlowDataset(data.Dataset):
    def __init__(self,dataPath,loadSize,fineSize,test=False,max_magnitude=10.0):
        super(FlowDataset,self).__init__()
        self.dataPath = dataPath
        self.field_list = [x for x in os.listdir(dataPath) if is_npz_file(x)]
        self.field_list = sorted(self.field_list)
        self.max_magnitude = max_magnitude 

        '''
        #These won't work.  They assume a PIL file.
        if not test:
            self.transform = transforms.Compose([
            		         transforms.Resize(fineSize),
            		         transforms.RandomCrop(fineSize),
                             transforms.RandomHorizontalFlip(),
            		         transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
            		         transforms.Resize(fineSize),
            		         transforms.ToTensor()])
        '''
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.test = test
        if len(self.field_list) == 0:
            print("DATASET ERROR: Found no flow files in {}".format(dataPath))

    def normalize(self, t):
        #put into 0-1 from an assumed range of [-max_magnitude, max_magnitude]
        t.clamp_(-self.max_magnitude, self.max_magnitude)
        t = t + self.max_magnitude
        t = t / (2.0 * self.max_magnitude)
        return t

    def denormalize(self, t):
        t = t * 2.0 * self.max_magnitude
        t = t - self.max_magnitude
        return t

    def __getitem__(self,index):
        dataPath = os.path.join(self.dataPath,self.field_list[index])

        with np.load(dataPath) as filedata:
            x_vel = filedata["vel_x"]
            y_vel = filedata["vel_y"]
            z_vel = filedata["vel_z"]
            blur_x_vel = filedata["blur_vel_x"]
            blur_y_vel = filedata["blur_vel_y"]
            blur_z_vel = filedata["blur_vel_z"]

        x_vel = torch.from_numpy(x_vel)
        y_vel = torch.from_numpy(y_vel)
        z_vel = torch.from_numpy(z_vel)
        blur_x_vel = torch.from_numpy(blur_x_vel)
        blur_y_vel = torch.from_numpy(blur_y_vel)
        blur_z_vel = torch.from_numpy(blur_z_vel)

        if z_vel.size(0) == 0:
            #2D field expand it out
            z_vel = z_vel.expand_as(x_vel)
            z_vel.fill_(0.0)

        vel = torch.stack((x_vel, y_vel, z_vel), dim=0)
        blur_vel = torch.stack((blur_x_vel, blur_y_vel, blur_z_vel), dim=0)

        vel = self.normalize(vel)
        blur_vel = self.normalize(blur_vel)

        fldName = self.field_list[index]
        fldName = fldName.split('.')[0]
        return vel, blur_vel, fldName

    def __len__(self):
        return len(self.field_list)
