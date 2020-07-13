import os.path
from data.base_dataset import BaseDataset, get_transform
#from data.image_folder import make_dataset
from data.spec_folder import make_dataset
from PIL import Image
import numpy as np
import random

def default_spec_adjust(spec):
    # mean = np.mean(spec)
    # std = np.std(spec)
    # spec = (spec - mean)/std
    return np.pad(spec,[(0,0),(0,int(4*np.ceil(spec.shape[1]/4))-spec.shape[1]),(0,int(4*np.ceil(spec.shape[2]/4))-spec.shape[2])],'constant')
    # return np.pad(spec,[(0,0),(0,int(4*np.ceil(spec.shape[0]/4))-spec.shape[0]),(0,int(4*np.ceil(spec.shape[1]/4))-spec.shape[1])],'constant')
    #return np.pad(spec,[(0,4-spec.shape[0]%4),(0,4-spec.shape[1]%4)],'constant')

class UnalignedSpecDataset(BaseDataset):
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     return parser

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        #self.transform = get_transform(opt)
        self.transform = default_spec_adjust

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        #A_img = Image.open(A_path).convert('RGB')
        #B_img = Image.open(B_path).convert('RGB')
        A_spec = np.load(A_path)
        B_spec = np.load(B_path)

        
        A = self.transform(A_spec)
        B = self.transform(B_spec)
        # print(f'A shape {A.shape}, max {A.max()}, min {A.min()}, max {B.max()}, min {B.min()}')
        # if A.max() != 1 or A.min() != -1:
        #     print("nan file: ",A_path)
        # A = np.expand_dims(A,axis=0)
        # B = np.expand_dims(B,axis=0)

        """
        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        """
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
