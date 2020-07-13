from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
from data.spec_folder import make_dataset
from PIL import Image
import numpy as np

def default_spec_adjust(spec):
    # mean = np.mean(spec)
    # std = np.std(spec)
    # spec = (spec - mean)/std
    return np.pad(spec,[(0,0),(0,int(4*np.ceil(spec.shape[1]/4))-spec.shape[1]),(0,int(4*np.ceil(spec.shape[2]/4))-spec.shape[2])],'constant')
    # return np.pad(spec,[(0,0),(0,int(4*np.ceil(spec.shape[0]/4))-spec.shape[0]),(0,int(4*np.ceil(spec.shape[1]/4))-spec.shape[1])],'constant')
    #return np.pad(spec,[(0,4-spec.shape[0]%4),(0,4-spec.shape[1]%4)],'constant')

class SingleSpecDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = default_spec_adjust

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        # A_img = Image.open(A_path).convert('RGB')
        A_img = np.load(A_path)
        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
