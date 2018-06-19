from __future__ import absolute_import
import os.path as osp
from PIL import Image

class Preprocessor(object):
    # def __init__(self, root=None, image_names=None, centers = None, transform=None):
    def __init__(self, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.root = root
        # self.centers = centers
        self.transform = transform

    def __len__(self):
        return len(self.image_names) 

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        # fname = self.image_names[index]
        fname="{:0>11d}".format(index)+'.jpg'
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # location = self.centers[index]
        # return img, index
        return img




# class Preprocessor(object):
#     # def __init__(self, root=None, image_names=None, centers = None, transform=None):
#     def __init__(self, root=None, image_names=None, transform=None):
#         super(Preprocessor, self).__init__()
#         self.root = root
#         self.image_names = image_names
#         # self.centers = centers
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_names) 

#     def __getitem__(self, indices):
#         if isinstance(indices, (tuple, list)):
#             return [self._get_single_item(index) for index in indices]
#         return self._get_single_item(indices)

#     def _get_single_item(self, index):
#         fname = self.image_names[index]
#         if self.root is not None:
#             fpath = osp.join(self.root, fname)
#         img = Image.open(fpath).convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)
#         # location = self.centers[index]
#         # return img, index
#         return img
