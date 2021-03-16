import torch
import numpy as np


class CategoriesSampler():
    """
    CategoriesSampler
    
    Samples data points for the current batch. This is present to sample N-way (N-classes) and k shot + q query samples in every batch.
    This is called in every iteration of a single epoch. Hence its length is the number of episodes, which is equal to the number of batches.
    This returns the indices for the current batch, which are passed on to the __getitem__ of the dataloader to get the image and label.
    
    To check: 
    1. Why isn't this inheriting ```Sampler``` class from PyTorch?
    2. The paper used RANDOMSAMPLE without replacement, but here it is done w/ replacement?
    """
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

