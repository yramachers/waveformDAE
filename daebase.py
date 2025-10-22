''' DAE base code using pytorch '''
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class WrappedDataLoader:
    ''' run data product from DataLoader through Python function. '''
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(b))


class DAE(nn.Module):
    ''' custom denoising autoencoder, DAE, model 
        structure of convolutions and maxpooling is
        inspired by https://doi.org/10.48550/arXiv.1803.04189
    '''
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv1d(1, 64, 3),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3),
            nn.ReLU(),
            nn.Conv1d(32, 16, 3),
            nn.ReLU(),
            nn.Conv1d(16, 8, 3),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=1),
            nn.Conv1d(8, 32, 4),
            nn.ReLU(),
            nn.Conv1d(32, 16, 4),
            nn.ReLU(),
            nn.Conv1d(16, 8, 4),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=1),
            nn.Conv1d(8, 32, 4),
            nn.ReLU(),
            nn.Conv1d(32, 16, 4),
            nn.ReLU(),
            nn.Conv1d(16, 8, 4),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.Conv1d(8, 64, 3),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3),
            nn.ReLU(),
            nn.Conv1d(32, 16, 3),
            nn.ReLU(),
            nn.Conv1d(16, 8, 3),
            nn.ReLU()
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose1d(8, 8, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 16, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 32, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 64, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(64, 8, 4),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 16, 4),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 32, 4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(32, 8, 4),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 16, 4),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 32, 4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(32, 8, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 16, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 32, 3),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 64, 3),
            nn.ReLU(),
            nn.Conv1d(64, 1, 7)
        )

    def forward(self, data):
        ''' forward pass '''
        x = self.encode(data)
        return self.decode(x)


def preprocess(x):
    ''' reshape tensor for use in DAE layers. '''
    return x[0].view(-1, 1, x[0].shape[-1]), x[1].view(-1, 1, x[1].shape[-1])


# The following convenience functions follow the torch tutorial
# here: https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html
# where appropriate for this specific example project.
def get_data(tdset, vdset, bs):
    ''' hand over data loaders for train and validation. '''
    return (DataLoader(dataset=tdset, batch_size=bs, shuffle=True, drop_last=True),
            DataLoader(dataset=vdset, batch_size=2*bs, shuffle=False, drop_last=True))


def get_model_opt(lrate):
    ''' instatiate model and optimizer '''
    model = DAE()
    return model, optim.Adam(model.parameters(), lr=lrate)


def loss_batch(model, loss_func, indata, target, opt=None):
    ''' use the loss function on batch of data for train and validate. '''
    loss = loss_func(model(indata), target)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), len(indata)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    ''' train and validate function. '''
    for epoch in range(epochs):
        model.train()
        for xb, target in train_dl:
            loss_batch(model, loss_func, xb, target, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, target) for xb, target in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print("Epoch: ",epoch, "; validation loss: ", val_loss)
