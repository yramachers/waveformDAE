''' train/test DAE structures using pytorch '''
import torch
import daebase as dae

# main script
# set up
lrate = 1.e-4
epochs = 20
batch_size = 24

# pulse denoise; use bespoke data set object for different
# training and validation data, e.g. read from file, see daebase.py.
pset = dae.NoisyPulse()

train_set, valid_set = dae.get_data(pset, pset, batch_size)  # get data loader objects
train_dl = dae.WrappedDataLoader(train_set, dae.preprocess)  # run data loader output through function
valid_dl = dae.WrappedDataLoader(valid_set, dae.preprocess)  # run data loader output through function

# model setup
loss_func = torch.nn.MSELoss(reduction='sum')
mod, opt = dae.get_model_opt(lrate)

# train and validate
dae.fit(epochs, mod, loss_func, opt, train_dl, valid_dl)
torch.save(mod.state_dict(), "daemodel.pth")

