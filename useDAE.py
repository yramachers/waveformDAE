''' Example: load and use a saved model '''
import numpy as np
import matplotlib.pyplot as plt
import torch
import daebase as dae

# functions to produce an example waveform as input to the model.
def simple_pulse(length, onset, amplitude, risetime, decaytime):
    ''' pulse model function to work with numpy.'''
    time = np.arange(length, dtype=np.float32)
    pulse = np.exp(-(time - onset)/risetime) - np.exp(-(time - onset)/decaytime)
    pulse[np.where(time < onset)] = 0.0 # not defined before onset time, set 0
    return -amplitude * pulse


def get_data_item(length, onset, amplitude, risetime, decaytime):
    ''' make a noisy pulse with type as expected for DAE. '''
    rng = np.random.default_rng()
    clean = simple_pulse(length, onset, amplitude, risetime, decaytime)
    noisy = clean + 0.1*rng.standard_normal(size=len(clean), dtype=np.float32)
    return clean, noisy


def convert_for_DAE(pulse):
    ''' model needs tensor input of the right format hence numpy in, tensor out. 
    torch requires 1-D array input like a waveform to be presented as 3D tensor
    with a redundant channel dimension, see below, and a batch dimension
    required for training. Input data also must have this 3D structure hence
    this function.
    '''
    data = torch.from_numpy(pulse)  # 1-d tensor from numpy
    data = torch.unsqueeze(data, 0)  # add redundant channel index to tensor
    return data.view(-1, 1, data.shape[-1])  # add batch index to tensor


# Load and set up
# Order is important: load, object instantiation, then load state dict.
st_dict = torch.load("daemodel.pth", weights_only=True)
mod = dae.DAE()

mod.load_state_dict(st_dict)

# Run forward pass from here
mod.eval()

# make one noisy pulse example as input to model.
samples = 1000
amp = 1.0
onset = 0.25 * samples
rt = 3.0
dt = 100.0
wfm, noisy_pulse = get_data_item(samples, onset, amp, rt, dt)
datain = convert_for_DAE(noisy_pulse)

# use the single waveform as input to the model
with torch.no_grad():
    pred = mod(datain)  # action: use model here; calls forward method in model.

# going back to numpy for plotting/printing convenience
original  = datain.numpy()  # as numpy
predicted = pred.numpy()  # as numpy

# all the waveform data is in the final index of the tensor.
print("baseline std original : ", np.std(original[0,0,:int(onset)]))
print("baseline std predicted: ", np.std(predicted[0,0,:int(onset)]))

# plotting
plt.plot(original[0,0,:])
plt.plot(wfm, 'g')
plt.plot(predicted[0,0,:], 'r')
plt.show()
