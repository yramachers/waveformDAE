''' Collection of data readers in form of Pytorch datasets '''
import numpy as np
import torch
from torch.utils.data import Dataset
import tables as tb
from scipy.special import erf
import atexit

# Noise waveforms from HDF5 file
class HDF5NoiseWaveforms(Dataset):
    ''' custom dataset class: Gaussian noise waveform only. '''
    def __init__(self, fname):
        super().__init__()
        self.infile = tb.open_file(fname, mode="r")
        self.table = self.infile.root.waveforms
        self.nsamples = 36000  # limit training sample
        # self.nsamples = self.table.nrows
        self.rng = np.random.default_rng()
        atexit.register(self.cleanup)
        
    def __len__(self):
        return self.nsamples
    
    def __getitem__(self, index):
        # noisy pulse
        # template pulse first
        amp = self.rng.integers(1, 6)
        amp *= 25000.0  # adapted to ADC raw data waveforms
        clean = self.function(amp)  # ndarray
        target = torch.from_numpy(clean)  # Tensor
        # add to noise
        row = self.table[index]
        noisy_pulse = clean + row['waveform']  # add to noise waveform
        return torch.from_numpy(noisy_pulse), target  # 1D waveform

    def cleanup(self):
        print('closing file...')
        self.infile.close()
        
    def function(self, A):
        ''' DS model pulse '''
        offset = 690  # template choice
        sigma  = 65.0
        e_sigma1 = 10.0
        e_sigma2 = 25.0
        time = np.arange(4000, dtype=np.float32)  # fixed wfm size
        x    = time - offset
        g    = np.exp(-(x)**2/2/sigma**2)/np.sqrt(2*np.pi)/sigma
        e1   = 1 + erf((x)/np.sqrt(2)/e_sigma1)
        e2   = 1 + erf((x)/np.sqrt(2)/e_sigma2)
        return A*(g*e1*e2)


# Example dataset: swap with own waveform source
# like, typically, reading from file.
# required functions are init(), len() and getitem() as below.
class NoisyPulse(Dataset):
    ''' custom dataset class: Gaussian noise waveform only. '''
    def __init__(self):
        super().__init__()
        self.nsamples = 12000
        self.rng = np.random.default_rng()
        
    def __len__(self):
        return self.nsamples
    
    def __getitem__(self, index):  # all random noise, no index required
        # noisy pulse
        amp = self.rng.integers(1, 10)
        clean = self._simplePulse(1000, amp)  # ndarray
        target = torch.from_numpy(clean)  # Tensor
        noisy_pulse = clean + 0.1*self.rng.standard_normal(
            size=len(clean), dtype=np.float32)
        return torch.from_numpy(noisy_pulse), target  # 1D waveform

    def _simplePulse(self, length, amp):
        time = np.arange(length, dtype=np.float32)
        onset = 0.25*length
        risetime  = 3
        decaytime = 100
        pulse = np.exp(-(time-onset)/risetime)-np.exp(-(time-onset)/decaytime)
        pulse[np.where(time < onset)] = 0.0
        return -amp * pulse

