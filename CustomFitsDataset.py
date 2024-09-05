import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as transforms
from astropy.io import fits
import numpy as np
from torchvision.transforms import ToPILImage, ToTensor
import time
class CustomFitsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.fits_files = [f for f in os.listdir(root_dir) if f.endswith('.fits')]

    def __len__(self):
        return len(self.fits_files)

    def __getitem__(self, idx):
        # Obtener el nombre del archivo .fits
        
        fits_name = os.path.join(self.root_dir, self.fits_files[idx])
        
        # Cargar los datos desde el archivo .fits
        with fits.open(fits_name) as hdul:
            data = hdul[0].data
        
        # Convertir a un tensor
        data = data.astype(np.float32)

        # Calcular la FFT bidimensional
        fft_data = np.fft.fft2(data)
        fft_data_shifted = np.fft.fftshift(fft_data)

        # Obtener los canales real e imaginario
        real_channel = np.real(fft_data_shifted)
        imag_channel = np.imag(fft_data_shifted)
        
        # Convertir los canales a tensores
        real_tensor = torch.tensor(real_channel, dtype=torch.float32).unsqueeze(0)
        imag_tensor = torch.tensor(imag_channel, dtype=torch.float32).unsqueeze(0)
        
        combined_tensor = torch.cat((real_tensor, imag_tensor), dim=0).unsqueeze(0)
        combined_tensor = combined_tensor.squeeze(2)
        
        # Extraer mdisk, rdisk y posang desde el nombre del archivo - incluir los demas parametros mas adelante
        filename_parts = self.fits_files[idx].replace('.fits', '').split('_')
        parameters = torch.tensor([float(part) for part in filename_parts], dtype=torch.float32)
        
        # Aplica transformaciones si es necesario
        if self.transform:
            combined_tensor = self.transform(combined_tensor)
        # Retornar la imagen y los parámetros como un diccionario o una tupla
        combined_tensor = combined_tensor.squeeze(0)
        return combined_tensor, parameters
