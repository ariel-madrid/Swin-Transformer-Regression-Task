import torch
from torch.utils.data import Sampler
import random

class CustomParameterSampler(Sampler):
    def __init__(self, dataset, batch_size, group_by='mdisk', shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.group_by = group_by

        # Obtener los valores del parámetro por el cual vamos a agrupar
        self.param_values = self._get_param_values()

        # Ordenar índices según el parámetro seleccionado (de menor a mayor)
        self.indices = sorted(range(len(self.dataset)), key=lambda x: self.param_values[x])

        if self.shuffle:
            # Mezclar los índices si shuffle es True
            self.indices = torch.randperm(len(self.dataset)).tolist()

    def _get_param_values(self):
        param_values = []
        for idx in range(len(self.dataset)):
            _, params = self.dataset.__getitem__(idx)
            if self.group_by in ['mdisk', 'gamma', 'rc', 'h100', 'psi']:
                # Si self.group_by es uno de los parámetros
                param_index = {
                    'mdisk': 0,
                    'gamma': 1,
                    'rc': 2,
                    'h100': 3,
                    'psi': 4
                }.get(self.group_by)
                param_value = params[param_index].item()  # Convertir tensor a valor escalar
                param_values.append(param_value)
            else:
                raise ValueError(f"Parámetro {self.group_by} no reconocido en los datos.")
        return param_values

    def __iter__(self):
        # Devolver iterador de índices en el orden determinado
        return iter(self.indices)

    def __len__(self):
        return len(self.dataset)
        
    def set_epoch(self, epoch):
        if self.shuffle:
            random.seed(epoch)
            random.shuffle(self.indices)
