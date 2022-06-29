import torch
import pandas as pd
import numpy as np

TRAINSIZE = 463715

class YearPredictionDataSet(torch.utils.data.Dataset):
    def __init__(self, csv_name: str, train: bool):
        # TODO: если потребуется то добавить во-сть трансформации данных
        
        if train:
            self.sample_target = pd.read_csv(csv_name, header = None).to_numpy()[:TRAINSIZE]
        else:
            self.sample_target = pd.read_csv(csv_name, header=None).to_numpy()[TRAINSIZE:]

        self.data = np.copy(self.sample_target[:, list(np.arange(1, 91))])
        self.label = np.copy(self.sample_target[:, 0])

    def __len__(self):
        return self.sample_target.shape[0]

    def __getitem__(self, item):
        sample = {'sample': torch.tensor(self.data[item], dtype=torch.float64),
              'target': self.label[item]}

        return sample



