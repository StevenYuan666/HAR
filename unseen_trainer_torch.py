import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
import numpy as np
import random
from model import CNN
import data_preprocessing as d


class Trainer:
    def __init__(self, data, device):
        self.device = device
        self.model = CNN().double()
        if data == "phone":
            x_train, x_test, y_train, y_test = d.get_phone_data()
        else:
            x_train, x_test, y_train, y_test = d.get_watch_data()
        train_data = []
        for i in range(len(x_train)):
            train_data.append([torch.tensor(x_train[i]), torch.tensor(y_train[i])])
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=32)
        y_test = torch.tensor(y_test)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.n_classes = 18

    def _do_min_epoch(self):
        pass

    def _do_min_iter(self):
        pass

    def _do_max_phase(self):
        pass

    def do_training(self):
        pass

    def do_test(self):
        pass

    def test_func(self):
        pass

def main():
    pass

if __name__ == "__main__":
    pass


