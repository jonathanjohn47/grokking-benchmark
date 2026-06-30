import data.modular_arithmetic as ma
from torch.utils.data import DataLoader


if __name__ == "__main__":
    number_of_tuples = 97
    dataloaders = ma.get_dataloaders(number_of_tuples)
    print(dataloaders[0])