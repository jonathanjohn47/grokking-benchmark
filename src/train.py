from data.dataset import ModularArithmeticDataset
import data.modular_arithmetic as ma
from torch.utils.data import DataLoader


if __name__ == "__main__":
    print(ModularArithmeticDataset(5))