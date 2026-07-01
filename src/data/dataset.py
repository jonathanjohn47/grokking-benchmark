from torch.utils.data import DataLoader, random_split

from data.modular_arithmetic import generate_pairs


class ModularArithmeticDataset:
    def __init__(self, number):
        self.data = list(generate_pairs(number))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def get_dataloaders(number):
    dataset = ModularArithmeticDataset(number)
    train_size = int(0.3 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    return train_loader, val_loader