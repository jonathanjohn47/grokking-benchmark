from data.modular_arithmetic import generate_pairs
from torch.utils.data import DataLoader


class ModularArithmeticDataset:
    def __init__(self, number_of_tuples):
        self.number_of_tuples = number_of_tuples
        self.data = list(generate_pairs(number_of_tuples))
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    
    
def get_dataloaders(number_of_tuples):
    pairs = list(generate_pairs(number_of_tuples))
    train_ds, test_ds = pairs[:int(0.3*len(pairs))], pairs[int(0.3*len(pairs)):]
    return DataLoader(train_ds, batch_size=32, shuffle=True), DataLoader(test_ds, batch_size=32, shuffle=False)