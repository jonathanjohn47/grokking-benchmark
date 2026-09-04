from torch import long, tensor
from torch.utils.data import Dataset, DataLoader, random_split

def generate_pairs(number):
    pairs = []
    for i in range(number):
        for j in range(number):
            pairs.append((i, j, (i + j) % number))
    return pairs

def get_dataloaders(number, batch_size):
    modular_arithmetic_dataset = ModularArithmeticDataset(number)
    train_size = int(0.3 * len(modular_arithmetic_dataset))
    test_size = len(modular_arithmetic_dataset) - train_size

    train_dataset, test_dataset = random_split(modular_arithmetic_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader


class ModularArithmeticDataset(Dataset):
    def __init__(self, number):
        self.number = number
        self.pairs = generate_pairs(number)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return (self.get_tensor(self.pairs[idx]), self.pairs[idx][2])

    def get_tensor(self, item):
        # item is (a, b, answer). The third sequence position is the "="
        # token, whose id is the modulus itself (number tokens are 0..p-1,
        # so the "=" token id is p). Previously hardcoded to 97; now
        # parametrised so the Nanda-Unified protocol (p = 113) gives the
        # "=" token id 113. See configs/nanda_unified.yaml.
        sequence = [item[0], item[1], self.number]
        return tensor(sequence)


if __name__ == "__main__":
    print(tensor([5, 3, 8, 113]))