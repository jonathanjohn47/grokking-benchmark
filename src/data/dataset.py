from data.modular_arithmetic import generate_pairs


class ModularArithmeticDataset:
    def __init__(self, number):
        self.data = list(generate_pairs(number))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]