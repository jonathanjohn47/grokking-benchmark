from data.modular_arithmetic import generate_pairs


class ModularArithmeticDataset:
    def __init__(self, number_of_tuples):
        self.number_of_tuples = number_of_tuples
        self.data = list(generate_pairs(number_of_tuples))
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]