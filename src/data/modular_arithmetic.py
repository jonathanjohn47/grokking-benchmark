def generate_pairs(number):
    for i in range(number):
        for j in range(number):
            yield (i, j, (i+j)%number)

def get_dataloaders(number_of_tuples):
    pairs = list(generate_pairs(number_of_tuples))
    train, test = pairs[:int(0.3*len(pairs))], pairs[int(0.3*len(pairs)):]
    return train, test