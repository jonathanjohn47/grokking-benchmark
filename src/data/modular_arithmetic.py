def generate_pairs(number):
    for i in range(number):
        for j in range(number):
            yield (i, j, (i+j)%number)