def train_xor():

    # XOR Truth Table
    t_data = [
        ([0.0, 0.0], [0.0]),    # False ^ False = False
        ([0.0, 1.0], [1.0]),    # False ^ True = True
        ([1.0, 0.0], [1.0]),    # True ^ False = True
        ([1.0, 1.0], [0.0])     # True ^ True = False
    ]

if __name__ == "__main__":
    train_xor()