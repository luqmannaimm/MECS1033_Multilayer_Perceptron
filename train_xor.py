import math
import random

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + math.exp(-x))

def solve_xor():
    """Train neural network to solve XOR problem"""

    # XOR Truth Table
    t_data = [
        (0.0, 0.0, 0.0),    # False ^ False = False
        (0.0, 1.0, 1.0),    # False ^ True = True
        (1.0, 0.0, 1.0),    # True ^ False = True
        (1.0, 1.0, 0.0)     # True ^ True = False
    ]

    # Random seed
    random.seed(30)

    # Initialize weights and biases
    w_1 = random.uniform(-1, 1)  # Input to hidden neuron 1
    w_2 = random.uniform(-1, 1)  # Input to hidden neuron 1
    b_1 = random.uniform(-1, 1)  # Hidden layer bias

    w_3 = random.uniform(-1, 1)  # Input to hidden neuron 2
    w_4 = random.uniform(-1, 1)  # Input to hidden neuron 2
    b_2 = random.uniform(-1, 1)  # Hidden layer bias

    w_5 = random.uniform(-1, 1)  # Hidden to output neuron
    w_6 = random.uniform(-1, 1)  # Hidden to output neuron
    b_3 = random.uniform(-1, 1)  # Hidden layer bias

    # Training parameters
    learning_rate = 0.5
    max_epochs = 10000
    target_error = 0.01

    ############
    # TRAINING #
    ############

    for epoch in range(1, max_epochs + 1):

        # Initialize total error for this epoch
        total_error = 0.0

        # Iterate through each XOR example
        for x_1, x_2, target in t_data:

            ################
            # FORWARD PASS #
            ################

            # Hidden neuron 1
            h1_input = (w_1*x_1) + (w_2* x_2) + b_1
            h1_output = sigmoid(h1_input)

            # Hidden neuron 2
            h2_input = (w_3*x_1) + (w_4*x_2) + b_2
            h2_output = sigmoid(h2_input)

            # Output neuron
            o_input = (w_5 * h1_output) + (w_6 * h2_output) + b_3
            o_output = sigmoid(o_input)

            # Calculate error
            error = target - o_output
            total_error += error ** 2   # Sum of squared errors

            #################
            # BACKWARD PASS #
            #################

            # Delta for output neuron
            d_o = o_output * (1.0 - o_output) * error

            # Deltas for hidden neurons
            d_h1 = h1_output * (1.0 - h1_output) * (d_o * w_5)
            d_h2 = h2_output * (1.0 - h2_output) * (d_o * w_6)

            # Update weights and biases from hidden to output layer
            w_5 += learning_rate * d_o * h1_output
            w_6 += learning_rate * d_o * h2_output
            b_3 += learning_rate * d_o

            # Update weights and biases from input to hidden neuron 1
            w_1 += learning_rate * d_h1 * x_1
            w_2 += learning_rate * d_h1 * x_2
            b_1 += learning_rate * d_h1

            # Update weights and biases from input to hidden neuron 2
            w_3 += learning_rate * d_h2 * x_1
            w_4 += learning_rate * d_h2 * x_2
            b_2 += learning_rate * d_h2

        # Print progress every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Total Error: {total_error:.6f}")

        # Check for convergence
        if total_error < target_error:
            print(f"Converged at epoch {epoch} with total error {total_error:.6f}!")
            break
    
    print("\nTraining on XOR problem complete!")

    ###########
    # TESTING #
    ###########

    print("\nTesting trained network on XOR inputs...")

    for x_1, x_2, target in t_data:

        ################
        # FORWARD PASS #
        ################

        # Hidden neuron 1
        h1_input = (w_1*x_1) + (w_2* x_2) + b_1
        h1_output = sigmoid(h1_input)   

        # Hidden neuron 2
        h2_input = (w_3*x_1) + (w_4*x_2) + b_2
        h2_output = sigmoid(h2_input)

        # Output neuron
        o_input = (w_5 * h1_output) + (w_6 * h2_output) + b_3
        o_output = sigmoid(o_input)

        # Determine predicted class (True if output >= 0.5)
        predicted_class = 1 if o_output >= 0.5 else 0

        # Print results
        print(f"\nInput: ({x_1}, {x_2})\nTarget: {target}\nOutput: {o_output:.6f}\nPredicted Class: {predicted_class}")

if __name__ == "__main__":
    solve_xor()