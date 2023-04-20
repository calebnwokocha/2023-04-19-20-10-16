"""
Author: Caleb Princewill Nwokocha
Website: https://calebnwokocha.github.io/profile/

The purpose of this program is to compare two activation functions:
    (1) y = wx + b
    (2) y = [(w1 * x) / (w1 + (w2 * x))] * x
where w <=> weight, b <=> bias, w1 <=> weight1, w2 = weight2, x <=> input, and y <=> output.

Two nodes are created, one for each activation function. Node1 uses activation function (1) and Node2 uses activation function (2).

A global Random object is created to generate the same random numbers for the weights and biases of the two nodes, so that 
Node1 weight is the same as weight1 of Node2, and Node1 bias is the same as weight2 of Node2.

The two nodes are trained and tested on two tasks:
    (1) Predicting the square of a number
    (2) Predicting the prime number

For the first task, the two nodes are trained to predict the square of 1 to 100. Then, they are tested to predict the square of 101 to 200. 

For the second task, the two nodes are trained to predict prime numbers of index 1 to 100. Then, they are tested to predict prime numbers of index 101 to 200. 

The mean squared error is calculated for each node and shown after testing.

I recommend you run this program multiple times to see the average difference in the mean squared error of the two nodes.
"""

import math

from Random import Random
from Node1 import Node1
from Node2 import Node2

def isPrime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5)+1):
        if num % i == 0:
            return False
    return True

if __name__ == "__main__":
    # Initialize random number generator and nodes
    random = Random()
    node1 = Node1(random)
    node2 = Node2(random)
    
    # To train the two nodes for predicting square numbers
    # Create input and target datasets for the first 100 square numbers
    training_inputs = [float(i) for i in range(1, 101)]
    training_targets = [float(math.pow(num, 2)) for num in range(1, 10000)][:100]

    # Create input and target datasets for the next 100 square numbers
    testing_inputs = [float(i) for i in range(101, 201)]
    testing_targets = [float(math.pow(num, 2)) for num in range(10201, 40401)][:100]

    # Print input and target datasets
    print()
    print("DATASET:")
    print()
    print("Input -> Target")
    for i in range(100):
        print(f"{training_inputs[i]}   ->   {training_targets[i]}")
    print()

    # Train nodes on input and target datasets
    print("TRAINING NODES...")
    for epoch in range(1000000):
        for i in range(100):
            node1.train([training_inputs[i-1]], [training_targets[i-1]], 0.000001)
            node2.train([training_inputs[i-1]], [training_targets[i-1]], 0.000001)
    print()
    print("TRAINING COMPLETE")

    # Test the node1 and node2 on the same set of inputs and targets
    print()
    print("TESTING NODES...")
    print()
    for i in range(100):
        print("Test", i+1)
        print(f"Node 1: Input = {testing_inputs[i]}, Target = {testing_targets[i]}, Prediction = {node1.activate(testing_inputs[i])}, Weight = {node1.getWeight()}, Bias = {node1.getBias()}")
        print(f"Node 2: Input = {testing_inputs[i]}, Target = {testing_targets[i]}, Prediction = {node2.activate(testing_inputs[i])}, Weight1 = {node2.getW1()}, Weight2 = {node2.getW2()}")
        print()

    # Evaluate mean squared error for the nodes on the input and target datasets
    mse1 = node1.getMeanSquaredError(testing_inputs, testing_targets)
    mse2 = node2.getMeanSquaredError(testing_inputs, testing_targets)
    print(f"Node 1: Mean Squared Error = {mse1}")
    print(f"Node 2: Mean Squared Error = {mse2}")
    print()
    print("TESTING COMPLETE")
    

    # Uncomment the code below to see the two nodes preformance on prime number prediction test.
    
    # To train the two nodes for predicting prime numbers
    # Create input and target datasets for the first 100 prime numbers
    training_inputs = [float(i) for i in range(1, 101)]
    training_targets = [float(num) for num in range(2, 547) if isPrime(num)][:100]

    # Create input and target datasets for the next 100 prime numbers
    testing_inputs = [float(i) for i in range(101, 201)]
    testing_targets = [float(num) for num in range(547, 1231) if isPrime(num)][:100]
    
    # Print input and target datasets
    print()
    print("DATASET:")
    print()
    print("Input -> Target")
    for i in range(100):
        print(f"{training_inputs[i]}   ->   {training_targets[i]}")
    print()
    
    # Train nodes on input and target datasets
    print("TRAINING NODES...")
    for epoch in range(1000000):
        for i in range(100):
            node1.train([training_inputs[i-1]], [training_targets[i-1]], 0.000001)
            node2.train([training_inputs[i-1]], [training_targets[i-1]], 0.000001)
    print()
    print("TRAINING COMPLETE")

    # Test the node1 and node2 on the same set of inputs and targets
    print()
    print("TESTING NODES...")
    print()
    for i in range(100):
        print("Test", i+1)
        print(f"Node 1: Input = {testing_inputs[i]}, Target = {testing_targets[i]}, Prediction = {node1.activate(testing_inputs[i])}, Weight = {node1.getWeight()}, Bias = {node1.getBias()}")
        print(f"Node 2: Input = {testing_inputs[i]}, Target = {testing_targets[i]}, Prediction = {node2.activate(testing_inputs[i])}, Weight1 = {node2.getW1()}, Weight2 = {node2.getW2()}")
        print()

    # Evaluate mean squared error for the nodes on the input and target datasets
    mse1 = node1.getMeanSquaredError(testing_inputs, testing_targets)
    mse2 = node2.getMeanSquaredError(testing_inputs, testing_targets)
    print(f"Node 1: Mean Squared Error = {mse1}")
    print(f"Node 2: Mean Squared Error = {mse2}")
    print()
    print("TESTING COMPLETE")
    