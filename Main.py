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

    testing_inputs = [float(i) for i in range(102, 202)]
    testing_targets = [float(math.pow(num, 2)) for num in range(102, 40401)][:100]

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
    for epoch in range(100000):
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

    # Evaluate mean squared error for nodes on input and target datasets
    mse1 = node1.getMeanSquaredError(testing_inputs, testing_targets)
    mse2 = node2.getMeanSquaredError(testing_inputs, testing_targets)
    print(f"Node 1: Mean Squared Error = {mse1}")
    print(f"Node 2: Mean Squared Error = {mse2}")
    print()
    print("TESTING COMPLETE")
    

    # Uncomment the code below to see the two nodes preformance on prime number dataset.
    """
    # Create input and target datasets for the first 100 prime numbers
    training_inputs = [float(i) for i in range(1, 101)]
    training_targets = [float(num) for num in range(2, 547) if isPrime(num)][:100]

    testing_inputs = [float(i) for i in range(102, 202)]
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
    for epoch in range(100000):
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

    # Evaluate mean squared error for nodes on input and target datasets
    mse1 = node1.getMeanSquaredError(testing_inputs, testing_targets)
    mse2 = node2.getMeanSquaredError(testing_inputs, testing_targets)
    print(f"Node 1: Mean Squared Error = {mse1}")
    print(f"Node 2: Mean Squared Error = {mse2}")
    print()
    print("TESTING COMPLETE")
    """