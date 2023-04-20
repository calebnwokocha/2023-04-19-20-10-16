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

    # Create input and target datasets for the first 100 square numbers
    inputs = [float(i) for i in range(1, 101)]
    targets = [float(math.pow(num, 2)) for num in range(1, 10000)][:100]

    # Print input and target datasets
    print()
    print("DATASET:")
    print()
    print("Input -> Target")
    for i in range(100):
        print(f"{inputs[i]}   ->   {targets[i]}")
    print()

    # Train nodes on input and target datasets
    print("TRAINING NODES...")
    for j in range(100000):
        for i in range(100):
            node1.train([inputs[i-1]], [targets[i-1]], 0.000001)
            node2.train([inputs[i-1]], [targets[i-1]], 0.000001)
    print()
    print("TRAINING COMPLETE")

    # Test the node1 and node2 on the same set of inputs and targets
    print()
    print("TESTING NODES...")
    print()
    for i in range(100):
        print("Test", i+1)
        print(f"Node 1: Input = {inputs[i]}, Target = {targets[i]}, Prediction = {node1.activate(inputs[i])}, Weight = {node1.getWeight()}, Bias = {node1.getBias()}")
        print(f"Node 2: Input = {inputs[i]}, Target = {targets[i]}, Prediction = {node2.activate(inputs[i])}, Weight1 = {node2.getW1()}, Weight2 = {node2.getW2()}")
        print()

    # Evaluate mean squared error for nodes on input and target datasets
    mse1 = node1.getMeanSquaredError(inputs, targets)
    mse2 = node2.getMeanSquaredError(inputs, targets)
    print(f"Node 1: Mean Squared Error = {mse1}")
    print(f"Node 2: Mean Squared Error = {mse2}")
    print()
    print("TESTING COMPLETE")


    # Uncomment the code below to see the two nodes preformance on prime number dataset.
    """
    # Create input and target datasets for the first 100 prime numbers
    inputs = [float(i) for i in range(1, 101)]
    targets = [float(num) for num in range(2, 547) if isPrime(num)][:100]
    
    # Print input and target datasets
    print()
    print("DATASET:")
    print()
    print("Input -> Target")
    for i in range(100):
        print(f"{inputs[i]}   ->   {targets[i]}")
    print()
    
    # Train nodes on input and target datasets
    print("TRAINING NODES...")
    for j in range(100000):
        for i in range(100):
            node1.train([inputs[i-1]], [targets[i-1]], 0.000001)
            node2.train([inputs[i-1]], [targets[i-1]], 0.000001)
    print()
    print("TRAINING COMPLETE")

    # Test the node1 and node2 on the same set of inputs and targets
    print()
    print("TESTING NODES...")
    print()
    for i in range(100):
        print("Test", i+1)
        print(f"Node 1: Input = {inputs[i]}, Target = {targets[i]}, Prediction = {node1.activate(inputs[i])}, Weight = {node1.getWeight()}, Bias = {node1.getBias()}")
        print(f"Node 2: Input = {inputs[i]}, Target = {targets[i]}, Prediction = {node2.activate(inputs[i])}, Weight1 = {node2.getW1()}, Weight2 = {node2.getW2()}")
        print()

    # Evaluate mean squared error for nodes on input and target datasets
    mse1 = node1.getMeanSquaredError(inputs, targets)
    mse2 = node2.getMeanSquaredError(inputs, targets)
    print(f"Node 1: Mean Squared Error = {mse1}")
    print(f"Node 2: Mean Squared Error = {mse2}")
    print()
    print("TESTING COMPLETE")
    """

