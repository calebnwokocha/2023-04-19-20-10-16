import math

class Node1:
    def __init__(self, random):
        self.weight = random.getNum1()
        self.bias = random.getNum2()

    def activate(self, x):
        return (self.weight * x) + self.bias

    def train(self, inputs, targets, learningRate):
        for i in range(len(inputs)):
            output = self.activate(inputs[i])
            error = targets[i] - output

            # Gradient descent
            weightDerivative = inputs[i]
            biasDerivative = 1
            weightDelta = learningRate * error * weightDerivative
            biasDelta = learningRate * error * biasDerivative
            self.weight += weightDelta
            self.bias += biasDelta

    def getWeight(self):
        return self.weight

    def getBias(self):
        return self.bias

    def getMeanSquaredError(self, inputs, targets):
        sumSquaredError = 0.0
        for i in range(len(inputs)):
            output = self.activate(inputs[i])
            error = targets[i] - output
            sumSquaredError += math.pow(error, 2)
        return sumSquaredError / len(inputs)