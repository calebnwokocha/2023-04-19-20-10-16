"""
Author: Caleb Princewill Nwokocha
Website: https://calebnwokocha.github.io/profile/
"""

import math

class Node2:
    def __init__(self, random):
        self.w1 = random.getNum1()
        self.w2 = random.getNum2()

    def activate(self, x):
        return ((self.w1 * x) / (self.w1 + (self.w2 * x))) * x

    def train(self, inputs, targets, learningRate):
        for i in range(len(inputs)):
            output = self.activate(inputs[i])
            error = targets[i] - output

            # Gradient descent
            derivativeW1 = (math.pow(inputs[i], 3) * self.w2) / math.pow(self.w1 + (self.w2 * inputs[i]), 2)
            derivativeW2 = -(math.pow(inputs[i], 3) * self.w1) / math.pow(self.w1 + (self.w2 * inputs[i]), 2)
            deltaW1 = learningRate * error * derivativeW1 * inputs[i]
            deltaW2 = learningRate * error * derivativeW2 * inputs[i]
            self.w1 += deltaW1
            self.w2 += deltaW2

    def getW1(self):
        return self.w1

    def getW2(self):
        return self.w2

    def getMeanSquaredError(self, inputs, targets):
        sumSquaredError = 0.0
        for i in range(len(inputs)):
            output = self.activate(inputs[i])
            error = targets[i] - output
            sumSquaredError += math.pow(error, 2)
        return sumSquaredError / len(inputs)
