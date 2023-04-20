"""
Author: Caleb Princewill Nwokocha
Website: https://calebnwokocha.github.io/profile/
"""

import random

class Random:
    def __init__(self):
        self.num1 = random.random()
        self.num2 = random.random()

    def getNum1(self):
        return self.num1

    def getNum2(self):
        return self.num2
