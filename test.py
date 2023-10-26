import numpy as np

spread = 0.001

# a = 2 * np.random.rand(4) - 1
# b = np.random.normal(loc=0, scale=spread)
# print(a)
# print(b)
# print(a + b)

for i in range(10):
    print("in for loop")
    print(i)
    a = np.random.normal(loc=0, scale=spread)
    print(a)
    while a < 0:
        print("in while loop")
        a = a + 0.0001
        print(a)
        print("negative")
    print("out of while loop")
    print(a)
if np.random.normal(loc=0, scale=spread) > 0:
    print("in if statement")
    print("positive")