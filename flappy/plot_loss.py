import math
from matplotlib import pyplot as plt

# with open('loss.txt', 'r') as f:
with open('loss.txt', 'r') as f:
    values = f.read().strip().split("\n")
    # print(values)
    values = [math.log(float(i)) for i in values]
    # values = [(float(i)) for i in values]

plt.plot(range(1, len(values)*3, 3), values, 'r.')
plt.xlabel('Episode number')
plt.ylabel('log10(Loss)')
plt.show()