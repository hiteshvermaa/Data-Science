import numpy as np
import matplotlib.pyplot as plt

from pylab import *
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

np.random.seed(2)

pageSpeed = np.random.normal(3, 1, 100)
purchaseAmount = np.random.normal(50, 30, 100) / pageSpeed

trainx = pageSpeed[:80]
testx = pageSpeed[80:]
# trainx, testx, trainy, testy = train_test_split(pageSpeed, purchaseAmount, train_size = 0.8, test_size = 0.2)
trainy = purchaseAmount[:80]
testy = purchaseAmount[80:]

# scatter(trainx,  trainy)
# scatter(testx, testy)
# x = np.array(trainx)
# print(testy)
# print(x)

p4 = np.poly1d(np.polyfit(trainx, trainy, 6))

xp = np.linspace(0, 7, 100)
axes = plt.axes()
axes.set_xlim(0, 7)
axes.set_ylim(0, 200)
scatter(trainx, trainy)
plt.plot(xp, p4(xp), c='r')
plt.show()

r2 = r2_score(np.array(trainy), p4(np.array(trainx)))
print('r2_score of training data = ', r2)
test2 = r2_score(np.array(testy), p4(np.array(testx)))
print('r2_score of test data     = ', test2)