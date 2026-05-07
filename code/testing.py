#from main import optimizer_h0_5
import pandas as pd
import itertools

x = 1.5625000000023461e-38
print(x**2)

import matplotlib.pyplot as plt
f1 = plt.figure()
f2 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot(range(0,10))
ax2 = f2.add_subplot(111)
ax2.plot(range(10,20))
plt.show()

x = pd.DataFrame()
x["test"] = [0]
dict_regimes_combinations = [combs for combs in itertools.product([0,1], repeat=4)]
trained_regimes = {
    "h=0.5":0,
    "h=1.0":1,
    "h=2.0":0,
    "h=10⁻⁶":0
}
trained_regimes.values() = (0,1,1,1)


