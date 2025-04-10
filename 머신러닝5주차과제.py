// 1-1

import numpy as np
import pandas as pd
from google.colab import drive
import matplotlib.pyplot as plt
drive.mount('/content/drive')

x = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data0405.csv', usecols=['x']).values
y = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data0405.csv', usecols=['y']).values

A = np.hstack([x**0, x])
A = np.asmatrix(A)
w = (A.T * A).I * A.T * y

plt.figure(figsize = (6,4))
plt.title('Linear Regression', fontsize = 16)
plt.xlabel('X', fontsize = 12)
plt.ylabel('Y', fontsize = 12)
plt.grid(alpha = 0.4)
plt.plot(x, y, 'bo', label = 'Training Data')

xp = np.arange(np.min(x), np.max(x), 0.1).reshape(-1,1)
yp = w[0,0] + w[1,0]*xp

plt.plot(xp, yp, 'r', linewidth = 2, label = 'Linear model')
plt.legend()
plt.grid(alpha = 0.3)
plt.show()


//  1-2
import numpy as np
import pandas as pd
from google.colab import drive
import matplotlib.pyplot as plt
drive.mount('/content/drive')

x = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data0405.csv', usecols=['x']).values
y = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data0405.csv', usecols=['y']).values

A = np.hstack([x**0, x, x**2])
A = np.asmatrix(A)
w = (A.T * A).I * A.T * y

plt.figure(figsize = (6,4))
plt.title('Non-Linear Regression', fontsize = 16)
plt.xlabel('X', fontsize = 12)
plt.ylabel('Y', fontsize = 12)
plt.grid(alpha = 0.4)
plt.plot(x, y, 'bo', label = 'Training Data')

xp = np.arange(np.min(x), np.max(x), 0.1).reshape(-1,1)
yp = w[0,0] + w[1,0]*xp + w[2,0]*xp**2

plt.plot(xp, yp, 'r', linewidth = 2, label = 'Non-Linear model')
plt.legend()
plt.grid(alpha = 0.3)
plt.show()

// 2-1

import numpy as np
import pandas as pd
from google.colab import drive
import matplotlib.pyplot as plt
drive.mount('/content/drive')

x = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data0405.csv', usecols=['x']).values
y = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data0405.csv', usecols=['y']).values

alpha = 0.001
epoch = 500
n = float(len(x))
w = np.zeros((2,1))

for i in range(epoch):
  y_hat = w[0,0] + w[1,0]*x
  dw1 = np.sum(x*(y_hat - y)) / n
  dw0 = np.sum(y_hat - y) / n
  w[0,0] = w[0,0] - alpha*dw0
  w[1,0] = w[1,0] - alpha*dw1

plt.figure(figsize = (6,4))
plt.title('Linear Regression', fontsize = 16)
plt.xlabel('X', fontsize = 12)
plt.ylabel('Y', fontsize = 12)
plt.plot(x, y, 'bo', label = 'Training Data')
plt.grid(alpha = 0.4)

xp = np.arange(np.min(x), np.max(x), 0.01).reshape(-1,1)
yp = w[0,0] + w[1,0]*xp
plt.plot(xp, yp, 'r', linewidth = 2, label = 'Linear model')
plt.legend()
plt.grid(alpha = 0.3)
plt.show()

//  2-2

import numpy as np
import pandas as pd
from google.colab import drive
import matplotlib.pyplot as plt
drive.mount('/content/drive')

x = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data0405.csv', usecols=['x']).values
y = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data0405.csv', usecols=['y']).values

alpha = 0.001
epoch = 500
n = float(len(x))
w = np.zeros((5,1))

for i in range(epoch):
  y_hat = w[0,0] + w[1,0]*x + w[2,0]*x**2 + w[3,0]*x**3 + w[4,0]*x**4
  dw4 = np.sum((x**4)*(y_hat - y)) / n
  dw3 = np.sum((x**3)*(y_hat - y)) / n
  dw2 = np.sum((x**2)*(y_hat - y)) / n
  dw1 = np.sum(x*(y_hat - y)) / n
  dw0 = np.sum(y_hat - y) / n
  w[0,0] = w[0,0] - alpha*dw0
  w[1,0] = w[1,0] - alpha*dw1
  w[2,0] = w[2,0] - alpha*dw2
  w[3,0] = w[3,0] - alpha*dw3
  w[4,0] = w[4,0] - alpha*dw4

plt.figure(figsize = (6,4))
plt.title('Linear Regression', fontsize = 16)
plt.xlabel('X', fontsize = 12)
plt.ylabel('Y', fontsize = 12)
plt.plot(x, y, 'bo', label = 'Training Data')
plt.grid(alpha = 0.4)

xp = np.arange(np.min(x), np.max(x), 0.01).reshape(-1,1)
yp = w[0,0] + w[1,0]*xp +  w[2,0]*xp**2 + w[3,0]*xp**3 + w[4,0]*xp**4
plt.plot(xp, yp, 'r', linewidth = 2, label = 'Linear model')
plt.legend()
plt.grid(alpha = 0.3)
plt.show()
