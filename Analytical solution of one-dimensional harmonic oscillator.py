import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
from sympy import hermite

# 创建数据点

h = 197.3269804
u = 939
w = 10/h
a = math.sqrt(u*w/h)
x = np.linspace(-10, 10, 1000)
c = a*x
n_list = [0,1,2,3,4,]
N = [math.sqrt(a/(math.sqrt(math.pi)*(2**n)*(math.factorial(n)))) for n in n_list]
y1 = (N[0])*np.exp(-1/2*(c)**2)
y2 = (N[1])*np.exp(-1/2*(c)**2)*2*c
y3 = (N[2])*np.exp(-1/2*(c)**2)*(4*c*c-2)
y4 = (N[3])*np.exp(-1/2*(c)**2)*(8*c**3-12*c)
y5 = (N[4])*np.exp(-1/2*(c)**2)*(16*c**4-48*c**2+12)
# 创建图形
plt.figure(figsize=(10, 6))
plt.plot(x, y1, 'b-', label = 'n = 0')
plt.plot(x, y2, 'r-', label = 'n = 1')
plt.plot(x, y3, 'g-', label = 'n = 2')
plt.plot(x, y4, 'y-', label = 'n = 3')
plt.plot(x, y5, 'm-', label = 'n = 4')
plt.title('一维谐振子')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()

# 显示图形
plt.show()
