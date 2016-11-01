import time
import numpy as np

beg = time.time()
A = np.ones([15, 15])
B = np.eye(15, 15)
u = []
for i in range(10000000):
    u.append(A-B)
end = time.time()
print(end-beg)
