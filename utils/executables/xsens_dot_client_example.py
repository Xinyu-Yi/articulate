r"""
    An example of Xsens Dot client.
"""

import socket
import numpy as np

n_imus = 2   # must be the same as the server
cs = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cs.bind(('127.0.0.1', 8777))
cs.settimeout(5)

while True:
    try:
        data, server = cs.recvfrom(32 * n_imus)
        data = np.frombuffer(data, np.float32)
        t = data[:n_imus]
        q = data[n_imus:5 * n_imus].reshape(n_imus, 4)
        a = data[5 * n_imus:].reshape(n_imus, 3)
        print(server, t, q, a)
    except socket.timeout:
        print('[warning] no data received for 5 seconds')
