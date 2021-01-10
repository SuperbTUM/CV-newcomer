import scratch
import numpy as np
import time

if __name__ == '__main__':
    start = time.time()
    scratch.target_mean_v8_parallel_test()
    end = time.time()
    print((end-start) * 1000)
