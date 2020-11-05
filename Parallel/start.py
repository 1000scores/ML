import multiprocessing as mp
from multiprocessing import Process, Queue
import numpy as np
import time

def test_processes():
    start_time = time.time()

    def my_for():
        for i in np.arange(2e7):
            kek = 0

    processes = [Process(target=my_for, args=()) for x in range(mp.cpu_count())]
    for p in processes:
        p.start()

    for p in processes:
        p.join()
    print("--- %s seconds ---" % (time.time() - start_time))

def foo(x):
    return x**10000000

def test_pool():
    start_time = time.time()
    pool = mp.Pool(mp.cpu_count())
    ar = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    # result = pool.map(foo, ar)
    result = list(map(foo, ar))
    #print(result)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    #test_processes()
    test_pool()