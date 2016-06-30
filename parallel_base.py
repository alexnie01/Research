# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 02:59:05 2016

@author: Alexander

Attempts at learning how to use multiprocessing
"""
import time
import multiprocessing as mp
import numpy as np

batch = np.random.randint(1,100, (10,2))

def task(c, q):
    print c
    q.put((c[0], c[1], np.sqrt(c[0] * c[1])))
    return
    
if __name__ == "__main__":
    t0 = time.time()
    q = mp.Queue()
    processes = []
    for i in np.arange(0,len(batch)):
        p = mp.Process(target = task, args = (batch[i], q))
        processes.append(p)
        p.start()
    for j in processes:
        j.join()
    data1 = []
    for i in np.arange(0, len(batch)):
        data1.append(q.get())        
        
#    p1 = mp.Process(target=task1, args = (q,))
#    p2 = mp.Process(target=task2, args = (q,))
#    p1.start()
#    p2.start()
#    p1.join()
#    p2.join()
    t1 = time.time()
    print "time: ", t1-t0
    
    t2 = time.time()
    data2 = []
    for i in np.arange(0, len(batch)):
        data2.append((batch[i][0], batch[i][1], np.sqrt(batch[i][0] * batch[i][1])))
    t3 = time.time()
    print "time: ", t3-t2