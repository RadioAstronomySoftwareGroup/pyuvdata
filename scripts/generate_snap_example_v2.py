import numpy as np
import time

NCHAN=10
NPOL=4
NBL=6
NTIME=8
NBLTIMES=8*6
data=np.random.randn(NBLTIMES,NCHAN,NPOL)\
+1j*np.random.randn(NBLTIMES,NCHAN,NPOL)
pols=range(NPOL)
times=np.arange(NBLTIMES)*6.+time.time()
frequencies=100e6+100e6/NCHAN*np.arange(NCHAN)
ant1_array=np.array([[0,0,0,1,1,2] for t in range(NTIME)]).astype(int).flatten()
ant2_array=np.array([[0,1,2,1,2,2] for t in range(NTIME)]).astype(int).flatten()


np.savez('snap_correlation_%d.npz'%(times[0]),
data=data,polarizations=pols,frequencies=frequencies,
         times=times,ant1_array=ant1_array,ant2_array=ant2_array)
