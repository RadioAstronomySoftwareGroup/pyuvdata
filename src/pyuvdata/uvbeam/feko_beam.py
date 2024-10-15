import numpy as np
import glob

def read_feko(file_name_prefix,freq_p,theta_p,phi_p):
    """
    Reads in a typical feko output file and returns numpy arrays of the power beam and efield beams.
    """
    beam_square = np.zeros((freq_p,theta_p,phi_p))
    etheta_square90 = np.zeros((freq_p,theta_p,phi_p),dtype = 'complex')
    ephi_square90 = np.zeros((freq_p,theta_p,phi_p),dtype = 'complex')
   
    f1 = open(file_name_prefix)
   

    z = theta_p*phi_p +10# ---> change this to no.of theta * no.of phi + No.of header lines
    c=0
    for line in f1:
        if c%z ==0:
            co=0
        if c % z >= 10: 
            x = list(map(float,line.split()))
            beam_square[int(c/z), co%181,int(co/181)] = 10**(x[8]/10)##*** beam_square [freq,theta,phi]***
            etheta_square90[int(c/z), co%181,int(co/181)] = x[2] + 1j*(x[3])
            ephi_square90[int(c/z), co%181,int(co/181)] = x[4] + 1j*x[5]
            co = co+1
        c = c+1

    return beam_square, etheta_square90, ephi_square90
