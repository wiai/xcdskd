import sys
import time

def print_progress_line(tstart, i, imax, every=100):
    """ 
    print simple progress info line
    
    i and imax are the for-loop-values using a python range, starting from 0
    """
    if ( (i+1) % int(every) == 0):
        # update time info every 'every' patterns
        if (imax != 0):
            progress=100.0 * (i+1)/imax
        tup = time.time()
        togo = (100.0-progress)*(tup-tstart)/(60.0*progress)
        sys.stdout.write("\rtotal points:%5i current:%5i progress: %4.2f%% -> %6.1f min to go" % (imax,i+1,progress,togo))
        sys.stdout.flush()
    if ((i+1) == imax):
        tup = time.time()
        ttime = (tup-tstart) / 60.0
        sys.stdout.write("\rtotal points:%5i current:%5i finished -> total calculation time : %6.1f min " % (imax,i+1,ttime))
        sys.stdout.write("\n")
        sys.stdout.flush()
    return