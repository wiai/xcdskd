import numpy as np

def make2Dmap(Data1D,Xidx,Yidx,NHeight,NWidth):
    ''' make 2D map array from 1D Data1D list with index values in Xidx and Yidx
    final map array height and width are NHeight,NWidth
    '''
    #resulting 2D data set
    Map2D=np.zeros((NHeight,NWidth),dtype=Data1D.dtype)
    # fill map with values from 1D data
    Map2D[Yidx,Xidx]=Data1D.T
    return Map2D   