def take_nap(x, y, patterns, indexmap, nn=0):
    """ get neighbor average pattern
    
    use -nn..+nn neighbors, 
    i.e. at nn=1, there will be 8 neighbors = 9 pattern average
    
    patterns in assumed to be 1D array of 2D patterns
    index of pattern as function of x,y is in indexmap
    """
    ref_pattern = np.copy(patterns[indexmap[x,y]]).astype(np.int64)
    npa = np.zeros_like(ref_pattern, dtype=np.int64)
    for ix in range(x-nn,x+nn+1):
        for iy in range(y-nn, y+nn+1):
            print(ix, iy)
            if ((ix<0) or (ix>=patterns.shape[1]) or (iy<0) or (iy>=patterns.shape[0])):
                npa = npa + ref_pattern
            else:
                npa = npa + patterns[indexmap[ix,iy]]
    return npa