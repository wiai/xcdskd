import numpy as np

def fit_affine3D(xpri=None,ysec=None):
    """ Solve the least squares problem xpri * A = ysec
    """
    # Pad the data with ones, so that our transformation can do translations too
    n = xpri.shape[0] # number of rows of x; we need a column of n ones
    print(xpri.shape)
    print(n)
    
    # hstack stacks column-wise
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    X = pad(xpri)
    Y = pad(ysec)

    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    A, res, rank, s = np.linalg.lstsq(X, Y)

    transform = lambda x: unpad(np.dot(pad(x), A))

    print('Matrix: ', A)
    print("Target y:")
    print(ysec)
    print("Result: x*A")
    print(transform(xpri))
    print("Max error:", np.abs(ysec - transform(xpri)).max())
    return A

def transform_affine(x,A):
    """ transform x (n x 3) by A (4x4) to y (n x 3)
    """
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    transform = lambda x: unpad(np.dot(pad(x), A))
    return transform(x)



def do_fit_affine3D():
    """ test the affine transformation fit
    """
    x = np.array([[40., 1160., 0.],
                  [40., 40., 0.],
                  [260., 40., 0.],
                  [260., 1160., 0.]])

    y = np.array([[610., 560., 2.],
                  [610., -560., 0.],
                  [390., -560., 2.],
                  [390., 560., 10.]])

    A = fit_affine3D(x,y)
    print(A)
    return

def sem_fit_affine3D():
    x=np.loadtxt("beam_indices.txt")
    x[:,-1]=0
    print(x)
    y=np.loadtxt("pc_coords.txt")
    print(x.shape)
    print(y.shape)
    A = fit_affine3D(x,y)
    print(A)

    xtest=np.array([[1,0,0]]) # row vector
    ytest=transform_affine(xtest,A)
    print(ytest)


if __name__ == '__main__':
    #do_fit_affine3D()
    sem_fit_affine3D()