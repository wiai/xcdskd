
"""
2D image filter of numpy arrays, via FFT.

Connelly Barnes, public domain 2007.
"""

import numpy

__version__ = '1.0.0'

def filter(I, K, cache=None):
  """
  Filter image I with kernel K.

  Image color values outside I are set equal to the nearest border color on I.

  To filter many images of the same size with the same kernel more efficiently, use:

    >>> cache = []
    >>> filter(I1, K, cache)
    >>> filter(I2, K, cache)
    ...

  An even width filter is aligned by centering the filter window around each given
  output pixel and then rounding down the window extents in the x and y directions.
  """
  def roundup_pow2(x):
    y = 1
    while y < x:
      y *= 2
    return y

  I = numpy.asarray(I)
  K = numpy.asarray(K)

  if len(I.shape) == 3:
    s = I.shape[0:2]
    L = []
    ans = numpy.concatenate([filter(I[:,:,i], K, L).reshape(s+(1,))
                             for i in range(I.shape[2])], 2)
    return ans
  if len(K.shape) != 2:
    raise ValueError('kernel is not a 2D array')
  if len(I.shape) != 2:
    raise ValueError('image is not a 2D or 3D array')

  s = (roundup_pow2(K.shape[0] + I.shape[0] - 1),
       roundup_pow2(K.shape[1] + I.shape[1] - 1))
  Ipad = numpy.zeros(s)
  Ipad[0:I.shape[0], 0:I.shape[1]] = I

  if cache is not None and len(cache) != 0:
    (Kpad,) = cache
  else:
    Kpad = numpy.zeros(s)
    Kpad[0:K.shape[0], 0:K.shape[1]] = numpy.flipud(numpy.fliplr(K))
    Kpad = numpy.fft.rfft2(Kpad)
    if cache is not None:
      cache[:] = [Kpad]

  Ipad[I.shape[0]:I.shape[0]+(K.shape[0]-1)//2,:I.shape[1]] = I[I.shape[0]-1,:]
  Ipad[:I.shape[0],I.shape[1]:I.shape[1]+(K.shape[1]-1)//2] = I[:,I.shape[1]-1].reshape((I.shape[0],1))

  xoff = K.shape[0]-(K.shape[0]-1)//2-1
  yoff = K.shape[1]-(K.shape[1]-1)//2-1
  Ipad[Ipad.shape[0]-xoff:,:I.shape[1]] = I[0,:]
  Ipad[:I.shape[0],Ipad.shape[1]-yoff:] = I[:,0].reshape((I.shape[0],1))

  Ipad[I.shape[0]:I.shape[0]+(K.shape[0]-1)//2,I.shape[1]:I.shape[1]+(K.shape[1]-1)//2] = I[-1,-1]
  Ipad[Ipad.shape[0]-xoff:,I.shape[1]:I.shape[1]+(K.shape[1]-1)//2] = I[0,-1]
  Ipad[I.shape[0]:I.shape[0]+(K.shape[0]-1)//2,Ipad.shape[1]-yoff:] = I[-1,0]
  Ipad[Ipad.shape[0]-xoff:,Ipad.shape[1]-yoff:] = I[0,0]

  ans = numpy.fft.irfft2(numpy.fft.rfft2(Ipad) * Kpad, Ipad.shape)

  off = ((K.shape[0]-1)//2, (K.shape[1]-1)//2)
  ans = ans[off[0]:off[0]+I.shape[0],off[1]:off[1]+I.shape[1]]

  return ans


def gaussian(sigma=0.5, shape=None):
  """
  Gaussian kernel numpy array with given sigma and shape.

  The shape argument defaults to ceil(6*sigma).
  """
  sigma = max(abs(sigma), 1e-10)
  if shape is None:
    shape = max(int(6*sigma+0.5), 1)
  if not isinstance(shape, tuple):
    shape = (shape, shape)
  x = numpy.arange(-(shape[0]-1)/2.0, (shape[0]-1)/2.0+1e-8)
  y = numpy.arange(-(shape[1]-1)/2.0, (shape[1]-1)/2.0+1e-8)
  Kx = numpy.exp(-x**2/(2*sigma**2))
  Ky = numpy.exp(-y**2/(2*sigma**2))
  ans = numpy.outer(Kx, Ky) / (2.0*numpy.pi*sigma**2)
  return ans/sum(sum(ans))


def test():
  print('Testing:')
  def arrayeq(A, B):
    return sum(sum(abs(A-B)))<1e-7
  A = [[1,2],[3,4]]
  assert arrayeq(filter(A, [[1]]), A)
  assert arrayeq(filter(A, [[0,1,0],[1,1,1],[0,1,0]]),
                           [[8,11],[14,17]])
  assert arrayeq(filter(A, [[1,1,1,1,1]]),
                           [[7,8],[17,18]])
  assert arrayeq(filter(A, [[1,1,1,1,1]]*5),
                           [[55,60],[65,70]])
  assert arrayeq(filter([[1]], [[0,1,0],[1,1,1],[0,1,0]]),
                               [[5]])
  assert arrayeq(filter([[2]], [[3]]), [[6]])
  B = [[1,2,3],[4,5,6],[7,8,9]]
  assert arrayeq(filter(B, [[0,1,0],[1,1,1],[0,1,0]]),
                           [[9,13,17],[21,25,29],[33,37,41]])
  assert arrayeq(filter(A, A), [[10,16],[24,30]])

  print('  filter:     OK')

  assert arrayeq(gaussian(0), [[1]])
  assert arrayeq(gaussian(0.001), [[1]])
  assert arrayeq(gaussian(-0.5), gaussian(0.5))
  assert arrayeq(gaussian(0.5), [[ 0.01134374,  0.08381951,  0.01134374],
                                 [ 0.08381951,  0.61934703,  0.08381951],
                                 [ 0.01134374,  0.08381951,  0.01134374]])
  assert abs(sum(sum(gaussian(3)))-1)<1e-7
  assert abs(sum(sum(gaussian(3,(3,3))))-1)<1e-7
  assert abs(sum(sum(gaussian(3,(1,30))))-1)<1e-7
  assert gaussian(3,(3,3)).shape == (3,3)
  assert gaussian(3,(1,30)).shape == (1,30)
  assert gaussian(3,(30,1)).shape == (30,1)
  assert arrayeq(gaussian(100), gaussian(-100))
  print('  gaussian:   OK')


if __name__ == '__main__':
  test()
