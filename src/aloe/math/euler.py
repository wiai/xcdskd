""" Rotation matrices from Euler Angles

see:

  J. B. Kuipers "Quaternions and Rotation Sequences", 
  Princeton University Press, 1999

"""

import numpy as np

def Rx(rotation_rad):
    """
    Provide the rotation matrix around the X (:math:`\\vec{e_1}`) axis 
    in a cartesian system, with the input rotation angle in radians.

    Meaning of Rx acting from left on a COLUMN of VECTORS: 
    
    Transformation matrix U for VECTORS.
    This matrix rotates a set of "old" basis vectors 
    :math:`(\\vec{e_1},\\vec{e_2},\\vec{e_3})^T` (column) by +RotAngle (right hand rule) 
    to a new set of basis vectors :math:`(\\vec{e_1}',\\vec{e_2}',\\vec{e_3}')^T` (column)

    Meaning of Rx acting from left a COLUMN of COORDINATE VALUES:
    
    1. (N_P_O):  coordinates of a fixed vector in a "New" basis that is 
    rotated by +RotAngle (passive rotation)
    
    2. (O_P_O):  active rotation of vector coordinates in the same 
    "Old" basis by -RotAngle
    """
    mat=np.array([ [1,                     0,                   0 ],
                   [0,  np.cos(rotation_rad), np.sin(rotation_rad)],
                   [0, -np.sin(rotation_rad), np.cos(rotation_rad)]])
    return mat
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
def Rz(rotation_rad):
    """
    Provide the rotation matrix around the X (:math:`\\vec{e_1}`) axis 
    in a cartesian system, with the input rotation angle in radians.

    Meaning of Rx acting from left on a COLUMN of VECTORS: 
    
    Transformation matrix U for VECTORS.
    This matrix rotates a set of "old" basis vectors 
    :math:`(\\vec{e_1},\\vec{e_2},\\vec{e_3})^T` (column) by +RotAngle (right hand rule) 
    to a new set of basis vectors :math:`(\\vec{e_1}',\\vec{e_2}',\\vec{e_3}')^T` (column)

    Meaning of Rx acting from left a COLUMN of COORDINATE VALUES:
    
    1. (N_P_O):  coordinates of a fixed vector in a "New" basis that is 
    rotated by +RotAngle (passive rotation)
    
    2. (O_P_O):  active rotation of vector coordinates in the same 
    "Old" basis by -RotAngle
    """
    mat=np.array([[ np.cos(rotation_rad) , np.sin(rotation_rad), 0 ],
                  [-np.sin(rotation_rad) , np.cos(rotation_rad), 0 ],
                  [                     0,                    0, 1 ]])
    return mat
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
def Ry(RotAngle):
    """
    Provide the rotation matrix around the X (:math:`\\vec{e_1}`) axis 
    in a cartesian system, with the input rotation angle in radians.

    Meaning of Rx acting from left on a COLUMN of VECTORS: 
    
    Transformation matrix U for VECTORS.
    This matrix rotates a set of "old" basis vectors 
    :math:`(\\vec{e_1},\\vec{e_2},\\vec{e_3})^T` (column) by +RotAngle (right hand rule) 
    to a new set of basis vectors :math:`(\\vec{e_1}',\\vec{e_2}',\\vec{e_3}')^T` (column)

    Meaning of Rx acting from left a COLUMN of COORDINATE VALUES:
    
    1. (N_P_O):  coordinates of a fixed vector in a "New" basis that is 
    rotated by +RotAngle (passive rotation)
    
    2. (O_P_O):  active rotation of vector coordinates in the same 
    "Old" basis by -RotAngle
    """
    mat=np.array([[ np.cos(RotAngle) , 0, -np.sin(RotAngle)],
                  [                0 , 1,                0 ],
                  [ np.sin(RotAngle) , 0,  np.cos(RotAngle)]])
    return mat
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def Rz2D(RotAngle):
    """
    provides the Z axis (e3) rotation matrix in cartesian systems,
    input "RotAngle" in radians

    Transformation matrix U for VECTORS.
    This matrix rotates a set of "old" basis vectors 
    :math:`(\\vec{e_1},\\vec{e_2},\\vec{e_3})^T` (column) by +RotAngle (right hand rule) 
    to a new set of basis vectors :math:`(\\vec{e_1}',\\vec{e_2}',\\vec{e_3}')^T` (column)

    Meaning of Rx acting from left a COLUMN of COORDINATE VALUES:
    
    1. (N_P_O):  coordinates of a fixed vector in a "New" basis that is 
    rotated by +RotAngle (passive rotation)
    
    2. (O_P_O):  active rotation of vector coordinates in the same 
    "Old" basis by -RotAngle
    """
    mat=np.array([[ np.cos(RotAngle) , np.sin(RotAngle)],
                  [-np.sin(RotAngle) , np.cos(RotAngle)]])
    return mat
#------------------------------------------------------------------------------

def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    # http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
        
    theta, phi, z = randnums
    
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M
    
    
    
def euler_tsl2global(phi1_tsl, Phi_tsl, phi2_tsl):
    """
    transform euler angles form edax-tsl software to global reference system
    
    see: M. Jackson etal. Integrating Materials and Manufacturing Innovation 2014, 3:4 Page 8 of 12
        http://www.immijournal.com/content/3/1/4
    """
    from transforms3d import euler

    # 2nd rotation is around NEGATIVE Y axis in global reference system,
    # so we use negative angle around positive Y axis
    Phi_tsl = -Phi_tsl
    R = euler.euler2mat(phi1_tsl, Phi_tsl, phi2_tsl, 'rzyz')
    phi1, Phi, phi2 = euler.mat2euler(R, 'rzxz') 
    return np.array([phi1, Phi, phi2])