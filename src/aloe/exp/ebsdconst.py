"""
This module contains a number constants which are relevant for different EBSD systems. 

Constants:
----------

BRKR_WIDTH_MICRONS: The full width of Bruker EBSD detector in microns.
    This value is constant for all measured patterns, and is the only one which should be relied on
    when calculating with projection center values from a given image.
    If the aspect ratio of an image measured with the Bruker detector is not exactly 4:3, e.g.
    400x300, 160x120, but e.g. 400x288 or 160x115, the image has been clipped by removing some
    of its top lines and the effective BRKR_HEIGHT_MICRONS (see below) is reduced from its ideal
    4:3 value.
BRKR_HEIGHT_MICRONS: The full height of Bruker EBSD detector. 
    Due to hardware limitations, the top part of the measured patterns 
    can be removed by setting a "top clip" value in the measurement software.
    The top clip is usually 0.04 or 0.05, which leads to pattern dimensions like 160x115.
    The effective screen height is thus reduced.
    When calculating the absolute coordinates of the projection center in microns, 
    the possible top clip value has to be considered 
    because the PCX,PCY,DD are given relative to the pixel-image dimensions.
    The absolute physical image height then corresponds to BRKR_HEIGHT_MICRONS*(1.0-top_clip).
    
"""

# BRUKER 4:3 screen absolute values to relate to absolute stepsize
BRKR_WIDTH_MICRONS=31600 # only the width can be relied on
BRKR_HEIGHT_MICRONS=23700

# TIMEPIX dimensions 
TIMEPIX_WIDTH_MICRONS=256*55
TIMEPIX_HEIGHT_MICRONS=256*55