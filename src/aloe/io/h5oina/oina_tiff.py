from PIL import Image
from PIL.TiffTags import TAGS
from xml.etree.ElementTree import fromstring, ElementTree


xml_reference = """<?xml version="1.0" encoding="us-ascii" standalone="yes"?>
<oina-image image-type="ebsp">
  <ebsp-image>
    <pattern-center-x-pu>0.4787524681126526</pattern-center-x-pu>
    <pattern-center-y-pu>0.589386518536846</pattern-center-y-pu>
    <detector-distance-pu>0.53689704241463132</detector-distance-pu>
    <sem-acc-voltage-kv>20</sem-acc-voltage-kv>
    <sem-working-distance-mm>13.504590034484863</sem-working-distance-mm>
    <specimen-tilt-deg>69.999992841008563</specimen-tilt-deg>
    <specimen-tilt-axis>X</specimen-tilt-axis>
    <detector-orientation-euler1-deg>1.2301909031903346</detector-orientation-euler1-deg>
    <detector-orientation-euler2-deg>96.05026355678514</detector-orientation-euler2-deg>
    <detector-orientation-euler3-deg>2.410994407779977</detector-orientation-euler3-deg>
    <lens-distortion>-0.0237</lens-distortion>
    <lens-field-of-view-mm>32.25</lens-field-of-view-mm>
    <detector-insertion-distance-mm>219.98033142089844</detector-insertion-distance-mm>
    <beam-position-offset-x-um>18.782774111417233</beam-position-offset-x-um>
    <beam-position-offset-y-um>-21.587080583562926</beam-position-offset-y-um>
  </ebsp-image>
</oina-image>
"""


def get_oina_xml_from_img(img):
    """ extract XML metadata from OINA tiff image 
    """
    #img = Image.open(filename)    
    xml_tag = img.tag[51122]
    tree = ElementTree(fromstring(xml_tag[0]))
    root = tree.getroot()
    return root



def get_oina_tiff_pc(img):
    """ extract calibration info from OINA tiff header
    """
    #img = Image.open(filename) 
    w,h = img.width, img.height
    root = get_oina_xml_from_img(img)
    for child in root:
        if child.tag == 'ebsp-image':
            for element in child:
                #print(element.tag, element.text)
                if element.tag == 'pattern-center-x-pu':
                    pcx = float(element.text)
                if element.tag == 'pattern-center-y-pu':
                    pcy = float(element.text) * w/h
                if element.tag == 'detector-distance-pu':
                    pcz = float(element.text) * w/h
                    
    return pcx, pcy, pcz

def get_oina_tiff_pc_from_file(filename):
    """ extract calibration info from OINA tiff header
    OINA convention
    """
    img = Image.open(filename) 
    w,h = img.width, img.height
    root = get_oina_xml_from_img(img)
    for child in root:
        if child.tag == 'ebsp-image':
            for element in child:
                #print(element.tag, element.text)
                if element.tag == 'pattern-center-x-pu':
                    pcx = float(element.text)
                if element.tag == 'pattern-center-y-pu':
                    pcy = float(element.text) #* w/h BRKR
                if element.tag == 'detector-distance-pu':
                    pcz = float(element.text) #* w/h BRKR
                    
    return pcx, pcy, pcz

def get_oina_detector_angles(img):
    """ get detector orientation parameters from OINA tiff
    """
    root = get_oina_xml_from_img(img)
    for child in root:
        if child.tag == 'ebsp-image':
            for element in child:
                #print(element.tag, element.text)
                if element.tag == 'detector-orientation-euler1-deg':
                    euler1 = float(element.text)
                if element.tag == 'detector-orientation-euler2-deg':
                    euler2 = float(element.text) 
                if element.tag == 'detector-orientation-euler3-deg':
                    euler3 = float(element.text)
    return [euler1, euler2, euler3]


def test_oina_tiff():
    filename = '../data/12_14.tiff'
    img = Image.open(filename)
    print('TEST: ', filename)
    x,y,z = get_oina_tiff_pc(img)
    print(x,y,z)
    euler = get_oina_detector(img) 
    print(euler)
    return
  
  
def print_oina_tiff_info(filename):
    """ show info from Aztec TIFF file
    """
    import xml.dom.minidom
    print("TIFF info from file: ", filename)
    img = Image.open(filename)
    xml_tag = img.tag[51122]
    dom = xml.dom.minidom.parseString(xml_tag[0]) 
    pretty_xml_as_string = dom.toprettyxml()
    print(pretty_xml_as_string)
    return

                    
if __name__ == "__main__":
    import sys
    tiff_filename = sys.argv[1] 
    print_oina_tiff_info(tiff_filename)
    #test_oina_tiff()