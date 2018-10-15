"""
Adaptor and Manager classes for EBSD pattern data saved in various fil formats.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt

from aloe.image import kikufilter

def pcoi_to_pcaloe(pcoi):
    """
    convert AZTEC convention to ALOE convention
    """
    return np.array([pcoi[0], 1.0-pcoi[1]*1.333333, pcoi[2]*1.333333])


class H5PatternDataProviderCPRCTF(object):
    """
    Adaptor to Oxford Instruments data: CTF and CPR file, directory with TIFF files
    """
    def __init__(self, h5filename, h5pcfilename=None, 
        scan = 'Scan/', pattern_name = 'Patterns'):
        
        try:
            self.h5=h5py.File(h5filename, "r")
            self.scan = scan
            self.data_group = self.scan + 'EBSD/Data/'
            self.header_group = self.scan + 'EBSD/Header/'
        
            self.patterns     = self.h5[self.data_group + pattern_name]
            self.patterns_raw = self.h5[self.data_group + pattern_name]
        
            self.num_patterns = self.patterns.shape[0]
            print(self.num_patterns)
            self.PatternWidth = np.copy(self.h5[self.header_group+'PatternWidth']) 
            self.PatternHeight = np.copy(self.h5[self.header_group+'PatternHeight']) 
            self.ScreenAspect = float(self.PatternWidth)/float(self.PatternHeight)
            
            try:
                self.static_background = np.copy(self.h5[self.data_group+'StaticBackground'])
            except:
                self.static_background = None
            
            try:
                self.X_CameraTilt =np.copy(self.h5[self.header_group+'CameraTilt' ]) 
            except:
                self.X_CameraTilt =0.0

            try:
                self.SampleTilt =np.copy(self.h5[self.header_group+'SampleTilt' ])
            except:
                self.SampleTilt =90.0
            
            self.xtilt=((self.SampleTilt-90.0)-self.X_CameraTilt) 
            self.xtilt_rad = np.radians(self.xtilt) 
            
            self.SEMImage = self.h5[self.data_group+'/bse_map']
            self.bse_total = self.h5[self.data_group+'/bse_map']
            self.map_step_fac= 1.0 #self.h5[self.header_group+"MapStepFactor"].value
        
            self.map_width = np.copy(self.h5[self.header_group+'NCOLS'])
            self.map_height = np.copy(self.h5[self.header_group+'NROWS'])
            self.x_beam = self.h5[self.data_group+"X BEAM"]
            self.y_beam = self.h5[self.data_group+"Y BEAM"]
        
            try:
                self.indexmap = self.h5[self.header_group+"indexmap"]
            except:
                self.indexmap = None
            
        except:
            print('ERROR: Could not find all HDF5 datasets')
            
        #print('Camera, Sample, Xtilt:',self.X_CameraTilt,self.SampleTilt, self.xtilt )
        #print('HDF5 pattern height/width: ', self.PatternHeight, self.PatternWidth)

        self.pcfit = None
        self.use_pcfit = False
        if h5pcfilename is not None:
            self.set_pcfit_source(h5pcfilename)  
            self.use_pcfit = True          


    def get_scan_group(self, f):
        """ determine a valid scan name"""
        self.scan_name=''
        def checkname(name):
            #print(name)
            if 'EBSD/Data/' in name:
                parts=name.split('/')
                print(parts)
                self.scan_name=parts[0]
                print(self.scan_name)
        f.visit(checkname)
        print('auto scan name:', self.scan_name)
        return 

    def set_pcfit_source(self, h5pcfile):
        """
        take pc values from separate calibration file with maps of pcx,pcy,dd
        """
        self.pcfit=h5py.File(h5pcfile, "r")
        print('Set projection center calibration file: ', h5pcfile)
        return
        

    def get_pattern_data(self, i):
        #pattern=np.copy(self.h5[self.data_group+'RawPatterns'][i])
        pattern=self.h5[self.data_group+'Patterns'][i]
        
        #flatfield=np.mean(self.h5[self.data_group+'RawPatterns'][:,:,:], axis=0, dtype=np.float64)
        #plt.imshow(flatfield)
        #plt.show()
        
        bx=np.copy(self.h5[self.data_group+'X BEAM'][i])
        by=np.copy(self.h5[self.data_group+'Y BEAM'][i])
        
        xs=np.copy(self.h5[self.data_group+'X SAMPLE'][i])
        ys=np.copy(self.h5[self.data_group+'Y SAMPLE'][i])
        
        coordinates=np.array([xs,ys,bx,by])
        
        phi1=np.copy(self.h5[self.data_group+'phi1'][i])
        PHI =np.copy(self.h5[self.data_group+'PHI' ][i])
        phi2=np.copy(self.h5[self.data_group+'phi2'][i])
        euler=np.array([phi1,PHI,phi2],dtype=np.float32)
        
        if (self.use_pcfit):
            print('use PC fit calibration... ')
            BRKR_PCX=np.copy(self.pcfit['/PCFITTER/BRKR_PC_PCX_MAP'][by,bx])
            BRKR_PCY=np.copy(self.pcfit['/PCFITTER/BRKR_PC_PCY_MAP'][by,bx])
            BRKR_DD =np.copy(self.pcfit['/PCFITTER/BRKR_PC_DDD_MAP'][by,bx])
        else:   
            BRKR_PCX=np.copy(self.h5[self.data_group+'PCX'][i])
            BRKR_PCY=np.copy(self.h5[self.data_group+'PCY'][i])
            BRKR_DD =np.copy(self.h5[self.data_group+'DD' ][i])

        
        PC_BRKR=np.array([BRKR_PCX,BRKR_PCY,BRKR_DD])
        plt.imshow(pattern, cmap='gray')
        plt.show()
        return pattern, euler, PC_BRKR, coordinates

    def get_pattern_row_col(self, row, col):
        idx = self.indexmap[row, col]   
        print(row, col, idx)      
        if (idx == -1):
            return None        
        return self.patterns[idx,:,:]

    def get_pattern_idx(self, n):
        return self.patterns[n,:,:]
    
    def get_pattern_raw(self, x, y):
        idx = self.indexmap[y, x]                  
        return self.patterns_raw[idx,:,:]

    def get_euler_rad(self, x, y):
        idx = self.indexmap[y, x] 
        phi1=np.copy(self.h5[self.data_group+'phi1'][idx])
        phi =np.copy(self.h5[self.data_group+'PHI' ][idx])
        phi2=np.copy(self.h5[self.data_group+'phi2'][idx])
        return np.radians([phi1, phi, phi2], dtype=np.float32)
                           
    def get_pc_brkr(self, bx, by):
        if (self.use_pcfit):
            #print('reading external PCFitter calibration... ')
            pcx = np.copy(self.pcfit['/PCFITTER/BRKR_PC_PCX_MAP'][by,bx])
            pcy = np.copy(self.pcfit['/PCFITTER/BRKR_PC_PCY_MAP'][by,bx])
            dd = np.copy(self.pcfit['/PCFITTER/BRKR_PC_DDD_MAP'][by,bx])
        else:   
            #print('using PC in pattern hdf5... ')
            i = self.indexmap[by, bx] 
            pcx = np.copy(self.h5[self.data_group+'PCX'][i])
            pcy = np.copy(self.h5[self.data_group+'PCY'][i])
            dd = np.copy(self.h5[self.data_group+'DD' ][i])

        #pcx = np.copy(self.h5[self.data_group+'PCX'][idx])
        #pcy = np.copy(self.h5[self.data_group+'PCY'][idx])
        #dd  = np.copy(self.h5[self.data_group+'DD' ][idx])
        return np.array([pcx, pcy, dd])
                         
    def get_xtilt_rad(self):
        return self.xtilt_rad    

        
        
class H5PatternDataProviderEDAX(object):
    """
    Adaptor to EDAX data: directory with raw files, ANG file
    """
    def __init__(self, h5filename, h5pcfilename=None, 
        scan = 'Scan/', pattern_name = 'Patterns'):
        
        try:
            self.h5=h5py.File(h5filename, "r")
            self.scan = scan
            self.data_group = self.scan + 'EBSD/Data/'
            self.header_group = self.scan + 'EBSD/Header/'
        
            self.patterns     = self.h5[self.data_group + pattern_name]
            self.patterns_raw = self.h5[self.data_group + pattern_name]
        
            self.num_patterns = self.patterns.shape[0]
            print(self.num_patterns)
            self.PatternWidth = np.copy(self.h5[self.header_group+'PatternWidth']) 
            self.PatternHeight = np.copy(self.h5[self.header_group+'PatternHeight']) 
            self.ScreenAspect = float(self.PatternWidth)/float(self.PatternHeight)
            
            try:
                self.static_background = np.copy(self.h5[self.data_group+'StaticBackground'])
            except:
                self.static_background = None
            
            try:
                self.X_CameraTilt =np.copy(self.h5[self.header_group+'CameraTilt' ]) 
            except:
                self.X_CameraTilt =0.0

            try:
                self.SampleTilt =np.copy(self.h5[self.header_group+'SampleTilt' ])
            except:
                self.SampleTilt =90.0
            
            self.xtilt=((self.SampleTilt-90.0)-self.X_CameraTilt) 
            self.xtilt_rad = np.radians(self.xtilt) 
            
            self.SEMImage = self.h5[self.data_group+'/bse_map']
            self.bse_total = self.h5[self.data_group+'/bse_map']
            self.map_step_fac= 1.0 #self.h5[self.header_group+"MapStepFactor"].value
        
            self.map_width = np.copy(self.h5[self.header_group+'NCOLS'])
            self.map_height = np.copy(self.h5[self.header_group+'NROWS'])
            self.x_beam = self.h5[self.data_group+"X BEAM"]
            self.y_beam = self.h5[self.data_group+"Y BEAM"]
        
            try:
                self.indexmap = self.h5[self.header_group+"indexmap"]
            except:
                self.indexmap = None
            
        except:
            print('ERROR: Could not find all HDF5 datasets')
            
        #print('Camera, Sample, Xtilt:',self.X_CameraTilt,self.SampleTilt, self.xtilt )
        #print('HDF5 pattern height/width: ', self.PatternHeight, self.PatternWidth)

        self.pcfit = None
        self.use_pcfit = False
        if h5pcfilename is not None:
            self.set_pcfit_source(h5pcfilename)  
            self.use_pcfit = True          


    def get_scan_group(self, f):
        """ determine a valid scan name"""
        self.scan_name=''
        def checkname(name):
            #print(name)
            if 'EBSD/Data/' in name:
                parts=name.split('/')
                print(parts)
                self.scan_name=parts[0]
                print(self.scan_name)
        f.visit(checkname)
        print('auto scan name:', self.scan_name)
        return 

    def set_pcfit_source(self, h5pcfile):
        """
        take pc values from separate calibration file with maps of pcx,pcy,dd
        """
        self.pcfit=h5py.File(h5pcfile, "r")
        print('Set projection center calibration file: ', h5pcfile)
        return
        

    def get_pattern_data(self, i):
        #pattern=np.copy(self.h5[self.data_group+'RawPatterns'][i])
        pattern=self.h5[self.data_group+'Patterns'][i]
        
        #flatfield=np.mean(self.h5[self.data_group+'RawPatterns'][:,:,:], axis=0, dtype=np.float64)
        #plt.imshow(flatfield)
        #plt.show()
        
        bx=np.copy(self.h5[self.data_group+'X BEAM'][i])
        by=np.copy(self.h5[self.data_group+'Y BEAM'][i])
        
        xs=np.copy(self.h5[self.data_group+'X SAMPLE'][i])
        ys=np.copy(self.h5[self.data_group+'Y SAMPLE'][i])
        
        coordinates=np.array([xs,ys,bx,by])
        
        phi1=np.copy(self.h5[self.data_group+'phi1'][i])
        PHI =np.copy(self.h5[self.data_group+'PHI' ][i])
        phi2=np.copy(self.h5[self.data_group+'phi2'][i])
        euler=np.array([phi1,PHI,phi2],dtype=np.float32)
        
        if (self.use_pcfit):
            print('use PC fit calibration... ')
            BRKR_PCX=np.copy(self.pcfit['/PCFITTER/BRKR_PC_PCX_MAP'][by,bx])
            BRKR_PCY=np.copy(self.pcfit['/PCFITTER/BRKR_PC_PCY_MAP'][by,bx])
            BRKR_DD =np.copy(self.pcfit['/PCFITTER/BRKR_PC_DDD_MAP'][by,bx])
        else:   
            BRKR_PCX=np.copy(self.h5[self.data_group+'PCX'][i])
            BRKR_PCY=np.copy(self.h5[self.data_group+'PCY'][i])
            BRKR_DD =np.copy(self.h5[self.data_group+'DD' ][i])

        
        PC_BRKR=np.array([BRKR_PCX,BRKR_PCY,BRKR_DD])
        plt.imshow(pattern, cmap='gray')
        plt.show()
        return pattern, euler, PC_BRKR, coordinates

    def get_pattern_row_col(self, row, col):
        idx = self.indexmap[row, col]   
        print(row, col, idx)      
        if (idx == -1):
            return None        
        return self.patterns[idx,:,:]

    def get_pattern_idx(self, n):
        return self.patterns[n,:,:]
    
    def get_pattern_raw(self, x, y):
        idx = self.indexmap[y, x]                  
        return self.patterns_raw[idx,:,:]

    def get_euler_rad(self, x, y):
        idx = self.indexmap[y, x] 
        phi1=np.copy(self.h5[self.data_group+'phi1'][idx])
        phi =np.copy(self.h5[self.data_group+'PHI' ][idx])
        phi2=np.copy(self.h5[self.data_group+'phi2'][idx])
        return np.radians([phi1, phi, phi2], dtype=np.float32)
                           
    def get_pc_brkr(self, bx, by):
        if (self.use_pcfit):
            #print('reading external PCFitter calibration... ')
            pcx = np.copy(self.pcfit['/PCFITTER/BRKR_PC_PCX_MAP'][by,bx])
            pcy = np.copy(self.pcfit['/PCFITTER/BRKR_PC_PCY_MAP'][by,bx])
            dd = np.copy(self.pcfit['/PCFITTER/BRKR_PC_DDD_MAP'][by,bx])
        else:   
            #print('using PC in pattern hdf5... ')
            i = self.indexmap[by, bx] 
            pcx = np.copy(self.h5[self.data_group+'PCX'][i])
            pcy = np.copy(self.h5[self.data_group+'PCY'][i])
            dd = np.copy(self.h5[self.data_group+'DD' ][i])

        #pcx = np.copy(self.h5[self.data_group+'PCX'][idx])
        #pcy = np.copy(self.h5[self.data_group+'PCY'][idx])
        #dd  = np.copy(self.h5[self.data_group+'DD' ][idx])
        return np.array([pcx, pcy, dd])
                         
    def get_xtilt_rad(self):
        return self.xtilt_rad 
        
        
class H5PatternDataProviderBRKR(object):

    def __init__(self, h5filename, h5pcfilename=None):
        print('HDF5 FILE SETUP: ', h5filename)
        self.h5=h5py.File(h5filename, "r")
        self.scan = '/Scan 0/'
        self.data_group='/Scan 0/EBSD/Data/'
        self.header_group='/Scan 0/EBSD/Header/'
        try:
            self.patterns = self.h5[self.data_group+'Patterns']
        except:
            self.patterns = self.h5[self.data_group+'ProcessedPatterns']

        try:
            self.patterns_raw = self.h5[self.data_group+'RawPatterns']
        except:
            self.patterns_raw = None
        
        self.num_patterns = np.copy(self.h5[self.header_group+'NPoints'])
        self.PatternWidth = np.copy(self.h5[self.header_group+'PatternWidth']) 
        self.PatternHeight = np.copy(self.h5[self.header_group+'PatternHeight']) 
        self.ScreenAspect = float(self.PatternWidth)/float(self.PatternHeight)
        self.BRKR_CameraTilt =np.copy(self.h5[self.header_group+'CameraTilt' ]) 
        self.BRKR_SampleTilt =np.copy(self.h5[self.header_group+'SampleTilt' ]) 
        self.xtilt=((self.BRKR_SampleTilt-90.0)-self.BRKR_CameraTilt) 
        self.xtilt_rad = np.radians(self.xtilt) 
        self.SEMImage=self.h5[self.scan+'SEM/SEM Image'][0,:,:]
        self.map_step_fac=self.h5[self.header_group+"MapStepFactor"].value
        try:
            self.indexmap = self.h5[self.data_group+"MAP_POSITIONS"]
        except:
            self.indexmap = self.h5[self.header_group+"indexmap"]

        self.map_width = np.copy(self.h5[self.header_group+'NCOLS'])
        self.map_height = np.copy(self.h5[self.header_group+'NROWS'])
        try:
            self.bse_total = self.h5[self.data_group+"BSE_TOTAL"]
        except:
            self.bse_total = None

        print('Camera, Sample, Xtilt:',self.BRKR_CameraTilt,self.BRKR_SampleTilt, self.xtilt )
        print('HDF5 pattern height/width: ', self.PatternHeight, self.PatternWidth)

        self.pcfit = None
        self.use_pcfit = False
        if h5pcfilename is not None:
            self.set_pcfit_source(h5pcfilename)  
            self.use_pcfit = True          


    def get_scan_group(self, f):
        """ determine valid scan name"""
        self.scan_name=''
        def checkname(name):
            #print(name)
            if 'EBSD/Data/' in name:
                parts=name.split('/')
                print(parts)
                self.scan_name=parts[0]
                print(self.scan_name)
        f.visit(checkname)
        print('auto scan name:', self.scan_name)
        return 

    def set_pcfit_source(self, h5pcfile):
        """
        take pc values from separate calibration file with maps of pcx,pcy,dd
        """
        self.pcfit=h5py.File(h5pcfile, "r")
        print('Set projection center calibration file: ', h5pcfile)
        return
        

     
    def get_pattern_data(self, i):
        #pattern=np.copy(self.h5[self.data_group+'RawPatterns'][i])
        pattern=self.h5[self.data_group+'Patterns'][i]
        
        #flatfield=np.mean(self.h5[self.data_group+'RawPatterns'][:,:,:], axis=0, dtype=np.float64)
        #plt.imshow(flatfield)
        #plt.show()
        
        bx=np.copy(self.h5[self.data_group+'X BEAM'][i])
        by=np.copy(self.h5[self.data_group+'Y BEAM'][i])
        
        xs=np.copy(self.h5[self.data_group+'X SAMPLE'][i])
        ys=np.copy(self.h5[self.data_group+'Y SAMPLE'][i])
        
        coordinates=np.array([xs,ys,bx,by])
        
        phi1=np.copy(self.h5[self.data_group+'phi1'][i])
        PHI =np.copy(self.h5[self.data_group+'PHI' ][i])
        phi2=np.copy(self.h5[self.data_group+'phi2'][i])
        euler=np.array([phi1,PHI,phi2],dtype=np.float32)
        
        if (self.use_pcfit):
            print('use PC fit calibration... ')
            BRKR_PCX=np.copy(self.pcfit['/PCFITTER/BRKR_PC_PCX_MAP'][by,bx])
            BRKR_PCY=np.copy(self.pcfit['/PCFITTER/BRKR_PC_PCY_MAP'][by,bx])
            BRKR_DD =np.copy(self.pcfit['/PCFITTER/BRKR_PC_DDD_MAP'][by,bx])
        else:   
            BRKR_PCX=np.copy(self.h5[self.data_group+'PCX'][i])
            BRKR_PCY=np.copy(self.h5[self.data_group+'PCY'][i])
            BRKR_DD =np.copy(self.h5[self.data_group+'DD' ][i])

        
        PC_BRKR=np.array([BRKR_PCX,BRKR_PCY,BRKR_DD])
        plt.imshow(pattern, cmap='gray')
        plt.show()
        return pattern, euler, PC_BRKR, coordinates

    def get_pattern_row_col(self, row, col):
        idx = self.indexmap[row, col]   
        print(row, col, idx)      
        if (idx == -1):
            return None        
        return self.patterns[idx,:,:]

    def get_pattern_idx(self, n):
        return self.patterns[n,:,:]
    
    def get_pattern_raw(self, x, y):
        idx = self.indexmap[y, x]                  
        return self.patterns_raw[idx,:,:]

    def get_euler_rad(self, x, y):
        idx = self.indexmap[y, x] 
        phi1=np.copy(self.h5[self.data_group+'phi1'][idx])
        phi =np.copy(self.h5[self.data_group+'PHI' ][idx])
        phi2=np.copy(self.h5[self.data_group+'phi2'][idx])
        return np.radians([phi1, phi, phi2], dtype=np.float32)
                           
    def get_pc_brkr(self, bx, by):
        if (self.use_pcfit):
            #print('reading external PCFitter calibration... ')
            pcx = np.copy(self.pcfit['/PCFITTER/BRKR_PC_PCX_MAP'][by,bx])
            pcy = np.copy(self.pcfit['/PCFITTER/BRKR_PC_PCY_MAP'][by,bx])
            dd = np.copy(self.pcfit['/PCFITTER/BRKR_PC_DDD_MAP'][by,bx])
        else:   
            #print('using PC in pattern hdf5... ')
            i = self.indexmap[by, bx] 
            pcx = np.copy(self.h5[self.data_group+'PCX'][i])
            pcy = np.copy(self.h5[self.data_group+'PCY'][i])
            dd = np.copy(self.h5[self.data_group+'DD' ][i])

        #pcx = np.copy(self.h5[self.data_group+'PCX'][idx])
        #pcy = np.copy(self.h5[self.data_group+'PCY'][idx])
        #dd  = np.copy(self.h5[self.data_group+'DD' ][idx])
        return np.array([pcx, pcy, dd])
                         
    def get_xtilt_rad(self):
        return self.xtilt_rad 


class EBSDDataManager:
    """ provide functionality common to all data sources
        e.g. HDF5, BCF, TIFFZIP, ...
    """
    def __init__(self, datasource):
        self.datasource = datasource
        self.patterns     = self.datasource.patterns
        self.patterns_raw = self.datasource.patterns_raw
        self.indexmap = self.datasource.indexmap
        self.map_width = self.datasource.map_width
        self.map_height = self.datasource.map_height
        self.static_background = self.datasource.static_background
        
        self.nap = 0
        self.pattern_preprocessing = False
        return
    
    def get_pattern_data(self, x, y, invert=False):
        """ return dictionary with pattern, 
        calibration and orientation data
        from map position ix, iy"""
        pattern = self.get_nap(x, y, invert=invert)
        euler_rad = self.datasource.get_euler_rad(x,y)
        xtilt_rad = self.datasource.get_xtilt_rad()
        pc_brkr = self.datasource.get_pc_brkr(x,y)
        return { 'pattern':pattern, 'euler_rad':euler_rad,
                 'xtilt_rad':xtilt_rad, 'pc_brkr':pc_brkr }

    def prepare_pattern(self, raw_pattern):
        """ filter raw pattern data: background correction """
        sigma=0.03*raw_pattern.shape[1]
        pattern = kikufilter.process_ebsp(raw_pattern, sigma=sigma)
        return pattern  

    def pattern_preprocessor(self, raw_pattern):
        """
        run raw_pattern through background correction process 
        """
        if self.pattern_preprocessing:
            sigma=0.03*raw_pattern.shape[1]
            pattern = kikufilter.process_ebsp(raw_pattern, sigma=sigma, static_background = self.static_background)
            return pattern
        else: 
            return raw_pattern
        
    def get_nap(self, x, y, invert=False):
        """ get neighbor average pattern
        
        use -nn..+nn neighbors, 
        i.e. at nn=1, there will be 8 neighbors = 9 pattern average

        patterns in assumed to be 1D array of 2D patterns
        index of pattern as function of x,y is in indexmap
        """
        if ((x<0) or (x>=self.map_width) or (y<0) or (y>=self.map_height)):
            return None
        else:
            idx = self.indexmap[y,x]

        if idx == -1:
            # -1 = no pattern at this point in map
            print(x, y, idx)
            return None 
        
        if invert:
            inv_factor = -1.0
        else:
            inv_factor =  1.0
            
        nn = self.nap

        ref_pattern = inv_factor * np.copy(self.patterns[idx]).astype(np.float32)
        npa = np.zeros_like(ref_pattern, dtype=np.float32)
        count = 0

        for ix in range(x-nn, x+nn+1):
            for iy in range(y-nn, y+nn+1):
                if not ((ix<0) or (ix>=self.map_width) or (iy<0) or (iy>=self.map_height)):
                    idx = self.indexmap[iy,ix]
                    if not (idx == -1):
                        npa = npa + self.pattern_preprocessor(self.patterns[idx])
                        count += 1
        npa = inv_factor * npa / count

        if invert:
            npa =npa - np.min(npa)

        return npa 