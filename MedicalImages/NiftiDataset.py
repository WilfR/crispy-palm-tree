import nibabel as nib
from glob import glob
from random import randint
import os

AXIAL    = 101
SAGITTAL = 102
CORONAL  = 103

def loadNiftiVolume( filename ) :
    x = nib.load(filename)
    return x.get_fdata()

def getSliceFromVolume( volume, sliceIndex, direction ) :
        if direction == CORONAL :
            return volume[ sliceIndex, :, : ]

        if direction == AXIAL :
            return volume[ :, sliceIndex, : ]

        if direction == SAGITTAL :
            return volume[ :, :, sliceIndex ]


class NiftiDataset():
    def __init__(self, sourceImageDir, targetImageDir, transforms = []):
        self.t1ImageDir = sourceImageDir
        self.t2ImageDir = targetImageDir
        self.t1Filenames = glob(os.path.join(self.t1ImageDir, '*.nii*'))
        self.t2Filenames = glob(os.path.join(self.t2ImageDir, '*.nii*'))
        self.transforms = transforms

        t1FileDict={}
        for n in self.t1Filenames :
            key = os.path.basename(n)[0:6]
            t1FileDict[key] = n

        t2FileDict={}
        for n in self.t2Filenames :
            key = os.path.basename(n)[0:6]
            t2FileDict[key] = n

        self.t1Images = {}
        self.t2Images = {}

        for key in t1FileDict:
            t1Volume = loadNiftiVolume (t1FileDict[key])
            t2Volume = loadNiftiVolume (t2FileDict[key])
            self.t1Images[ key ] = t1Volume
            self.t2Images[ key ] = t2Volume

        self.keys = [x for x in self.t1Images.keys()]

    def __len__(self):
        return len( self.keys )

    def __getitem__(self, idx):
        key = self.keys[idx]
        t1Volume = self.t1Images[ key ]
        t2Volume = self.t2Images[ key ]
        for transform in self.transforms :
                t1Volume, t2Volume = transform( t1Volume, t2Volume )

        return (t1Volume, t2Volume )

    def getSlices( self, volumeIndex, sliceIndex, direction) :
        t1Image, t2Image = self.__getitem__(volumeIndex)

        if direction == CORONAL :
            return ( t1Image[ sliceIndex, :, : ], t2Image[ sliceIndex, :, : ] )

        if direction == AXIAL :
            return ( t1Image[ :, sliceIndex, : ], t2Image[ :, sliceIndex, : ] )

        if direction == SAGITTAL :
            return ( t1Image[ :, :, sliceIndex ], t2Image[ :, :, sliceIndex ] )



class RandomCrop3D :
    pass

    def __init__ ( self, s ) :
        if isinstance(s,int) :
            self.shape=(s,s,s)
        elif isinstance(s,tuple) :
            self.shape=s

    def __call__( self, vol1, vol2 ) :
        w0 = self.shape[0]
        w1 = self.shape[1]
        w2 = self.shape[2]
        s0 = randint( 0, vol1.shape[0]-w0 )
        s1 = randint( 0, vol1.shape[1]-w1 )
        s2 = randint( 0, vol1.shape[2]-w2 )


        cropVol1 = vol1[ s0:(s0+w0), s1:(s1+w1), s2:(s2+w2) ]
        cropVol2 = vol2[ s0:(s0+w0), s1:(s1+w1), s2:(s2+w2) ]
        return (cropVol1, cropVol2)

def Main():
    pass

if __name__ == '__main__':
    Main()
