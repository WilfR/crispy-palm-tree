import unittest
import NiftiDataset
from matplotlib import pyplot as plt
import numpy as np



### @unittest.skip('skipping dataset tests ...')
class NiftiDatasetTests( unittest.TestCase ):

    def testCreateDataset( self ) :
        ds = NiftiDataset.NiftiDataset('./small/t1','./small/t2')
        self.assertEqual( len(ds.t1Filenames), len(ds.t2Filenames) )
        self.assertTrue( len(ds.t1Filenames) > 0 )

    def testSameShape( self ) :
        ds = NiftiDataset.NiftiDataset('./small/t1','./small/t2')
        self.assertEqual( 28, len(ds) )
        for i in range( len(ds) ):
            t1Image, t2Image = ds[i]
            self.assertEqual( t1Image.shape, t2Image.shape )

    def testSlices( self ) :
        ds = NiftiDataset.NiftiDataset('./small/t1','./small/t2')
        t1slice, t2slice = ds.getSlices( 13, 30, NiftiDataset.CORONAL )

        ### plt.subplot(1,2,1)
        ### plt.imshow(t1slice, interpolation='nearest', cmap='gray')
        ### plt.subplot(1,2,2)
        ### plt.imshow(t2slice, interpolation='nearest', cmap='gray')
        ### plt.show()


    def testTransforms( self ) :
        transforms = [ NiftiDataset.RandomCrop3D(80) ]
        ds = NiftiDataset.NiftiDataset('./small/t1','./small/t2',transforms)
        for i in range( len(ds) ) :
            t1Volume, t2Volume = ds[i]
            self.assertEqual( t1Volume.shape, (80,80,80) )
            self.assertEqual( t2Volume.shape, (80,80,80) )

    def testTupleTransform( self ) :
        transforms = [ NiftiDataset.RandomCrop3D((100,90,80)) ]
        ds = NiftiDataset.NiftiDataset('./small/t1','./small/t2',transforms)
        for i in range( len(ds) ) :
            t1Volume, t2Volume = ds[i]
            self.assertEqual( t1Volume.shape, (100, 90, 80) )
            self.assertEqual( t2Volume.shape, (100, 90, 80) )



class RandomCrop3DTest( unittest.TestCase ) :
    def testCreate( self ) :
        cropTransform = NiftiDataset.RandomCrop3D( 80 )
        t1Volume = NiftiDataset.loadNiftiVolume('./small/t1/IXI146-HH-1389-T1_fcm.nii.gz')
        t2Volume = NiftiDataset.loadNiftiVolume('./small/t2/IXI146-HH-1389-T2_reg_fcm.nii.gz')
        t1CroppedVolume, t2CroppedVolume = cropTransform(t1Volume,t2Volume)
        self.assertEqual( t1CroppedVolume.shape, t2CroppedVolume.shape )
        self.assertEqual( t1CroppedVolume.shape, (80,80,80) )

        t1Slice = NiftiDataset.getSliceFromVolume( t1Volume, 40, NiftiDataset.AXIAL )
        t2Slice = NiftiDataset.getSliceFromVolume( t2Volume, 40, NiftiDataset.AXIAL )

        t1CroppedSlice = NiftiDataset.getSliceFromVolume( t1CroppedVolume, 40, NiftiDataset.AXIAL )
        t2CroppedSlice = NiftiDataset.getSliceFromVolume( t2CroppedVolume, 40, NiftiDataset.AXIAL )

        ### plt.subplot(2,2,1)
        ### plt.gray()
        ### plt.imshow(t1Slice)
        ### plt.subplot(2,2,2)
        ### plt.imshow(t2Slice)
        ### plt.subplot(2,2,3)
        ### plt.imshow(t1CroppedSlice)
        ### plt.subplot(2,2,4)
        ### plt.imshow(t2CroppedSlice)
        ### plt.show()



if __name__ == '__main__':
    unittest.main()
