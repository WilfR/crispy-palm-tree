import unittest
import ConvNet
import NiftiDataset
import torchvision
import torch

class ConvNetTests( unittest.TestCase ):

    def testCreate( self ) :
        nn = ConvNet.ConvNet()

    ### @unittest.skip('skipping testForward tests ...')
    def testForward( self ) :
        nn = ConvNet.ConvNet()
        ds = NiftiDataset.NiftiDataset('./small/t1','./small/t2')
        t1InputVolume, t2Volume = ds[10]

        t1InputVolume_t = torchvision.transforms.functional.to_tensor(t1InputVolume)
        t1InputVolume_t = t1InputVolume_t.unsqueeze(0)
        t1InputVolume_t = t1InputVolume_t.unsqueeze(0)
        t1InputVolume_t = t1InputVolume_t.float()

        nInputChannels  = 1
        nOutputChannels = 1
        nBatchSize      = 1
        nSlices         = 90
        nRows           = 120
        nCols           = 120

        self.assertEqual( t1InputVolume_t.shape, torch.Size([nBatchSize, nInputChannels, nSlices, nRows, nCols]) );

        t2OutputVolume_t = nn.forward( t1InputVolume_t )
        self.assertEqual( t1InputVolume_t.shape, t2OutputVolume_t.shape )

    def testApplyConv2D( self ) :
        nInputChannels  = 1
        nOutputChannels = 16
        nBatchSize      = 32
        nRows           = 5
        nCols           = 10

        conv2D = torch.nn.Conv2d(nInputChannels,nOutputChannels,kernel_size=3, padding=1)
        img_t  = torch.ones(nBatchSize,nInputChannels,nRows,nCols)
        out_t  = conv2D( img_t )

        self.assertEqual( img_t.shape, torch.Size([nBatchSize,nInputChannels,nRows,nCols]) )
        self.assertEqual( out_t.shape, torch.Size([nBatchSize,nOutputChannels,nRows,nCols]) )

    def testApplyConv3D( self ) :
        nInputChannels  = 1
        nOutputChannels = 16
        nBatchSize      = 32
        nSlices         = 50
        nRows           = 5
        nCols           = 10

        conv3D = torch.nn.Conv3d(nInputChannels,nOutputChannels,kernel_size=3, padding=1)
        img_t  = torch.ones(nBatchSize,nInputChannels,nSlices,nRows,nCols)
        out_t  = conv3D( img_t )

        self.assertEqual( img_t.shape, torch.Size([nBatchSize,nInputChannels,nSlices,nRows,nCols]) )
        self.assertEqual( out_t.shape, torch.Size([nBatchSize,nOutputChannels,nSlices,nRows,nCols]) )




if __name__ == '__main__':
    unittest.main()
