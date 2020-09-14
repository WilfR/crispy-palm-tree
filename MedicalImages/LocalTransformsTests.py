import unittest
import LocalTransforms
from matplotlib import pyplot as plt
import numpy as np

class LocalTransformsTests( unittest.TestCase ):

    def testCreate( self ) :
        t = LocalTransforms.RandomCrop3D((128,128,32))

if __name__ == '__main__':
    unittest.main()
