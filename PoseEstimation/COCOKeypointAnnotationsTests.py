import unittest
import COCOKeypointAnnotations
from os import path


def fileExists(filename):
    return path.exists(filename)


class COCOKeypointAnnotationsTests(unittest.TestCase):

    # def testFail(self):
    #     self.assertEqual(0, 1)

    def testCreate(self):
        fname = 'f:\\home\\Data\\COCO\\annotations\\person_keypoints_val2017.json'
        kpas = COCOKeypointAnnotations.COCOKeypointAnnotations(fname)
        self.assertIsNotNone(kpas)

    def testGetFilename(self):
        fname = 'f:\\home\\Data\\COCO\\annotations\\person_keypoints_val2017.json'
        kpas = COCOKeypointAnnotations.COCOKeypointAnnotations(fname)
        imageFilename = kpas.getImageFilename(0)
        self.assertIsNotNone(imageFilename)
        self.assertTrue(fileExists(imageFilename))

    def testFileDoesExist(self):
        fname = './fileDoesExistFile.txt'
        self.assertTrue(fileExists(fname))

    def testFileDoesNotExist(self):
        fname = './fileDoesNotExistFile.txt'
        self.assertFalse(fileExists(fname))


if __name__ == '__main__':
    unittest.main()
