import COCOKeypointAnnotations
from os import path


def fileExists(filename):
    return path.exists(filename)


def test_create():
    fname = 'f:\\home\\Data\\COCO\\annotations\\person_keypoints_val2017.json'
    imageFolder = 'f:\\home\\Data\\COCO\\val2017'
    kpas = COCOKeypointAnnotations.COCOKeypointAnnotations(fname, imageFolder)
    assert kpas is not None


def testGetFilename():
    fname = 'f:\\home\\Data\\COCO\\annotations\\person_keypoints_val2017.json'
    imageFolder = 'f:\\home\\Data\\COCO\\val2017'
    kpas = COCOKeypointAnnotations.COCOKeypointAnnotations(fname, imageFolder)
    imageFilename = kpas.getImageFilename(0)
    assert imageFilename is not None
    assert fileExists(imageFilename) is True


def testGetBoundingBox():
    fname = 'f:\\home\\Data\\COCO\\annotations\\person_keypoints_val2017.json'
    imageFolder = 'f:\\home\\Data\\COCO\\val2017'
    kpas = COCOKeypointAnnotations.COCOKeypointAnnotations(fname, imageFolder)
    boundingBox = kpas.getBoundingBox(0)
    assert len(boundingBox) == 4
    assert boundingBox[2] >= 0
    assert boundingBox[3] >= 0


def testGetKeypoints():
    fname = 'f:\\home\\Data\\COCO\\annotations\\person_keypoints_val2017.json'
    imageFolder = 'f:\\home\\Data\\COCO\\val2017'
    kpas = COCOKeypointAnnotations.COCOKeypointAnnotations(fname, imageFolder)
    keypoints = kpas.getKeypoints(0)
    assert keypoints is not None
    assert len(keypoints) == 17  # 17 keypoints (x, y, visibility, name)


def testCreateKeypoint():
    keypoint = COCOKeypointAnnotations.Keypoint(0, 50, 2, 'nose')
    assert keypoint is not None
    assert keypoint.x == 0
    assert keypoint.y == 50
    assert keypoint.v == 2
    assert keypoint.name == 'nose'


def testFileDoesExist():
    fname = './fileDoesExistFile.txt'
    assert fileExists(fname) is True


def testFileDoesNotExist():
    fname = './fileDoesNotExistFile.txt'
    assert fileExists(fname) is False
