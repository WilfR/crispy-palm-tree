

className     = 'NiftiDataset'
classFileName = '%s.py' % className
testName      = '%sTests' % className
testFileName  = '%s.py' % testName

with open( classFileName, 'w' ) as classFile:
    classFile.write('class %s():\n' % className)
    classFile.write('    def __init__(self):\n')
    classFile.write('        pass\n')

    classFile.write('\ndef Main():\n')
    classFile.write('    pass\n')

    classFile.write("\nif __name__ == '__main__':\n    Main()\n")

with open( testFileName, 'w' ) as testFile:
    testFile.write('import unittest\n' )
    testFile.write('import %s\n'% className )

    testFile.write('\n')
    testFile.write('class %s( unittest.TestCase ):\n' % testName)
    testFile.write('\n')
    testFile.write('    def testFail( self ) :\n')
    testFile.write('        self.assertEqual(0,1)\n')

    testFile.write("\nif __name__ == '__main__':\n    unittest.main()\n")

