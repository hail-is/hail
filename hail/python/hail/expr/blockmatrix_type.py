import json

# Need a type that specifies the internal type of a block matrix
class tblockmatrix(object):
    @staticmethod
    def _from_java(jtbm):
        return tblockmatrix()
