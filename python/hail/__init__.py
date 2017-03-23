from hail.representation import IntervalTree

import hail.expr
from hail.context import HailContext
from hail.dataset import VariantDataset
from hail.expr import Type
from hail.keytable import KeyTable
<<<<<<< HEAD
from hail.utils import TextTableConfig
from hail.kinshipMatrix import KinshipMatrix

__all__ = ['HailContext', 'VariantDataset', 'KeyTable', 'TextTableConfig', 'IntervalTree', 'expr', 'representation', 'KinshipMatrix']
=======
from hail.type import Type
from hail.representation import IntervalTree
import hail.type

__all__ = ['HailContext', 'VariantDataset', 'KeyTable', 'IntervalTree', 'type', 'representation']
>>>>>>> Annotate loci
