package is.hail.methods

import org.apache.spark.mllib.linalg.distributed.BlockMatrix

/**
  * Represents a RelatednessMatrix, which is a matrix where the number at index (i, j) represents how related the samples at index
  * i and j in the samples array are.
  */
class RelatednessMatrix(val matrix: BlockMatrix, val samples: Array[String]) {
    assert(matrix.numCols().toInt == matrix.numRows().toInt && matrix.numCols().toInt == samples.length)
}
