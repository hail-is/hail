package is.hail.methods

import org.apache.spark.mllib.linalg.distributed.BlockMatrix

/**
  * Created by johnc on 3/15/17.
  */
class RelatednessMatrix(val matrix: BlockMatrix, val samples: Array[String]) {
  //assert(matrix.numCols() == matrix.numRows() == samples.length)
}
