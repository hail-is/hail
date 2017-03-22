package is.hail.methods

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}

/**
  * Represents a KinshipMatrix, which is a matrix where the number at index (i, j) represents how related the samples at index
  * i and j in the samples array are.
  */
class KinshipMatrix(val matrix: IndexedRowMatrix, val sampleIds: Array[String]) {
  assert(matrix.numCols().toInt == matrix.numRows().toInt && matrix.numCols().toInt == sampleIds.length)

  /**
    * Filters list of samples based on predicate, and removes corresponding row and column from the list.
    * @param pred The predicate that decides whether a sample is kept.
    */
  def filterSamples(pred: (String => Boolean)): KinshipMatrix = {
    val (samplePairsToTake, samplePairsToDrop) = sampleIds.zipWithIndex.partition(pair => pair match {case (sampleName, index) => pred(sampleName)})

    val filteredSamples = samplePairsToTake.map(_._1)
    val sampleNumsToDropSet = samplePairsToDrop.map(_._2).toSet
    val sampleNumsToTakeArray = samplePairsToTake.map(_._2)

    val filteredRows = matrix.rows.filter(ir => !sampleNumsToDropSet(ir.index.toInt))
    val filteredRowsAndCols = filteredRows.map(ir => {
      val index = ir.index - sampleNumsToDropSet.count(_ < ir.index)
      val arrayVec = ir.vector.toArray
      val filteredArray = sampleNumsToTakeArray.map(i => arrayVec(i))
      new IndexedRow(index, Vectors.dense(filteredArray))
    })

    new KinshipMatrix(new IndexedRowMatrix(filteredRowsAndCols), filteredSamples)
  }
}
//BE sure to test that deleting a row decrements other rows by one when testing filterSamples
