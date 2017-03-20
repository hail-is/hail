package is.hail.methods

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}

/**
  * Represents a RelatednessMatrix, which is a matrix where the number at index (i, j) represents how related the samples at index
  * i and j in the samples array are.
  */
class RelatednessMatrix(val matrix: IndexedRowMatrix, val samples: Array[String]) {
    assert(matrix.numCols().toInt == matrix.numRows().toInt && matrix.numCols().toInt == samples.length)

  /**
    * Filters list of samples based on predicate, and removes corresponding row and column from the list.
    * @param pred
    */
  def filterSamples(pred: (String => Boolean)): RelatednessMatrix = {
    val filteredSamplePairs= samples.zipWithIndex.filter(pair => pair match { case (sampleName, index) => pred(sampleName)})
    val filteredSamples = filteredSamplePairs.map(_._1)
    val sampleNums = filteredSamplePairs.map(_._2)

    val filteredRows = matrix.rows.filter(ir => sampleNums.contains(ir.index))
    val filteredRowsAndCols = filteredRows.map(ir =>
      new IndexedRow(ir.index, Vectors.dense(ir.vector.toArray.zipWithIndex.filter(tuple => tuple match { case (value, index) => sampleNums.contains(index)}).map(_._1))))

    //matrix.blocks.map(tuple => tuple match {case ((a, b), matrix) => {
    new RelatednessMatrix(new IndexedRowMatrix(filteredRowsAndCols), filteredSamples)
  }
}
