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
    val filteredSamplePairs= sampleIds.zipWithIndex.filter(pair => pair match { case (sampleName, index) => pred(sampleName)})
    val filteredSamples = filteredSamplePairs.map(_._1)
    val sampleNums = filteredSamplePairs.map(_._2)

    //Need to figure out a better way to subtract indices for deleted rows.
    val samplesToThrowAway = sampleIds.zipWithIndex.map(_._2).toSet -- sampleNums.toSet
    val filteredRows = matrix.rows.filter(ir => sampleNums.contains(ir.index)).map(ir => new IndexedRow(ir.index - samplesToThrowAway.count(_ < ir.index), ir.vector))

    //Filter with while loops.
    val filteredRowsAndCols = filteredRows.map(ir =>
      new IndexedRow(ir.index, Vectors.dense(ir.vector.toArray.zipWithIndex.filter(tuple => tuple match { case (value, index) => sampleNums.contains(index)}).map(_._1))))

    new KinshipMatrix(new IndexedRowMatrix(filteredRowsAndCols), filteredSamples)
  }
}
//BE sure to test that deleting a row decrements other rows by one when testing filterSamples
