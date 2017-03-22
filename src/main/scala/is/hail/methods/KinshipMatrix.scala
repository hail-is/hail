package is.hail.methods

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import scala.collection.Searching._

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
    val sampleNumsToDropArray = samplePairsToDrop.map(_._2)
    val sampleNumsToDropSet = sampleNumsToDropArray.toSet
    val sampleNumsToTakeArray = samplePairsToTake.map(_._2)

    val filteredRows = matrix.rows.filter(ir => !sampleNumsToDropSet(ir.index.toInt))
    val filteredRowsAndCols = filteredRows.map(ir => {
      val numBelowToDelete = sampleNumsToDropArray.search(ir.index.toInt) match {case InsertionPoint(i) => i}
      val index = ir.index - numBelowToDelete
      val arrayVec = ir.vector.toArray
      val filteredArray = sampleNumsToTakeArray.map(i => arrayVec(i))
      new IndexedRow(index, Vectors.dense(filteredArray))
    })

    new KinshipMatrix(new IndexedRowMatrix(filteredRowsAndCols), filteredSamples)
  }
}
