package is.hail.methods

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import scala.collection.Searching._

/**
  * Represents a KinshipMatrix. Entry (i, j) encodes the relatedness of the ith and jth samples in sampleIds.
  */
class KinshipMatrix(val matrix: IndexedRowMatrix, val sampleIds: Array[String]) {
  assert(matrix.numCols().toInt == matrix.numRows().toInt && matrix.numCols().toInt == sampleIds.length)

  /**
    * Filters list of samples based on predicate, and removes corresponding rows and columns from the matrix.
    * @param pred The predicate that decides whether a sample is kept.
    */
  def filterSamples(pred: (String => Boolean)): KinshipMatrix = {
    val (samplesWithIndicesToKeep, samplesWithIndicesToDrop) = sampleIds.zipWithIndex.partition(pair => pred(pair._1))

    val filteredSamplesIds = samplesWithIndicesToKeep.map(_._1)
    val sampleIndicesToDropArray = samplesWithIndicesToDrop.map(_._2)
    val sampleIndicesToDropSet = sampleIndicesToDropArray.toSet
    val sampleIndicesToTakeArray = samplesWithIndicesToKeep.map(_._2)

    val filteredRows = matrix.rows.filter(ir => !sampleIndicesToDropSet(ir.index.toInt))
    val filteredRowsAndCols = filteredRows.map(ir => {
      val InsertionPoint(numBelowToDelete) = sampleIndicesToDropArray.search(ir.index.toInt)
      val index = ir.index - numBelowToDelete
      val vecArray = ir.vector.toArray
      val filteredArray = sampleIndicesToTakeArray.map(i => vecArray(i))
      new IndexedRow(index, Vectors.dense(filteredArray))
    })

    new KinshipMatrix(new IndexedRowMatrix(filteredRowsAndCols), filteredSamplesIds)
  }
}
