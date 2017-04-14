package is.hail.methods

import breeze.linalg.SparseVector
import is.hail.HailContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}

import scala.collection.Searching._
import is.hail.utils._

/**
  * Represents a KinshipMatrix. Entry (i, j) encodes the relatedness of the ith and jth samples in sampleIds.
  */
class KinshipMatrix(val hc: HailContext, val matrix: IndexedRowMatrix, val sampleIds: Array[String]) {
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

    new KinshipMatrix(hc, new IndexedRowMatrix(filteredRowsAndCols), filteredSamplesIds)
  }

  /**
    * Writes out the matrix as a TSV file, with the sample names as a header on the first line.
    *
    * @param output The path to the output file.
    */
  def exportTSV(output: String) {
    require(output.endsWith(".tsv"), "Kinship matrix output must end in '.tsv'")
    prepareMatrixForExport(matrix).rows.map(ir => ir.vector.toArray.mkString("\t"))
      .writeTable(output, hc.tmpDir, Some(sampleIds.mkString("\t")))
  }

  /**
    * Creates an IndexedRowMatrix whose backing RDD is sorted by row index and has an entry for every row index.
    *
    * @param matToComplete The matrix to be completed.
    * @return The completed matrix.
    */
  private def prepareMatrixForExport(matToComplete: IndexedRowMatrix): IndexedRowMatrix = {
    val zeroVector = SparseVector.zeros[Double](sampleIds.length)

    new IndexedRowMatrix(matToComplete
      .rows
      .map(x => (x.index, x.vector))
      .rightOuterJoin(hc.sc.parallelize(0L until sampleIds.length).map(x => (x, ())))
      .map {
        case (idx, (Some(v), _)) => IndexedRow(idx, v)
        case (idx, (None, _)) => IndexedRow(idx, zeroVector)
      }
      .sortBy(_.index))
  }
}
