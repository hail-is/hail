package is.hail.methods

import java.io.DataOutputStream

import breeze.linalg.SparseVector
import is.hail.HailContext
import is.hail.annotations.Annotation
import is.hail.expr.types._
import is.hail.utils._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}

import scala.collection.Searching._

/**
  * Represents a KinshipMatrix. Entry (i, j) encodes the relatedness of the ith and jth samples in sampleIds.
  */
case class KinshipMatrix(hc: HailContext, sampleSignature: Type, matrix: IndexedRowMatrix, sampleIds: Array[Annotation], numVariantsUsed: Long) extends ExportableMatrix {
  assert(matrix.numCols().toInt == matrix.numRows().toInt && matrix.numCols().toInt == sampleIds.length)

  def requireSampleTString(method: String) {
    if (!sampleSignature.isOfType(TString()))
      fatal(s"in $method: key (sample) schema must be String, but found: $sampleSignature")
  }

  /**
    * Filters list of samples based on predicate, and removes corresponding rows and columns from the matrix.
    * @param pred The predicate that decides whether a sample is kept.
    */
  def filterSamples(pred: (Annotation => Boolean)): KinshipMatrix = {
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
      IndexedRow(index, Vectors.dense(filteredArray))
    })

    KinshipMatrix(hc, sampleSignature, new IndexedRowMatrix(filteredRowsAndCols), filteredSamplesIds, numVariantsUsed)
  }

  /**
    * Writes out the matrix as a TSV file, with the sample names as a header on the first line.
    *
    * @param output The path to the output file.
    */
  def exportTSV(output: String) {
    require(output.endsWith(".tsv"), "Kinship matrix output must end in '.tsv'")
    export(output, "\t", Some(sampleIds.mkString("\t")), ExportType.CONCATENATED)
  }

  def exportRel(output: String) {
    exportLowerTriangle(output, "\t", None, ExportType.CONCATENATED)
  }

  def exportGctaGrm(output: String) {
    val nVars = numVariantsUsed //required to avoid serialization error.
    prepareMatrixForExport(matrix).rows
    .mapPartitions{ (itr: Iterator[IndexedRow]) =>
      val sb = new StringBuilder
      itr.foreach{ (row: IndexedRow) =>
        val i = row.index
        val arr = row.vector.toArray
        var j = 0
        while (j <= i) {
          sb.append(s"${ i + 1 }\t${ j + 1 }\t$nVars\t${ arr(j) }")
          if (j < i) {
            sb.append("\n")
          }
          j += 1
        }
        sb += '\n'
      }
      sb.lines
    }.writeTable(output, hc.tmpDir)
  }

  def exportGctaGrmBin(output: String, nFile: Option[String] = None) {
    prepareMatrixForExport(matrix).rows
      .map { (row: IndexedRow) =>
        val i = row.index
        val arr = row.vector.toArray
        val result = new Array[Byte](4 * i.toInt + 4)
        var j = 0
        while (j <= i) {
          val bits: Int = java.lang.Float.floatToRawIntBits(arr(j).toFloat)
          result(j * 4) = (bits & 0xff).toByte
          result(j * 4 + 1) = ((bits >> 8) & 0xff).toByte
          result(j * 4 + 2) = ((bits >> 16) & 0xff).toByte
          result(j * 4 + 3) = (bits >> 24).toByte
          j += 1
        }
        result
      }.saveFromByteArrays(output, hc.tmpDir)

    nFile.foreach { f =>
      hc.sc.hadoopConfiguration.writeDataFile(f) { s =>
        for (_ <- 0 until sampleIds.length * (sampleIds.length + 1) / 2)
          writeFloatLittleEndian(s, numVariantsUsed.toFloat)
      }
    }
  }

  def exportIdFile(idFile: String) {
    requireSampleTString("export id file")

    hc.sc.hadoopConfiguration.writeTextFile(idFile) { s =>
      for (id <- sampleIds) {
        s.write(id.asInstanceOf[String])
        s.write("\t")
        s.write(id.asInstanceOf[String])
        s.write("\n")
      }
    }
  }

  private def writeFloatLittleEndian(s: DataOutputStream, f: Float) {
    val bits: Int = java.lang.Float.floatToRawIntBits(f)
    s.write(bits & 0xff)
    s.write((bits >> 8) & 0xff)
    s.write((bits >> 16) & 0xff)
    s.write(bits >> 24)
  }
}
