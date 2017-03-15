package is.hail.methods

import java.io.DataOutputStream

import breeze.linalg.SparseVector
import is.hail.stats.ToHWENormalizedIndexedRowMatrix
import is.hail.utils._
import is.hail.variant.VariantDataset
import org.apache.spark.mllib.linalg.distributed.IndexedRow

object GRM {
  def writeFloatLittleEndian(s: DataOutputStream, f: Float) {
    val bits: Int = java.lang.Float.floatToRawIntBits(f)
    s.write(bits & 0xff)
    s.write((bits >> 8) & 0xff)
    s.write((bits >> 16) & 0xff)
    s.write(bits >> 24)
  }

  def apply(vds: VariantDataset, path: String, format: String,
    idFile: Option[String] = None, nFile: Option[String] = None) {
    val (variants, mat) = ToHWENormalizedIndexedRowMatrix(vds)

    val nSamples = vds.nSamples
    assert(nSamples == mat.numCols())
    val nVariants = mat.numRows() // mat cached

    val bmat = mat.toBlockMatrix().cache()
    val grm = bmat.transpose.multiply(bmat)

    assert(grm.numCols == nSamples
      && grm.numRows == nSamples)

    idFile.foreach { f =>
      vds.sparkContext.hadoopConfiguration.writeTextFile(f) { s =>
        for (id <- vds.sampleIds) {
          s.write(id)
          s.write("\t")
          s.write(id)
          s.write("\n")
        }
      }
    }

    if (format != "gcta-grm-bin"
      && nFile.isDefined)
      warn(s"format $format: ignoring `--N-file'")

    val zeroVector = SparseVector.zeros[Double](nSamples)

    val indexSortedRowMatrix = grm.toIndexedRowMatrix
      .rows
      .map(x => (x.index, x.vector))
      .rightOuterJoin(vds.sparkContext.parallelize(0L until nSamples).map(x => (x, ())))
      .map {
        case (idx, (Some(v), _)) => IndexedRow(idx, v)
        case (idx, (None, _)) => IndexedRow(idx, zeroVector)
      }
      .sortBy(_.index)

    format match {
      case "rel" =>
        indexSortedRowMatrix
          .map { (row: IndexedRow) =>
            val i = row.index
            val arr = row.vector.toArray
            val sb = new StringBuilder
            var j = 0
            while (j <= i) {
              if (j > 0)
                sb += '\t'
              sb.append(arr(j))
              j += 1
            }
            sb.toString()
          }
          .writeTable(path, vds.hc.tmpDir)

      case "gcta-grm" =>
        indexSortedRowMatrix
          .map { (row: IndexedRow) =>
            val i = row.index
            val arr = row.vector.toArray
            val sb = new StringBuilder
            var j = 0
            while (j <= i) {
              sb.append(s"${ i + 1 }\t${ j + 1 }\t$nVariants\t${ arr(j) }")
              if (j < i) {
                sb.append("\n")
              }
              j += 1
            }
            sb.toString()
          }
          .writeTable(path, vds.hc.tmpDir)

      case "gcta-grm-bin" =>
        indexSortedRowMatrix
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
          }
          .saveFromByteArrays(path, vds.hc.tmpDir)

        nFile.foreach { f =>
          vds.sparkContext.hadoopConfiguration.writeDataFile(f) { s =>
            for (_ <- 0 until nSamples * (nSamples + 1) / 2)
              writeFloatLittleEndian(s, nVariants.toFloat)
          }
        }

      case _ => fatal(s"unknown output format `$format'")
    }
  }
}