package org.broadinstitute.hail.driver

import java.io.DataOutputStream

import breeze.linalg.DenseMatrix
import org.apache.hadoop.io.{BytesWritable, IntWritable, LongWritable, NullWritable}
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.methods.ToStandardizedIndexedRowMatrix
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object GRM extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-f", aliases = Array("--format"), usage = "Output format: one of rel, gcta-grm, gcta-grm-bin")
    var format: String = _

    @Args4jOption(required = false, name = "--id-file", usage = "ID file")
    var idFile: String = _

    @Args4jOption(required = false, name = "--N-file", usage = "N file, for gcta-grm-bin only")
    var nFile: String = _

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _

  }

  def newOptions = new Options

  def name = "grm"

  def description = "Compute the Genetic Relatedness Matrix (GRM)"

  def supportsMultiallelic = true

  def requiresVDS = true

  def writeFloatLittleEndian(s: DataOutputStream, f: Float) {
    val bits: Int = java.lang.Float.floatToRawIntBits(f)
    s.write(bits & 0xff)
    s.write((bits >> 8) & 0xff)
    s.write((bits >> 16) & 0xff)
    s.write(bits >> 24)
  }

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val (variants, mat) = ToStandardizedIndexedRowMatrix(vds)

    val nSamples = vds.nSamples
    assert(nSamples == mat.numCols())
    val nVariants = mat.numRows() // mat cached

    val bmat = mat.toBlockMatrix().cache()
    val grm = bmat.transpose.multiply(bmat)

    assert(grm.numCols == nSamples
      && grm.numRows == nSamples)

    if (options.idFile != null) {
      state.hadoopConf.writeTextFile(options.idFile) { s =>
        for (id <- vds.sampleIds) {
          s.write(id)
          s.write("\t")
          s.write(id)
          s.write("\n")
        }
      }
    }

    if (options.format != "gcta-grm-bin"
      && options.nFile != null)
      warn(s"format ${options.format}: ignoring `--N-file'")

    val tmpFileName = state.hadoopConf.getTemporaryFile(HailConfiguration.tmpDir)

    options.format match {
      case "rel" =>
        grm.toIndexedRowMatrix()
          .rows
          .map { (row: IndexedRow) =>
            val i = row.index
            val arr = row.vector.toArray
            val sb = new StringBuilder
            var j = 0
            while (j <= i) {
              if (j > 0)
                sb.append("\t")
              sb.append(arr(j))
              j += 1
            }
            sb.toString()
          }
          .saveAsTextFile(tmpFileName)

        state.hadoopConf.copyMerge(tmpFileName, options.output, true, false)

      case "gcta-grm" =>
        grm.toIndexedRowMatrix()
          .rows
          .map { (row: IndexedRow) =>
            val i = row.index
            val arr = row.vector.toArray
            val sb = new StringBuilder
            var j = 0
            while (j <= i) {
              sb.append(s"${i + 1}\t${j + 1}\t$nVariants\t${arr(j)}")
              if (j < i) { sb.append("\n") }
              j += 1
            }
            sb.toString()
          }
          .saveAsTextFile(tmpFileName)

        state.hadoopConf.copyMerge(tmpFileName, options.output, true, false)

      case "gcta-grm-bin" =>
        grm.toIndexedRowMatrix()
          .rows
          .sortBy(row => row.index)
          .map { (row: IndexedRow) =>
            val i = row.index
            val arr = row.vector.toArray
            val ab = mutable.ArrayBuilder.make[Byte]()
            var j = 0
            while (j <= i) {
              val bits: Int = java.lang.Float.floatToRawIntBits(arr(j).toFloat)
              ab += (bits & 0xff).toByte
              ab += ((bits >> 8) & 0xff).toByte
              ab += ((bits >> 16) & 0xff).toByte
              ab += (bits >> 24).toByte
              j += 1
            }
            ab.result()
          }
          .saveFromByteArrays(options.output)

        if (options.nFile != null) {
          state.hadoopConf.writeDataFile(options.nFile) { s =>
            for (_ <- 0 until nSamples * (nSamples + 1) / 2)
              writeFloatLittleEndian(s, nVariants.toFloat)
          }
        }

      case _ => fatal(s"unknown output format `${options.format}")
    }

    state
  }

}
