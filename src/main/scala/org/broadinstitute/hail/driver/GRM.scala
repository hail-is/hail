package org.broadinstitute.hail.driver

import java.io.DataOutputStream
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.methods.ToStandardizedIndexedRowMatrix
import org.kohsuke.args4j.{Option => Args4jOption}

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

    val grm = mat.computeGramianMatrix()

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

    options.format match {
      case "rel" =>
        state.hadoopConf.writeTextFile(options.output) { s =>
          for (i <- 0 until nSamples) {
            for (j <- 0 to i) {
              if (j > 0)
                s.write("\t")
              s.write(grm(i, j).toString)
            }
            s.write("\n")
          }
        }

      case "gcta-grm" =>
        state.hadoopConf.writeTextFile(options.output) { s =>
          for (i <- 0 until nSamples)
            for (j <- 0 to i)
              s.write(s"${i + 1}\t${j + 1}\t$nVariants\t${grm(i, j)}\n")
        }

      case "gcta-grm-bin" =>
        state.hadoopConf.writeDataFile(options.output) { s =>
          for (i <- 0 until nSamples)
            for (j <- 0 to i)
              writeFloatLittleEndian(s, grm(i, j).toFloat)
        }

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
