package org.broadinstitute.hail.driver

import org.broadinstitute.hail.methods.GQByDPBins
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.utils._
import sys.process._
import scala.language.postfixOps

object GQByDP extends Command {

  def name = "gqbydp"
  def description = "Compute percent GQ >= 20 by DP bins per sample"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }
  def newOptions = new Options

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val nBins = GQByDPBins.nBins
    val binStep = GQByDPBins.binStep
    val firstBinLow = GQByDPBins.firstBinLow
    val gqbydp = GQByDPBins(vds)

    state.hadoopConf.writeTextFile(options.output) { s =>
      s.write("sample")
      for (b <- 0 until nBins)
        s.write("\t" + GQByDPBins.binLow(b) + "-" + GQByDPBins.binHigh(b))

      s.write("\n")

      for (sample <- vds.sampleIds) {
        s.write(sample)
        for (b <- 0 until GQByDPBins.nBins) {
          gqbydp.get((sample, b)) match {
            case Some(percentGQ) => s.write("\t" + percentGQ)
            case None => s.write("\tNA")
          }
        }
        s.write("\n")
      }
    }

    state
  }
}
