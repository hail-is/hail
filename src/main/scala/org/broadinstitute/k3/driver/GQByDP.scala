package org.broadinstitute.k3.driver

import org.broadinstitute.k3.methods.GQByDPBins
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.k3.Utils._
import sys.process._
import scala.language.postfixOps

object GQByDP extends Command {

  def name = "gqbydp"
  def description = "Compute percent GQ >= 20 by DP bins per sample"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _

    @Args4jOption(name = "--plot", usage = "Plot output")
    var plot: Boolean = false
  }
  def newOptions = new Options

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val nBins = GQByDPBins.nBins
    val binStep = GQByDPBins.binStep
    val firstBinLow = GQByDPBins.firstBinLow
    val gqbydp = GQByDPBins(vds)

    writeTextFile(options.output, state.hadoopConf) { s =>
      s.write("sample")
      for (b <- 0 until nBins)
        s.write("\t" + GQByDPBins.binLow(b) + "-" + GQByDPBins.binHigh(b))

      s.write("\n")

      for (i <- vds.sampleIds.indices) {
        s.write(vds.sampleIds(i))
        for (b <- 0 until GQByDPBins.nBins) {
          gqbydp.get((i, b)) match {
            case Some(percentGQ) => s.write("\t" + percentGQ)
            case None => s.write("\tNA")
          }
        }
        s.write("\n")
      }
    }

    if (options.plot) {
      "Rscript " + state.installDir + "/scripts/Plot_gq20bydp.R " + options.output !
    }

    state
  }
}
