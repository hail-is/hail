package org.broadinstitute.hail.driver

import java.text.NumberFormat
import java.util.Locale

import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object Count extends Command {

  class Options extends BaseOptions {
    @Args4jOption(name = "-g", aliases = Array("--genotypes"),
      usage = "Calculate genotype call rate")
    var genotypes: Boolean = false
  }

  def newOptions = new Options

  def name = "count"

  def description = "Print number of samples, variants, and called genotypes in current dataset"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val (nVariants, nCalledOption) = if (options.genotypes) {
      val (nVar, nCalled) = vds.rdd.map { case (v, (va, gs)) =>
        (1L, gs.count(_.isCalled).toLong)
      }.fold((0L, 0L)) { (comb, x) =>
        (comb._1 + x._1, comb._2 + x._2)
      }
      (nVar, Some(nCalled))
    } else (vds.rdd.count(), None)

    val sb = new StringBuilder()

    val formatter = NumberFormat.getNumberInstance(Locale.US)
    def format(a: Any) = "%15s".format(formatter.format(a))


    sb.append("count:\n")
    sb.append(s"  nSamples   ${format(vds.nSamples)}\n")
    sb.append(s"  nVariants  ${format(nVariants)}")

    nCalledOption.foreach { nCalled =>
      val nGenotypes = nVariants * vds.nSamples
      val callRate = divOption(nCalled, nGenotypes)
      sb += '\n'
      sb.append(s"  nCalled    ${format(nCalled)}\n")
      sb.append(s"  callRate   ${"%15s".format(callRate.map(r => (r * 100).formatted("%.3f%%")).getOrElse("NA"))}")
    }

    info(sb.result())
    state
  }
}