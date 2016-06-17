package org.broadinstitute.hail.driver

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

    val (nVariants, nCalledOption) = if (options.genotypes)
      (vds.rdd.count(), None)
    else {
      val (nVar, nCalled) = vds.rdd.map { case (v, va, gs) =>
        (1L, gs.count(_.isCalled).toLong)
      }.fold((0L, 0L)) { (comb, x) =>
        (comb._1 + x._1, comb._2 + x._2)
      }
      (nVar, Some(nCalled))
    }

    val sb = new StringBuilder()

    sb.append("count:")
    sb.append(s"  nSamples = ${vds.nSamples}")
    sb.append(s"  nVariants = $nVariants")

    nCalledOption.foreach { nCalled =>
      val nGenotypes = nVariants * vds.nSamples
      val callRate = divOption(nCalled, nGenotypes)

      sb.append(s"nCalled = $nCalled")
      sb.append(s"callRate = ${callRate.map(r => (r * 100).formatted("%.3f%%")).getOrElse("NA")}")
    }

    info(sb.result())
    state
  }
}
