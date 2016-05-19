package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._

object Count extends Command {

  class Options extends BaseOptions

  def newOptions = new Options

  def name = "count"

  def description = "Print number of samples, variants, and called genotypes in current dataset"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val (nVariants, nCalled) =
      vds.rdd.map { case (v, va, gs) =>
        (1L, gs.count(_.isCalled).toLong)
      }.fold((0L, 0L)) { (comb, x) =>
        (comb._1 + x._1, comb._2 + x._2)
      }

    val nGenotypes = nVariants * vds.nSamples
    val callRate = divOption(nCalled, nGenotypes)

    info(
      s"""count:
          |  nSamples = ${vds.nSamples}
          |  nVariants = $nVariants
          |  nCalled = $nCalled
          |  callRate = ${callRate.map(r => (r * 100).formatted("%.3f%%")).getOrElse("NA")}""".stripMargin)

    state
  }
}
