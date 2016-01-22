package org.broadinstitute.hail.driver

import htsjdk.variant.variantcontext.Genotype
import org.broadinstitute.hail.annotations.AnnotationData
import org.broadinstitute.hail.variant.Variant

object Count extends Command {

  class Options extends BaseOptions

  def newOptions = new Options

  def name = "count"

  def description = "Print number of samples, variants, and called genotypes in current dataset"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val nVariants = vds.nVariants
    val nLocalSamples = vds.nLocalSamples
    val called = vds.rdd.aggregate(0L)({ case (count, (v, va, gs)) =>
      count + gs.count(_.isCalled)},
      _ + _)
    val nGenotypes = nVariants.toLong * nLocalSamples.toLong
    val callRate = called.toDouble / nGenotypes * 100

    println("  nSamples = " + vds.nSamples)
    println("  nLocalSamples = " + nLocalSamples)
    println("  nVariants = " + vds.nVariants)
    println(s"  nCalled = $called")
    println(s"  callRate = ${callRate.formatted("%.3f")}%")
    state
  }
}
