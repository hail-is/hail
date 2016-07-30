package org.broadinstitute.hail.driver

import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods.ImputeSexPlink
import org.kohsuke.args4j.{Option => Args4jOption}

object ImputeSex extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = false, name = "-m", aliases = Array("--maf-threshold"),
      usage = "Minimum minor allele frequency threshold")
    var mafThreshold: Double = 0.0

    @Args4jOption(required = false, name = "-i", aliases = Array("--include-par"),
      usage = "Include Pseudoautosomal regions")
    var includePAR: Boolean = false

    @Args4jOption(required = false, name = "-x", aliases = Array("--female-threshold"),
      usage = "Samples are called females if F < femaleThreshold (Default = 0.2)")
    var fFemaleThreshold: Double = 0.2

    @Args4jOption(required = false, name = "-y", aliases = Array("--male-threshold"),
      usage = "Samples are called males if F > maleThreshold (Default = 0.8)")
    var fMaleThreshold: Double = 0.8

    @Args4jOption(required = false, name = "-p", aliases = Array("--pop-freq"),
      usage = "Use a variant annotation for estimate of MAF rather than computing from the data")
    var popFreq: String = _
  }

  def newOptions = new Options

  def name = "imputesex"

  def description = "Impute sex of samples"

  def requiresVDS = true

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {

    val result = ImputeSexPlink(state.vds,
      options.mafThreshold,
      options.includePAR,
      options.fMaleThreshold,
      options.fFemaleThreshold,
      Option(options.popFreq))

    val signature = ImputeSexPlink.schema

    val (newSAS, insertSexCheck) = state.vds.saSignature.insert(signature, "imputesex")
    val newSampleAnnotations = state.vds.sampleIdsAndAnnotations
      .map { case (s, sa) =>
        insertSexCheck(sa, result.get(s))
      }

    state.copy(
      vds = state.vds.copy(
        sampleAnnotations = newSampleAnnotations,
        saSignature = newSAS)
    )
  }
}
