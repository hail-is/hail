package org.broadinstitute.hail.driver

import org.broadinstitute.hail.expr.{TInt, TDouble, TStruct}
import org.broadinstitute.hail.methods.ImputeGenderPlink
import org.kohsuke.args4j.{Option => Args4jOption}

object ImputeGender extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = false, name = "-m", aliases = Array("--mafthreshold"), usage = "Minimum minor allele frequency threshold")
    var mafThreshold: Double = 0.0

    @Args4jOption(required = false, name = "-e", aliases = Array("--excludepar"), usage = "Exclude Pseudoautosomal regions")
    var excludePAR: Boolean = false

    @Args4jOption(required = false, name = "-x", aliases = Array("--femalethreshold"), usage = "Samples are called females if F < femaleThreshold (Default = 0.2)")
    var fFemaleThreshold: Double = 0.2

    @Args4jOption(required = false, name = "-y", aliases = Array("--malethreshold"), usage = "Samples are called males if F > maleThreshold (Default = 0.8)")
    var fMaleThreshold: Double = 0.8
  }

  def newOptions = new Options

  def name = "imputegender"

  def description = "Impute gender of samples"

  def requiresVDS = true

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {

    val result = ImputeGenderPlink.imputeGender(state.vds, options.mafThreshold, options.excludePAR)
      .result(options.fFemaleThreshold, options.fMaleThreshold).collect().toMap

    val signature = TStruct("F" -> TDouble, "E" -> TDouble, "O" -> TDouble, "N" -> TInt, "T" -> TInt, "imputedSex" -> TInt)

    val (newSAS, insertSexCheck) = state.vds.saSignature.insert(signature, "imputegender")
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
