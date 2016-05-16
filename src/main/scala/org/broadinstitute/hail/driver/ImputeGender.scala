package org.broadinstitute.hail.driver

import org.broadinstitute.hail.expr.{TInt, TDouble, TStruct}
import org.broadinstitute.hail.methods.ImputeGenderPlink
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._


object ImputeGender extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = false, name = "-m", aliases = Array("--mafthreshold"), usage = "Minimum minor allele frequency threshold")
    var mafThreshold: Double = 0.0

    @Args4jOption(required = false, name = "-e", aliases = Array("--excludepar"), usage = "Exclude Pseudoautosomal regions")
    var excludePAR: Boolean = false
  }

  def newOptions = new Options

  def name = "imputegender"

  def description = "Impute gender of samples"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {

    val result = ImputeGenderPlink.imputeGender(state.vds, options.mafThreshold, options.excludePAR).result.collect().toMap

    val signature = TStruct("F" -> TDouble, "E" -> TDouble, "O" -> TDouble, "N" -> TInt, "T" -> TInt, "imputedSex" -> TInt)

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
