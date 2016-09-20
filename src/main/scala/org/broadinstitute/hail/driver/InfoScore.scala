package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.stats.imputeInfoScore

object InfoScore extends Command {

  class Options extends BaseOptions

  def newOptions = new Options

  def name = "infoscore"

  def description = "Compute info scores per variant from dosage data."

  def supportsMultiallelic = false

  def requiresVDS = true

  val signature = TStruct(
    "score" -> TDouble,
    "nIncluded" -> TInt
  )

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (!vds.isDosage)
      fatal("Genotypes must be dosages in order to compute an info score.")

    val (newVAS, insertInfoScore) = state.vds.saSignature.insert(signature, "infoscore")

    state.copy(
      vds = vds.mapAnnotations { case (v, va, gs) =>
        val (infoScore, nIncluded) = imputeInfoScore(gs)
        insertInfoScore(va, Some(Annotation(infoScore.orNull, nIncluded)))
      }
        .copy(vaSignature = newVAS)
    )
  }
}

