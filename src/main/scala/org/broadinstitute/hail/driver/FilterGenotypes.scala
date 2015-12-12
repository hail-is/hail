package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.variant.{Variant, Genotype, Sample}
import org.kohsuke.args4j.{Option => Args4jOption}

object FilterGenotypes extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "--keep", usage = "Keep only listed samples in current dataset")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove listed samples from current dataset")
    var remove: Boolean = false

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Filter condition expression")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "filtergenotypes"

  def description = "Filter genotypes in current dataset"

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val vas: AnnotationSignatures = state.vds.metadata.variantAnnotationSignatures
    val sas: AnnotationSignatures = state.vds.metadata.sampleAnnotationSignatures
    val sa = state.vds.metadata.sampleAnnotations

    if (!options.keep && !options.remove)
      fatal(name + ": one of `--keep' or `--remove' required")

    val p: ((Variant, AnnotationData) => ((Int, Sample, Genotype) => Boolean)) = try {
      val cf = new FilterGenotypeCondition(options.condition, vas, sas, sa)
      cf.typeCheck()
      cf.apply(sa)
    }

    val sampleIdsBc = state.sc.broadcast(state.vds.sampleIds)
    val localKeep = options.keep
    val localRemove = options.remove
    //FIXME put keep/remove logic here
    val newVDS = vds.mapValuesWithAll((v: Variant, va: AnnotationData, s: Int, g: Genotype) =>
      if (p(v, va)(s, Sample(sampleIdsBc.value(s)), g)) {
        if (localKeep) g else Genotype(-1, (0, 0), 0, null)
      }
      else {
        if (localRemove) g else Genotype(-1, (0, 0), 0, null)
      })
    state.copy(vds = newVDS)
  }
}
