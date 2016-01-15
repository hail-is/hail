package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.variant.{VariantDataset, Variant, Genotype, Sample}
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

    if (!options.keep && !options.remove)
      fatal(name + ": one of `--keep' or `--remove' required")

    val p: ((Variant, Annotations) => ((Int, Genotype) => Boolean)) = {
      val cf = new FilterGenotypeCondition(options.condition, vds.metadata)
      cf.typeCheck()

      val keep = options.keep
      (v: Variant, va: Annotations) => {
        val h = cf(v,va) _
        (sIndex: Int, g: Genotype) => Filter.keepThis(h(sIndex, g), keep)
      }
    }

    val newVDS = vds.mapValuesWithPartialApplication(
      (v: Variant, va: Annotations) =>
        (s: Int, g: Genotype) =>
          if (p(v, va)(s, g)) 
            g
          else
            Genotype(-1, (0, 0), 0, null))
    state.copy(vds = newVDS)
  }
}
