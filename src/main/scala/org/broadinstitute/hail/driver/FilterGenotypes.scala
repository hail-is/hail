package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.methods._
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

    if (!options.keep && !options.remove)
      fatal(name + ": one of `--keep' or `--remove' required")

    val p: (Variant, Sample, Genotype) => Boolean = try {
      val cf = new FilterGenotypeCondition(options.condition)
      cf.typeCheck()
      if (options.keep)
        cf.apply
      else
        (v: Variant, s: Sample, g: Genotype) => !cf(v, s, g)
    } catch {
      case e: scala.tools.reflect.ToolBoxError =>
        /* e.message looks like:
           reflective compilation has failed:

           ';' expected but '.' found. */
        fatal("parse error in condition: " + e.message.split("\n").last)
    }

    val sampleIdsBc = state.sc.broadcast(state.vds.sampleIds)

    val newVDS = vds.mapValuesWithKeys((v: Variant, s: Int, g: Genotype) =>
      if (p(v, Sample(sampleIdsBc.value(s)), g))
        g
      else
        Genotype(-1, (0, 0), 0, null))

    state.copy(vds = newVDS)
  }
}
