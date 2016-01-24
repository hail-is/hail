package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr
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
    val sc = state.sc
    val vds = state.vds

    if (!options.keep && !options.remove)
      fatal(name + ": one of `--keep' or `--remove' required")

    val p = new expr.Parser()
    val e = p.parseAll(p.expr, options.condition).get
    val symTab = Map(
      "v" ->(0, expr.TVariant),
      "va" ->(1, vds.metadata.variantAnnotationSignatures.toExprType),
      "s" ->(2, expr.TSample),
      "sa" ->(3, vds.metadata.sampleAnnotationSignatures.toExprType),
      "g" ->(4, expr.TGenotype))
    e.typecheck(symTab)

    val keep = options.keep
    val sampleIdsBc = sc.broadcast(vds.sampleIds)
    val sampleAnnotationsBc = sc.broadcast(vds.metadata.sampleAnnotations)
    val a = new Array[Any](5)
    val f: () => Any = e.eval((symTab, a))

    val noCall = Genotype(-1, (0, 0), 0, null)
    val newVDS = vds.mapValuesWithAll(
      (v: Variant, va: Annotations, s: Int, g: Genotype) => {
        a(0) = v
        a(1) = va
        a(2) = sampleIdsBc.value(s)
        a(3) = sampleAnnotationsBc.value(s)
        a(4) = g
        if (Filter.keepThisAny(f(), keep))
          g
        else
          noCall
      })
    state.copy(vds = newVDS)
  }
}
