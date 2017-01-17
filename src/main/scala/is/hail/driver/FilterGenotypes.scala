package is.hail.driver

import is.hail.utils._
import is.hail.annotations._
import is.hail.expr._
import is.hail.methods._
import is.hail.variant._
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

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val sc = state.sc
    val vds = state.vds
    val vas = vds.vaSignature
    val sas = vds.saSignature

    if (!(options.keep ^ options.remove))
      fatal("either `--keep' or `--remove' required, but not both")

    val keep = options.keep

    val cond = options.condition
    val symTab = Map(
      "v" ->(0, TVariant),
      "va" ->(1, vas),
      "s" ->(2, TSample),
      "sa" ->(3, sas),
      "g" ->(4, TGenotype),
      "global" -> (5, vds.globalSignature) )


    val ec = EvalContext(symTab)
    ec.set(5, vds.globalAnnotation)
    val f: () => Option[Boolean] = Parser.parseTypedExpr[Boolean](cond, ec)

    val sampleIdsBc = vds.sampleIdsBc
    val sampleAnnotationsBc = vds.sampleAnnotationsBc

    (vds.sampleIds, vds.sampleAnnotations).zipped.map((_, _))

    val noCall = Genotype()
    val newVDS = vds.mapValuesWithAll(
      (v: Variant, va: Annotation, s: String, sa: Annotation, g: Genotype) => {
        ec.setAll(v, va, s, sa, g)

        if (Filter.keepThis(f(), keep))
          g
        else
          noCall
      })
    state.copy(vds = newVDS)
  }
}
