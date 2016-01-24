package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr.TVariant
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations._
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.expr

object FilterVariants extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "--keep", usage = "Keep variants matching condition")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove variants matching condition")
    var remove: Boolean = false

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Filter condition: expression or .interval_list file")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "filtervariants"

  def description = "Filter variants in current dataset"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (!options.keep && !options.remove)
      fatal(name + ": one of `--keep' or `--remove' required")

    val cond = options.condition
    val vas = vds.metadata.variantAnnotationSignatures
    val keep = options.keep
    val p: (Variant, Annotations) => Boolean = cond match {
      case f if f.endsWith(".interval_list") =>
        val ilist = IntervalList.read(options.condition, state.hadoopConf)
        (v: Variant, va: Annotations) => Filter.keepThis(ilist.contains(v.contig, v.start), keep)
      case c: String =>
        val parser = new expr.Parser()
        val e = parser.parseAll(parser.expr, c).get
        val symTab = Map(
          "v" -> (0, TVariant),
          "va" -> (1, vds.metadata.variantAnnotationSignatures.toExprType))
        e.typecheck(symTab)
        val a = new Array[Any](2)
        val f: () => Any = e.eval(symTab, a)
        (v: Variant, va: Annotations) => {
          a(0) = v
          a(1) = va
          Filter.keepThisAny(f(), keep)
        }
    }

    state.copy(vds = vds.filterVariants(p))
  }
}
