package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable.ArrayBuffer

object FilterVariantsExpr extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-c", aliases = Array("--condition"),
      usage = "Filter expression involving `v' (variant) and `va' (variant annotations)")
    var condition: String = _

    @Args4jOption(required = false, name = "--keep", usage = "Keep variants matching condition")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove variants matching condition")
    var remove: Boolean = false

  }

  def newOptions = new Options

  def name = "filtervariants expr"

  def description = "Filter variants in current dataset using the Hail expression language"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if ((options.keep && options.remove)
      || (!options.keep && !options.remove))
      fatal("one `--keep' or `--remove' required, but not both")

    val vas = vds.vaSignature
    val cond = options.condition
    val keep = options.keep

    val symTab = Map(
      "v" ->(0, TVariant),
      "va" ->(1, vas))
    val a = new ArrayBuffer[Any]()
    for (_ <- symTab)
      a += null
    val f: () => Option[Boolean] = Parser.parse[Boolean](cond, symTab, a, TBoolean)
    val p = (v: Variant, va: Annotation) => {
      a(0) = v
      a(1) = va
      Filter.keepThis(f(), keep)
    }


    state.copy(vds = vds.filterVariants(p))
  }
}
