package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable.ArrayBuffer

object FilterSamplesExpr extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-c", aliases = Array("--condition"),
      usage = "Filter expression involving `s' (sample) and `sa' (sample annotations)")
    var condition: String = _

    @Args4jOption(required = false, name = "--keep", usage = "Keep only listed samples in current dataset")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove listed samples from current dataset")
    var remove: Boolean = false
  }

  def newOptions = new Options

  def name = "filtersamples expr"

  def description = "Filter samples in current dataset using the Hail expression language"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if ((options.keep && options.remove)
      || (!options.keep && !options.remove))
      fatal("one `--keep' or `--remove' required, but not both")

    val keep = options.keep
    val sas = vds.saSignature
    val symTab = Map(
      "s" ->(0, TSample),
      "sa" ->(1, sas))
    val a = new ArrayBuffer[Any]()
    for (_ <- symTab)
      a += null
    val f: () => Option[Boolean] = Parser.parse[Boolean](options.condition, symTab, a, TBoolean)
    val p = (s: String, sa: Annotation) => {
      a(0) = s
      a(1) = sa
      Filter.keepThis(f(), keep)
    }

    state.copy(vds = vds.filterSamples(p))
  }
}
