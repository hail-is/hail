package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

object FilterVariantsIntervals extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-i", aliases = Array("--input"),
      usage = "Path to interval list file")
    var input: String = _

    @Args4jOption(required = false, name = "--keep", usage = "Keep variants matching condition")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove variants matching condition")
    var remove: Boolean = false

  }

  def newOptions = new Options

  def name = "filtervariants intervals"

  def description = "Filter variants in current dataset with an interval list"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if ((options.keep && options.remove)
      || (!options.keep && !options.remove))
      fatal("one `--keep' or `--remove' required, but not both")

    val cond = options.input
    val keep = options.keep
    val ilist = IntervalList.read(options.input, state.hadoopConf)
    val ilistBc = state.sc.broadcast(ilist)
    val p = (v: Variant, va: Annotation, _: Iterable[Genotype]) =>
      Filter.keepThis(ilistBc.value.contains(v.contig, v.start), keep)

    state.copy(vds = vds.filterVariants(p))
  }
}
