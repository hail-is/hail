package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

object FilterVariants extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "--keep", usage = "Keep variants matching condition")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove variants matching condition")
    var remove: Boolean = false

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Filter condition: .interval_list file, isSNP or isIndel")
    var condition: String = _
  }
  def newOptions = new Options

  def name = "filtervariants"
  def description = "Filter variants in current dataset"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (!options.keep && !options.remove)
      fatal(name + ": one of `--keep' or `--remove' required")

    val p: (Variant) => Boolean = options.condition match {
      case f if f.endsWith(".interval_list") =>
        val ilist = IntervalList.read(options.condition)
        (v) => ilist.contains(v.contig, v.start)
      case "isSNP" =>
        _.isSNP
      case "isIndel" =>
        _.isIndel
      case _ =>
        fatal("unknown filter condition")
    }

    val newVDS = vds.filterVariants(if (options.keep)
      p
    else
      (v: Variant) => !p(v))
    
    state.copy(vds = newVDS)
  }
}

