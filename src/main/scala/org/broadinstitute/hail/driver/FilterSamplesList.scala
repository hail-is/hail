package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.methods._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.io.Source

object FilterSamplesList extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-i", aliases = Array("--input"),
      usage = "Path to sample list file")
    var input: String = _

    @Args4jOption(required = false, name = "--keep", usage = "Keep only listed samples in current dataset")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove listed samples from current dataset")
    var remove: Boolean = false
  }

  def newOptions = new Options

  def name = "filtersamples list"

  def description = "Filter samples in current dataset with a sample list"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if ((options.keep && options.remove)
      || (!options.keep && !options.remove))
      fatal("one `--keep' or `--remove' required, but not both")

    val keep = options.keep
    val samples = readFile(options.input, state.hadoopConf) { reader =>
      Source.fromInputStream(reader)
        .getLines()
        .filter(line => !line.isEmpty)
        .toSet
    }
    val p = (s: String, sa: Annotation) => Filter.keepThis(samples.contains(s), keep)

    state.copy(vds = vds.filterSamples(p))
  }
}
