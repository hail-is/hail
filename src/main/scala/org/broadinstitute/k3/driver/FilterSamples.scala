package org.broadinstitute.k3.driver

import java.io.File

import org.broadinstitute.k3.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.io.Source

object FilterSamples extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "--keep", usage = "Keep only listed samples in current dataset")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove listed samples from current dataset")
    var remove: Boolean = false

    @Args4jOption(required = true, name = "-s", aliases = Array("--samplelist"),
      usage = "file with a list of sample names, one per line")
    var samples: String = _
  }
  def newOptions = new Options

  def name = "filtersamples"
  def description = "Filter samples in current dataset"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (!options.keep && !options.remove)
      fatal(name + ": one of `--keep' or `--remove' required")

    val indexOfSample: Map[String, Int] = state.vds.sampleIds.zipWithIndex.toMap

    val samples = Source.fromFile(new File(options.samples))
      .getLines()
      .filter(line => !line.isEmpty)
      .map(indexOfSample)
      .toArray

    val p: (Int) => Boolean = (s) => samples.contains(s)

    val newVDS = vds.filterSamples(if (options.keep)
      p
    else
      (s: Int) => !p(s))
    
    state.copy(vds = newVDS)
  }
}
