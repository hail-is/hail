package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.io.Source

object FilterSamples extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "--keep", usage = "Keep only listed samples in current dataset")
    var keep: Boolean = false

    @Args4jOption(required = false, name = "--remove", usage = "Remove listed samples from current dataset")
    var remove: Boolean = false

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Filter condition: expression or .sample_list file (one sample name per line)")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "filtersamples"

  def description = "Filter samples in current dataset"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (!options.keep && !options.remove)
      fatal(name + ": one of `--keep' or `--remove' required")

    val indexOfSample: Map[String, Int] = state.vds.sampleIds.zipWithIndex.toMap

    val p = options.condition match {
      case f if f.endsWith(".sample_list") =>
        val samples = Source.fromInputStream(hadoopOpen(f, state.hadoopConf))
          .getLines()
          .filter(line => !line.isEmpty)
          .map(indexOfSample)
          .toSet
        samples.contains(_)
      case c: String =>
        try {
          val cf = new FilterSampleCondition(c)
          cf.typeCheck()

          val sampleIdsBc = state.sc.broadcast(state.vds.sampleIds)
          FilterUtils.pushToBooleanValue((s: Int) => cf(Sample(sampleIdsBc.value(s))), options.keep)
        } catch {
          case e: scala.tools.reflect.ToolBoxError =>
            /* e.message looks like:
               reflective compilation has failed:

               ';' expected but '.' found. */
            fatal("parse error in condition: " + e.message.split("\n").last)
        }
    }

    state.copy(vds = vds.filterSamples(p))
  }
}
