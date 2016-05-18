package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}
import scala.collection.mutable
import scala.io.Source

object RenameSamples extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"), usage = "Input file")
    var input: String = _
  }

  def newOptions = new Options

  def name = "renamesamples"

  def description = "Rename samples"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val m = readFile(options.input, state.hadoopConf) { s =>
      Source.fromInputStream(s)
        .getLines()
        .map {
          _.split("\t") match {
            case Array(old, news) => (old, news)
            case _ =>
              fatal("Invalid input.  Use two tab-separated columns.")
          }
        }.toMap
    }

    val vds = state.vds
    val newSamples = mutable.Set.empty[String]
    val newSampleIds = vds.sampleIds
      .map { case s =>
        val news = m.getOrElse(s, s)
        if (newSamples.contains(news))
          fatal(s"duplicate sample ID `$news' after rename")
        newSamples += news
        news
      }
    state.copy(vds = vds.copy(sampleIds = newSampleIds))
  }
}
