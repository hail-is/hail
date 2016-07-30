package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateSamplesList extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"),
      usage = "List of sample IDs")
    var input: String = _

    @Args4jOption(required = true, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `sa'")
    var root: String = _
  }

  def newOptions = new Options

  def name = "annotatesamples list"

  def description = "Annotate samples with boolean of presence/absence in list"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val samplesInList = readLines(options.input, state.hadoopConf) { lines =>
      if (lines.isEmpty)
        warn(s"Empty annotation file given ${options.input}")

      lines.map(_.value).toSet
    }

    val sampleAnnotations = vds.sampleIds.map { s => (s, samplesInList.contains(s)) }.toMap

    val annotated = vds.annotateSamples(sampleAnnotations, TBoolean,
      Parser.parseAnnotationRoot(options.root, Annotation.SAMPLE_HEAD))
    state.copy(vds = annotated)
  }
}

