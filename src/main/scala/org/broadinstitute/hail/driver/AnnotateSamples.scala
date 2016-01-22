package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotations
import org.broadinstitute.hail.methods.{FilterSampleCondition, Filter}
import org.broadinstitute.hail.variant.Sample
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.io.Source

object AnnotateSamples extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation file path")
    var condition: String = _

    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Define types of fields in annotations files")
    var types: String = ""

    @Args4jOption(required = true, name = "-s", aliases = Array("--sampleheader"),
      usage = "Identify the name of the column containing the sample IDs")
    var sampleCol: String = _
  }

  def newOptions = new Options

  def name = "annotatesamples"

  def description = "Annotate samples in current dataset"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val typeMap = options.types.split(",")
      .map(_.trim())
      .map(s => s.split("=").map(_.trim()))
      .toMap




    state
  }
}
