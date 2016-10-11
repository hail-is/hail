package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations.{Annotation, Inserter}
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods.Aggregators
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object AnnotateSamplesExpr extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation condition")
    var condition: String = _
  }

  def newOptions = new Options

  def name = "annotatesamples expr"

  def description = "Use the Hail Expression Language to compute new annotations from existing sample annotations, as well as perform genotype aggregation."

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    state.copy(vds = state.vds.annotateSamples(options.condition))
  }
}
