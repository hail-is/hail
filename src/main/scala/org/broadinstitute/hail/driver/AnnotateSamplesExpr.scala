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
    val vds = state.vds
    val localGlobalAnnotations = vds.globalAnnotation

    val cond = options.condition

    val ec = Aggregators.sampleEC(vds)

    val (paths, types, f) = Parser.parseAnnotationExprs(cond, ec, Some(Annotation.SAMPLE_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val finalType = (paths, types).zipped.foldLeft(vds.saSignature) { case (sas, (ids, signature)) =>
      val (s, i) = sas.insert(signature, ids)
      inserterBuilder += i
      s
    }
    val inserters = inserterBuilder.result()

    val sampleAggregationOption = Aggregators.buildSampleAggregations(vds, ec)

    val newAnnotations = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      sampleAggregationOption.foreach(f => f.apply(s))
      ec.setAll(localGlobalAnnotations, s, sa)
      f().zip(inserters)
        .foldLeft(sa) { case (sa, (v, inserter)) =>
          inserter(sa, v)
        }
    }

    state.copy(vds = vds.copy(
      sampleAnnotations = newAnnotations,
      saSignature = finalType
    ))
  }
}
