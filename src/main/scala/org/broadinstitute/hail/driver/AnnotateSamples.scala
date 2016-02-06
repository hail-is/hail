package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotations
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant.Sample
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateSamples extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation file path")
    var condition: String = _

    @Args4jOption(required = true, name = "-s", aliases = Array("--sampleheader"),
      usage = "Identify the name of the column containing the sample IDs")
    var sampleCol: String = _

    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Define types of fields in annotations files")
    var types: String = _

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"),
      usage = "Place annotations in the path 'sa.<root>.<field>, or sa.<field> if unspecified'")
    var annotationRoot: String = _

    @Args4jOption(required = false, name = "-m", aliases = Array("--missing"),
      usage = "Specify identifiers to be treated as missing (default: 'NA')")
    var missingIdentifiers: String = "NA"
  }

  def newOptions = new Options

  def name = "annotatesamples"

  def description = "Annotate samples in current dataset"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val typeMap = options.types match {
      case null => Map.empty[String, String]
      case _ => options.types.split(",")
        .map(_.trim())
        .map(s => s.split(":").map(_.trim()))
        .map { arr =>
          if (arr.length != 2)
            fatal("parse error in type declaration")
          arr
        }
        .map(arr => (arr(0), arr(1)))
        .toMap
    }

    val missing = options.missingIdentifiers
      .split(",")
      .map(_.trim())
      .toSet
    println(missing)

    println(typeMap)

    val cond = options.condition
    val newVds = {
      if (cond.endsWith(".tsv") || cond.endsWith(".tsv.gz"))
        Annotate.annotateSamplesFromTSV(vds, cond, options.annotationRoot, options.sampleCol, typeMap, missing)
      else
        null
    }
    state.copy(vds = newVds)
  }
}
