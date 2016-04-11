package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.annotators._
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Argument, Option => Args4jOption}

import scala.collection.JavaConverters._

object ImportAnnotations extends Command {

  class Options extends BaseOptions {
    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()

    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Define types of fields in annotations files")
    var types: String = ""

    @Args4jOption(required = false, name = "-m", aliases = Array("--missing"),
      usage = "Specify additional identifiers to be treated as missing")
    var missingIdentifier: String = "NA"

    @Args4jOption(required = false, name = "--vcolumns",
      usage = "Specify the column identifiers for chromosome, position, ref, and alt (in that order)")
    var vCols: String = "Chromosome, Position, Ref, Alt"
  }

  def newOptions = new Options

  def name = "importannotations"

  def description = "Import a TSV file containing variants / annotations into a sample-free VDS"

  def run(state: State, options: Options): State = {

    val files = hadoopGlobAll(options.arguments.asScala, state.hadoopConf)

    val (rdd, signature) = VariantTSVAnnotator(state.sc,
      files,
      AnnotateVariantsTSV.parseColumns(options.vCols),
      Parser.parseAnnotationTypes(options.types),
      options.missingIdentifier)

    val vds = new VariantDataset(
      VariantMetadata(IndexedSeq.empty, IndexedSeq.empty, Annotation.emptyIndexedSeq(0),
        TEmpty, signature, wasSplit = true),
      rdd.map { case (v, va) => (v, va, Iterable.empty) })

    state.copy(vds = vds)
  }

}
