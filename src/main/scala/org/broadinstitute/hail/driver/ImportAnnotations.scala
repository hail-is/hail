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

    @Args4jOption(required = false, name = "-d", aliases = Array("--delimiter"),
      usage = "Field delimiter regex")
    var delimiter: String = "\\t"
  }

  def newOptions = new Options

  def name = "importannotations"

  def description = "Import a TSV file containing variants / annotations into a sample-free VDS"

  def supportsMultiallelic = true

  def requiresVDS = false

  def run(state: State, options: Options): State = {
    val files = hadoopGlobAll(options.arguments.asScala, state.hadoopConf)

    if (files.isEmpty)
      fatal("Arguments referred to no files")

    val (rdd, signature) = VariantTableAnnotator(state.sc,
      files,
      AnnotateVariantsTable.parseColumns(options.vCols),
      Parser.parseAnnotationTypes(options.types),
      options.missingIdentifier, options.delimiter)

    val vds = new VariantDataset(
      VariantMetadata(IndexedSeq.empty, Annotation.emptyIndexedSeq(0), Annotation.empty,
        TStruct.empty, signature, TStruct.empty, wasSplit = true),
      rdd.map { case (v, va) => (v, va, Iterable.empty) })

    state.copy(vds = vds)
  }

}
