package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.annotators._
import org.kohsuke.args4j.{Argument, Option => Args4jOption}

import scala.collection.JavaConverters._

object AnnotateVariantsPositions extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Define types of fields in annotations files")
    var types: String = ""

    @Args4jOption(required = true, name = "-r", aliases = Array("--root"),
      usage = "Period-delimited path starting with `va'")
    var root: String = _

    @Args4jOption(required = false, name = "-m", aliases = Array("--missing"),
      usage = "Specify identifier to be treated as missing")
    var missingIdentifier: String = "NA"

    @Args4jOption(required = false, name = "-v", aliases = Array("--vcolumns"),
      usage = "Specify the column identifiers for chromosome and position (in that order)")
    var vCols: String = "Chromosome, Position"

    @Args4jOption(required = false, name = "-d", aliases = Array("--delimiter"),
      usage = "Field delimiter regex")
    var delimiter: String = "\\t"

    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()

  }

  def newOptions = new Options

  def name = "annotatevariants positions"

  def description = "Annotate variants with TSV file containing chromosome / position keys"

  def supportsMultiallelic = true

  def requiresVDS = true

  def parseColumns(s: String): Array[String] = {
    val split = s.split(",").map(_.trim)
    if (split.length != 2 && split.length != 1)
      fatal(
        s"""Cannot read chromosome and position columns from `$s':
            |  enter two comma-separated column identifiers for separate chr/pos columns, or
            |  one column identifier for a single chr:pos column.""".stripMargin)
    split
  }

  def run(state: State, options: Options): State = {

    val files = hadoopGlobAll(options.arguments.asScala, state.hadoopConf)

    val vds = state.vds
    val (rdd, signature) = VariantPositionAnnotator(vds.sparkContext, files,
      parseColumns(options.vCols),
      Parser.parseAnnotationTypes(options.types),
      options.missingIdentifier, options.delimiter)
    val annotated = vds
      .withGenotypeStream()
      .annotatePositions(rdd, signature,
        Parser.parseAnnotationRoot(options.root, Annotation.VARIANT_HEAD))

    state.copy(vds = annotated)
  }
}
