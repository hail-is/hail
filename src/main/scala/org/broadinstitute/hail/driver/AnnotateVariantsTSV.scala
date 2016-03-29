package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.io.annotators._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object AnnotateVariantsTSV extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--input"),
      usage = "TSV file path")
    var condition: String = _

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
      usage = "Specify the column identifiers for chromosome, position, ref, and alt (in that order)")
    var vCols: String = "Chromosome,Position,Ref,Alt"
  }

  def newOptions = new Options

  def name = "annotatevariants/tsv"

  override def hidden = true

  def description = "Annotate variants in current dataset"

  def parseTypeMap(s: String): Map[String, String] = {
    s.split(",")
      .map(_.trim())
      .map(s => s.split(":").map(_.trim()))
      .map {
        case Array(f, t) => (f, t)
        case arr => fatal("parse error in type declaration")
      }
      .toMap
  }

  def parseColumns(s: String): Array[String] = {
    val split = s.split(",").map(_.trim)
    fatalIf(split.length != 4 && split.length != 1,
      "Cannot read chr, pos, ref, alt columns from '" + s +
        "': enter 4 comma-separated column identifiers for separate chr/pos/ref/alt columns, " +
        "or one identifier for chr:pos:ref:alt")
    split
  }

  def parseRoot(s: String): List[String] = {
    val split = s.split("""\.""").toList
    fatalIf(split.isEmpty || split.head != "va", s"invalid root '$s': expect 'va.<path[.path2...]>'")
    split.tail
  }

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition

    val stripped = hadoopStripCodec(cond, state.sc.hadoopConfiguration)


    val conf = state.sc.hadoopConfiguration


    val (rdd, signature) = VariantTSVAnnotator(vds.sparkContext, cond,
      parseColumns(options.vCols),
      parseTypeMap(options.types),
      options.missingIdentifier)
    val annotated = vds.annotateVariants(rdd, signature, parseRoot(options.root))

    state.copy(vds = annotated)
  }
}
