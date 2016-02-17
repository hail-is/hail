package org.broadinstitute.hail.driver

import org.apache.hadoop.conf.Configuration
import org.apache.spark.SparkEnv
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotations
import org.broadinstitute.hail.io.annotators._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant.Variant
import org.kohsuke.args4j.{Option => Args4jOption}

object AnnotateVariants extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = "Annotation file path")
    var condition: String = _

    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Define types of fields in annotations files")
    var types: String = _

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"),
      usage = "Place annotations in the path 'va.<root>.<field>, or va.<field> if not specified'")
    var root: String = _

    @Args4jOption(required = false, name = "-m", aliases = Array("--missing"),
      usage = "Specify additional identifiers to be treated as missing (default: 'NA')")
    var missingIdentifiers: String = "NA"

    @Args4jOption(required = false, name = "-i", aliases = Array("--identifier"),
      usage = "For an interval list, use one boolean " +
        "for all intervals (set to true) with the given identifier.  If not specified, will expect a target column")
    var identifier: String = _

    @Args4jOption(required = false, name = "-v", aliases = Array("--vcolumns"),
      usage = "Specify the column identifiers for chromosome, position, ref, and alt (in that order)" +
        " (default: 'Chromosome,Position,Ref,Alt'")
    var vCols: String = "Chromosome, Position, Ref, Alt"
  }

  def newOptions = new Options

  def name = "annotatevariants"

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

  def parseMissing(s: String): Set[String] = {
    s.split(",")
      .map(_.trim())
      .toSet
  }

  def parseColumns(s: String): Array[String] = {
    val split = s.split(",").map(_.trim)
    fatalIf(split.length != 4 && split.length != 1,
      "Cannot read chr, pos, ref, alt columns from '" + s +
        "': enter 4 comma-separated column identifiers for separate chr/pos/ref/alt columns, " +
        "or one identifier for chr:pos:ref:alt")
    split
  }

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition

    val stripped = hadoopStripCodec(cond, state.sc.hadoopConfiguration)

    val annotator: VariantAnnotator = stripped match {
      case intervalList if intervalList.endsWith(".interval_list")  =>
        fatalIf(options.identifier == null, "annotating from .interval_list files requires the argument 'identifier'")
        new IntervalListAnnotator(cond, options.identifier, options.root)
      case tsv if tsv.endsWith(".tsv") =>
        new TSVAnnotatorCompressed(cond, parseColumns(options.vCols), parseTypeMap(options.types),
          parseMissing(options.missingIdentifiers), options.root)
      case bed if bed.endsWith(".bed") => new BedAnnotator(cond, options.root)
      case vcf if vcf.endsWith(".vcf") => new VCFAnnotatorCompressed(cond, options.root)
      case ser if ser.endsWith(".ser") => new SerializedAnnotator(cond, options.root)
      case _ => fatal(s"Unknown file type '$cond'.  Specify a .tsv, .bed, .vcf, .serialized, or .interval_list file")
    }

    state.copy(vds = vds.annotateVariants(annotator))
  }
}
