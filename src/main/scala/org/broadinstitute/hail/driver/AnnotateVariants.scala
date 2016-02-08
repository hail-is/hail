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

    @Args4jOption(required = false, name = "--vcolumns",
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
      .map { arr =>
        fatalIf(arr.length != 2, "parse error in type declaration")
        arr
      }
      .map(arr => (arr(0), arr(1)))
      .toMap
  }

  def parseMissing(s: String): Set[String] = {
    s.split(",")
      .map(_.trim())
      .toSet
  }

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val cond = options.condition

    val annotator: VariantAnnotator = {
      if (cond.endsWith(".interval_list") || cond.endsWith(".interval_list.gz")) {
        fatalIf(options.identifier == null, "annotating from .interval_list files requires the argument 'identifier'")
        new IntervalListAnnotator(cond, options.identifier, options.root)
      }
      else if (cond.endsWith(".tsv") || cond.endsWith(".tsv.gz")) {
        // this group works for interval lists and chr pos ref alt
        val vCols = options.vCols.split(",").map(_.trim)
        fatalIf(vCols.length != 4,
          "Cannot read chr, pos, ref, alt columns from" + options.vCols +
            ": enter 4 comma-separated column identifiers")
        new TSVAnnotatorCompressed(cond, vCols, parseTypeMap(options.types),
          parseMissing(options.missingIdentifiers), options.root)
      }
      else if (cond.endsWith(".bed") || cond.endsWith(".bed.gz"))
        new BedAnnotator(cond, options.root)
      else if (cond.endsWith(".vcf") || cond.endsWith(".vcf.gz") || cond.endsWith(".vcf.bgz")) {
        new VCFAnnotatorCompressed(cond, options.root)
      }
      else
        throw new UnsupportedOperationException
    }

    state.copy(vds = vds.annotateVariants(annotator))
  }
}
