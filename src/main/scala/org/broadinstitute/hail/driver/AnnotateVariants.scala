package org.broadinstitute.hail.driver

import org.apache.hadoop.conf.Configuration
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotations
import org.broadinstitute.hail.methods.{FilterSampleCondition, Filter, Annotate}
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
    var annotationRoot: String = _

    @Args4jOption(required = false, name = "-m", aliases = Array("--missing"),
      usage = "Specify additional identifiers to be treated as missing (default: 'NA')")
    var missingIdentifiers: String = "NA"

    @Args4jOption(required = false, name = "--intervals",
      usage = "indicates that the given TSV is an interval file")
    var intervalFile: Boolean = _

    @Args4jOption(required = false, name = "--identifier", usage = "For an interval list, use one boolean " +
      "for all intervals (set to true) with the given identifier.  If not specified, will expect a target column")
    var identifier: String = _

    @Args4jOption(required = false, name = "--vcolumns",
      usage = "Specify the column identifiers for chromosome, position, ref, and alt (in that order)" +
        " (default: 'Chromosome,Position,Ref,Alt'")
    var vCols: String = "Chromosome, Position, Ref, Alt"

    @Args4jOption(required = false, name = "--icolumns",
      usage = "Specify the column identifiers for chromosome, start, and end (in that order)" +
        " (default: 'Chromosome,Start,End'")
    var iCols: String = "Chromosome,Start,End"
  }

  def newOptions = new Options

  def name = "annotatevariants"

  def description = "Annotate variants in current dataset"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    // FIXME -- support this stuff
    /*
    to support:
    tsv + header (chr pos ref alt, like annotate samples)
    interval list, boolean
    interval list, include target
    BED
    VCF info field
    */
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

    val cond = options.condition
    val newVds = {
      if (cond.endsWith(".ser") || cond.endsWith(".ser.gz")) {
        Annotate.annotateVariantsFromKryo(vds, cond, options.annotationRoot)
      }
      else if (cond.endsWith(".tsv") || cond.endsWith(".tsv.gz")) {
        // this group works for interval lists and chr pos ref alt
        if (options.intervalFile) {
          val iCols = options.iCols.split(",").map(_.trim)
          if (iCols.length != 3)
            fatal(s"""Cannot read chr, start, end columns from "${options.iCols}": enter 3 comma-separated column identifiers""")
          Annotate.annotateVariantsFromIntervalList(vds, cond, options.annotationRoot, typeMap, iCols, options.identifier)
        }

        val vCols = options.vCols.split(",").map(_.trim)
        if (vCols.length != 4)
          fatal(s"""Cannot read chr, pos, ref, alt columns from "${options.vCols}": enter 4 comma-separated column identifiers""")

        Annotate.annotateVariantsFromTSV(vds, cond, options.annotationRoot, typeMap, missing, vCols)
      }
      else if (cond.endsWith(".bed") || cond.endsWith(".bed.gz"))
        Annotate.annotateVariantsFromBed(vds, cond, options.annotationRoot)
      else if (cond.endsWith(".vcf") || cond.endsWith(".vcf.gz") || cond.endsWith(".vcf.bgz")) {

        Annotate.annotateVariantsFromVCF(vds, cond, options.annotationRoot)
      }
      else
        throw new UnsupportedOperationException
    }

    state.copy(vds = newVds)
  }
}
