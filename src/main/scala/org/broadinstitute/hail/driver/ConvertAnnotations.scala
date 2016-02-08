package org.broadinstitute.hail.driver

import org.apache.hadoop.conf.Configuration
import org.apache.spark.SparkEnv
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.io.annotators._
import org.kohsuke.args4j.{Option => Args4jOption}

object ConvertAnnotations extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases = Array("--import"),
      usage = "Annotation file path")
    var condition: String = _

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "Path for writing write serialized file")
    var output: String = _

    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Define types of fields in annotations files")
    var types: String = _

    @Args4jOption(required = false, name = "-m", aliases = Array("--missing"),
      usage = "Specify additional identifiers to be treated as missing (default: 'NA')")
    var missingIdentifiers: String = "NA"

    @Args4jOption(required = false, name = "--vcolumns",
      usage = "Specify the column identifiers for chromosome, position, ref, and alt (in that order)" +
        " (default: 'Chromosome,Position,Ref,Alt'")
    var vCols: String = "Chromosome, Position, Ref, Alt"
  }

  def newOptions = new Options

  def name = "convertannotations"

  def description = "Convert a tsv or vcf file containing variant annotations into the fast hail format"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (!options.output.endsWith(".ser") && !options.output.endsWith(".ser.gz"))
      fatal("Output path must end in '.ser' or '.ser.gz'")

    val cond = options.condition

    val serializer = SparkEnv.get.serializer.newInstance()
    if (cond.endsWith(".tsv") || cond.endsWith(".tsv.gz")) {
      // this group works for interval lists and chr pos ref alt
      new TSVAnnotatorCompressed(cond, AnnotateVariants.parseColumns(options.vCols),
        AnnotateVariants.parseTypeMap(options.types),
        AnnotateVariants.parseMissing(options.missingIdentifiers),
        null)
        .serialize(options.output, serializer)
    }
    else if (cond.endsWith(".vcf") || cond.endsWith(".vcf.gz") || cond.endsWith(".vcf.bgz")) {
      new VCFAnnotatorCompressed(cond, null)
        .serialize(options.output, serializer)
    }
    else
      fatal(
        """This module requires an input file ending in one of the following:
          |  .tsv (tab separated values with chr, pos, ref, alt)
          |  .vcf (vcf, only the info field / filters / qual are parsed here)""".stripMargin)

    state
  }
}
