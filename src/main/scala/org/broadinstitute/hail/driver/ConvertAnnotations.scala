package org.broadinstitute.hail.driver

import java.nio.channels.Channels

import org.apache.hadoop.conf.Configuration
import org.apache.spark.SparkEnv
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant.Variant
import org.kohsuke.args4j.{Option => Args4jOption}

object ConvertAnnotations extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-i", aliases = Array("--import"),
      usage = "Annotation file path")
    var condition: String = _

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "Path of .ser file")
    var output: String = _

    @Args4jOption(required = false, name = "-t", aliases = Array("--types"),
      usage = "Define types of fields in annotations files")
    var types: String = _

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

  def name = "convertannotations"

  def description = "Convert a tsv or vcf file containing variant annotations into the fast hail format"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    if (!options.output.endsWith(".ser") && !options.output.endsWith(".ser.gz"))
      fatal("Output path must end in '.ser' or '.ser.gz'")

    val cond = options.condition

    if (cond.endsWith(".tsv") || cond.endsWith(".tsv.gz")) {
      // this group works for interval lists and chr pos ref alt
      if (options.intervalFile) {
        val iCols = options.iCols.split(",").map(_.trim)
        if (iCols.length != 3)
          fatal(s"""Cannot read chr, start, end columns from "${options.iCols}": enter 3 comma-separated column identifiers""")
      }

      val vCols = options.vCols.split(",").map(_.trim)
      if (vCols.length != 4)
        fatal(s"""Cannot read chr, pos, ref, alt columns from "${options.vCols}": enter 4 comma-separated column identifiers""")

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

      val conf = new Configuration()

      val reader = new TSVCompressor(options.condition, vCols, typeMap, missing)

      val (header, signatures, variantMap, compressedBytes) = reader.parse(conf, SparkEnv.get.serializer.newInstance())

      println("compressed bytes = " + compressedBytes.length)

      println(s"writing to ${options.output}")
      val dos = hadoopCreate(options.output, conf)
      org.apache.spark.SparkEnv.get.serializer
        .newInstance()
        .serializeStream(dos)
        .writeObject("tsv")
        .writeObject(header)
        .writeObject(signatures)
        .writeObject(compressedBytes)
        .writeObject(variantMap)
        .close()

    }
    else if (cond.endsWith(".vcf") || cond.endsWith(".vcf.gz") || cond.endsWith(".vcf.bgz")) {


      val conf = new Configuration
      val reader = new VCFCompressor(options.condition)

      val (signatures, variantMap, compressedBytes) = reader.parse(conf, SparkEnv.get.serializer.newInstance())

      val dos = hadoopCreate(options.output, conf)
      org.apache.spark.SparkEnv.get.serializer
        .newInstance()
        .serializeStream(dos)
        .writeObject("vcf")
        .writeObject(IndexedSeq.empty[String])
        .writeObject(signatures)
        .writeObject(variantMap)
        .writeObject(compressedBytes)
        .close()
    }
    else
      fatal(
        """This module requires an input file ending in one of the following:
          |  .tsv (tab separated values with chr, pos, ref, alt)
          |  .vcf (vcf, only the info field / filters / qual are parsed here)""".stripMargin)

    state
  }
}
