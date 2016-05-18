package org.broadinstitute.hail.driver

import org.apache.spark.RangePartitioner
import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.methods.ExportBedBimFam
import org.broadinstitute.hail.variant.Variant
import org.kohsuke.args4j.{Option => Args4jOption}

object ExportPlink extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "Output file base (will generate .bed, .bim, .fam)")
    var output: String = _
  }

  def newOptions = new Options

  def name = "exportplink"

  def description = "Write current dataset as .bed/.bim/.fam"

  def supportsMultiallelic = false

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val spaceRegex = """\s+""".r
    val badSampleIds = vds.sampleIds.filter(id => spaceRegex.findFirstIn(id).isDefined)
    if (badSampleIds.nonEmpty) {
      val msg =
        s"""Found ${badSampleIds.length} sample IDs with whitespace.  Please run `renamesamples' to fix this problem before exporting to plink format.""".stripMargin
      log.error(msg + s"\n  Bad sample IDs: \n  ${badSampleIds.mkString("  \n")}")
      fatal(msg + s"\n  Bad sample IDs: \n  ${badSampleIds.take(10).mkString("  \n")}${
        if (badSampleIds.length > 10) "\n  ...\n  See hail.log for full list of IDs" else ""}")
    }

    val bedHeader = Array[Byte](108, 27, 1)

    val plinkVariantRDD = vds
      .rdd
      .map {
        case (v, va, gs) =>
          (v, ExportBedBimFam.makeBedRow(gs))
      }

    plinkVariantRDD.persist(StorageLevel.MEMORY_AND_DISK)

    val sortedPlinkRDD = plinkVariantRDD
      .repartitionAndSortWithinPartitions(new RangePartitioner[Variant, Array[Byte]]
      (vds.rdd.partitions.length, plinkVariantRDD))

    sortedPlinkRDD
      .persist(StorageLevel.MEMORY_AND_DISK)

    sortedPlinkRDD.map { case (v, bed) => bed }
      .saveFromByteArrays(options.output + ".bed", header = Some(bedHeader))

    sortedPlinkRDD.map { case (v, bed) => ExportBedBimFam.makeBimRow(v) }
      .writeTable(options.output + ".bim")

    plinkVariantRDD.unpersist()
    sortedPlinkRDD.unpersist()

    val famRows = vds
      .sampleIds
      .map(ExportBedBimFam.makeFamRow)

    writeTextFile(options.output + ".fam", state.hadoopConf)(oos =>
      famRows.foreach(line => {
        oos.write(line)
        oos.write("\n")
      }))

    state
  }
}