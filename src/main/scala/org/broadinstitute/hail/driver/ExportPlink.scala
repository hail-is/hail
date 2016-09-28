package org.broadinstitute.hail.driver

import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.io.plink.ExportBedBimFam
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
      fatal(s"""Found ${ badSampleIds.length } sample IDs with whitespace
            |  Please run `renamesamples' to fix this problem before exporting to plink format
           |  Bad sample IDs: @1 """.stripMargin, badSampleIds)
    }

    val bedHeader = Array[Byte](108, 27, 1)

    val plinkRDD = vds.rdd
      .mapValuesWithKey { case (v, (va, gs)) => ExportBedBimFam.makeBedRow(gs) }
      .persist(StorageLevel.MEMORY_AND_DISK)

    plinkRDD.map { case (v, bed) => bed }
      .saveFromByteArrays(options.output + ".bed", header = Some(bedHeader))

    plinkRDD.map { case (v, bed) => ExportBedBimFam.makeBimRow(v) }
      .writeTable(options.output + ".bim")

    val famRows = vds
      .sampleIds
      .map(ExportBedBimFam.makeFamRow)

    state.hadoopConf.writeTextFile(options.output + ".fam")(out =>
      famRows.foreach(line => {
        out.write(line)
        out.write("\n")
      }))

    state
  }
}