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

    @Args4jOption(required = false, name = "-c", aliases = Array("--cutoff"),
      usage = "GQ cutoff below which calls will be dropped")
    var cutoff: Int = -1
  }

  def newOptions = new Options

  def name = "exportplink"

  def description = "Write current dataset as .bed/.bim/.fam"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val localCutoff = options.cutoff

    val bedHeader = Array[Byte](108, 27, 1)
    val plinkVariantRDD = vds
      .rdd
      .map {
        case (v, va, gs) =>
          (v, (ExportBedBimFam.makeBedRow(gs, localCutoff), ExportBedBimFam.makeBimRow(v)))
      }

    plinkVariantRDD.persist(StorageLevel.MEMORY_AND_DISK)

    val sortedPlinkRDD = plinkVariantRDD
      .repartitionAndSortWithinPartitions(new RangePartitioner[Variant, (Array[Byte], String)]
      (vds.rdd.partitions.length, plinkVariantRDD))

    sortedPlinkRDD
      .persist(StorageLevel.MEMORY_AND_DISK)

    plinkVariantRDD.unpersist()

    sortedPlinkRDD.map { case (v, (bed, bim)) => bed }
      .saveFromByteArrays(options.output + ".bed", header = Some(bedHeader))

    sortedPlinkRDD.map { case (v, (bed, bim)) => bim }
      .writeTable(options.output + ".bim")

    sortedPlinkRDD.unpersist()

    val famRows = vds
      .localSamples.iterator
      .map(vds.sampleIds)
      .map(ExportBedBimFam.makeFamRow)

    writeTextFile(options.output + ".fam", state.hadoopConf)(oos =>
      famRows.foreach(line => {
        oos.write(line)
        oos.write("\n")
      }))

    state
  }
}