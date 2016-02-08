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
    var cutoff: Int = 0
  }

  def newOptions = new Options

  def name = "exportplink"

  def description = "Write current dataset as .bed/.bim/.fam"

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val localCutoff = options.cutoff

    // FIXME magic numbers in header
    val bedHeader = Array[Byte](108, 27, 1)
    val bedRowRDD = vds
      .rdd
      .map {
        case (v, va, gs) => (v, ExportBedBimFam.makeBedRow(v.start, gs, localCutoff))
      }

    bedRowRDD.persist(StorageLevel.MEMORY_AND_DISK)
    bedRowRDD
      .repartitionAndSortWithinPartitions(new RangePartitioner[Variant, Array[Byte]](vds.rdd.partitions.length,
        bedRowRDD))
      .map(_._2)
      .saveFromByteArrays(options.output + ".bed", header = Some(bedHeader))
    bedRowRDD.unpersist()

    val bimRowRDD = vds
      .variantsAndAnnotations
      .map {
        case (v, va) => (v, ExportBedBimFam.makeBimRow(v))
      }

    bimRowRDD.persist(StorageLevel.MEMORY_AND_DISK)
    bimRowRDD
      .repartitionAndSortWithinPartitions(new RangePartitioner[Variant, String](vds.rdd.partitions.length, bimRowRDD))
      .map(_._2)
      .writeTable(options.output + ".bim")
    bimRowRDD.unpersist()

    val famRows = vds
      .localSamples
      .map(vds.sampleIds)
      .map(ExportBedBimFam.makeFamRow)
      .map(_ + "\n")

    writeTextFile(options.output + ".fam", state.hadoopConf)(oos => famRows.foreach(line => oos.write(line)))

    state
  }
}