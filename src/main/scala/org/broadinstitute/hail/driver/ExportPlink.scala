package org.broadinstitute.hail.driver

import org.apache.spark.RangePartitioner
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.methods.ExportBedBimFam
import org.broadinstitute.hail.variant.{Variant, Genotype}
import org.broadinstitute.hail.annotations.{VCFSignature, AnnotationData}
import org.kohsuke.args4j.{Option => Args4jOption}
import java.time._

object ExportPlink extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "Output file base (will generate .bed, .bim, .fam)")
    var output: String = _

    @Args4jOption(required = true, name = "-t", aliases = Array("--tmpdir"),
      usage = "Directory for temporary files")
    var tmpdir: String = _

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
    val header = new String(Array[Byte](108, 27, 1))
    val bedRowRDD = vds
      .rdd
      .map {
        case (v, va, gs) => (v, ExportBedBimFam.makeBedRow(v.start, gs, localCutoff))
      }
    bedRowRDD
      .repartitionAndSortWithinPartitions(new RangePartitioner[Variant, Array[Byte]](vds.rdd.partitions.length,
        bedRowRDD))
      .map(_._2)
      .writeTableSingleFile(options.tmpdir, options.output + ".bed", header = Some(header),
        deleteTmpFiles = true, newLines = false)

    val bimRowRDD = vds
      .variantsAndAnnotations
      .map {
        case (v, va) => (v, ExportBedBimFam.makeBimRow(v, va))
      }
    bimRowRDD
      .repartitionAndSortWithinPartitions(new RangePartitioner[Variant, String](vds.rdd.partitions.length, bimRowRDD))
      .map(_._2)
      .writeTableSingleFile(options.tmpdir, options.output + ".bim", deleteTmpFiles = true)

    val fsos = hadoopCreate(options.output + ".fam", state.hadoopConf)
    val famRows = vds
      .localSamples
      .map(vds.sampleIds)
      .map(ExportBedBimFam.makeFamRow)
      .map(_ + "\n")

    writeTextFile(options.output + ".fam", state.hadoopConf)(oos => famRows.foreach(line => oos.write(line)))

    state
  }
}