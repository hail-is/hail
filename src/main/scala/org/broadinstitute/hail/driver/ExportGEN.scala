package org.broadinstitute.hail.driver

import org.apache.spark.RangePartitioner
import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr.TString
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

object ExportGEN extends Command {
  def name = "exportgen"

  def description = "Export VDS as a GEN file"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "Output file base (will generate .gen & .sample)")
    var output: String = _

  }

  def newOptions = new Options

  def supportsMultiallelic = false

  def requiresVDS = true

  val emptyDosage = Array(0d, 0d, 0d)

  def formatDosage(d: Double): String = d.formatted("%.4f")

  def run(state: State, options: Options): State = {
    val sc = state.sc
    val vds = state.vds

    def writeSampleFile() {
      //FIXME: should output all relevant sample annotations such as phenotype, gender, ...
      sc.hadoopConfiguration.writeTable(options.output + ".sample",
        "ID_1 ID_2 missing" :: "0 0 0" :: vds.sampleIds.map(s => s"$s $s 0").toList)
    }

    def appendRow(sb: StringBuilder, v: Variant, va: Annotation, gs: Iterable[Genotype], rsidQuery: Querier, varidQuery: Querier) {
      sb.append(v.contig)
      sb += ' '
      sb.append(varidQuery(va).getOrElse(v.toString))
      sb += ' '
      sb.append(rsidQuery(va).getOrElse("."))
      sb += ' '
      sb.append(v.start)
      sb += ' '
      sb.append(v.ref)
      sb += ' '
      sb.append(v.alt)

      for (gt <- gs) {
        val dosages = gt.dosage.getOrElse(ExportGEN.emptyDosage)
        sb += ' '
        sb.append(formatDosage(dosages(0)))
        sb += ' '
        sb.append(formatDosage(dosages(1)))
        sb += ' '
        sb.append(formatDosage(dosages(2)))
      }
    }

    def writeGenFile() {
      val varidSignature = vds.vaSignature.getOption("varid")
      val varidQuery: Querier = varidSignature match {
        case Some(_) => val (t, q) = vds.queryVA("va.varid")
          t match {
            case TString => q
            case _ => a => None
          }
        case None => a => None
      }

      val rsidSignature = vds.vaSignature.getOption("rsid")
      val rsidQuery: Querier = rsidSignature match {
        case Some(_) => val (t, q) = vds.queryVA("va.rsid")
          t match {
            case TString => q
            case _ => a => None
          }
        case None => a => None
      }

      val isDosage = vds.isDosage

      val kvRDD = vds.rdd.map { case (v, (a, gs)) =>
        (v, (a, gs.toGenotypeStream(v, isDosage, compress = false)))
      }
      kvRDD.persist(StorageLevel.MEMORY_AND_DISK)
      kvRDD
        .repartitionAndSortWithinPartitions(new RangePartitioner[Variant, (Annotation, Iterable[Genotype])](vds.rdd.partitions.length, kvRDD))
        .mapPartitions { it: Iterator[(Variant, (Annotation, Iterable[Genotype]))] =>
          val sb = new StringBuilder
          it.map { case (v, (va, gs)) =>
            sb.clear()
            appendRow(sb, v, va, gs, rsidQuery, varidQuery)
            sb.result()
          }
        }.writeTable(options.output + ".gen", None)
      kvRDD.unpersist()
    }

    writeSampleFile()
    writeGenFile()

    state
  }
}