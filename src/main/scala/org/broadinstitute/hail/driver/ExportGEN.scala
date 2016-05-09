package org.broadinstitute.hail.driver

import org.apache.spark.RangePartitioner
import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr.TStruct
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

  override def supportsMultiallelic = false

  def run(state: State, options: Options): State = {
    val sc = state.sc
    val vds = state.vds

    def writeSampleFile() { //FIXME: should output all relevant sample annotations such as phenotype, gender, ...
      val header = Array("ID_1 ID_2 missing","0 0 0")
      writeTable(options.output + ".sample", sc.hadoopConfiguration, header ++ state.vds.sampleIds.map{case s => Array(s, s, "0").mkString(" ")})
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
        val dosages = gt.dosage match {
          case Some(x) => x
          case None => Array(0.0,0.0,0.0)
        }
        sb += ' '
        sb.append(dosages.mkString(" "))
      }
    }

    def writeGenFile() {
      val varidSignature = vds.vaSignature.getAsOption[TStruct]("varid")
      val varidQuery: Querier = varidSignature match {
        case Some(_) => vds.queryVA("va.varid")._2
        case None => a => None
      }

      val rsidSignature = vds.vaSignature.getAsOption[TStruct]("rsid")
      val rsidQuery: Querier = rsidSignature match {
        case Some(_) => vds.queryVA("va.rsid")._2
        case None => a => None
      }

      val kvRDD = vds.rdd.map { case (v, a, gs) =>
        (v, (a, gs.toGenotypeStream(v, compress = false)))
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
        }.writeTable(options.output + ".gen", None, deleteTmpFiles = true)
      kvRDD.unpersist()
    }

    writeSampleFile()
    writeGenFile()

    state
  }
}

