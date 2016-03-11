package org.broadinstitute.hail.driver

import org.apache.spark.RangePartitioner
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr
import RichRow._
import org.broadinstitute.hail.variant.{Variant, Genotype}
import org.broadinstitute.hail.annotations._
import org.kohsuke.args4j.{Option => Args4jOption}
import java.time._
import scala.collection.mutable
import scala.io.Source

object ExportVCF extends Command {

  class Options extends BaseOptions {
    @Args4jOption(name = "-a", usage = "Append file to header")
    var append: String = _

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _

  }

  def newOptions = new Options

  def name = "exportvcf"

  def description = "Write current dataset as VCF file"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val varAnnSig = vds.metadata.vaSignatures

    def header: String = {
      val sb = new StringBuilder()

      sb.append("##fileformat=VCFv4.2\n")
      sb.append(s"##fileDate=${LocalDate.now}\n")
      // FIXME add Hail version
      sb.append(
        """##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
          |##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for the ref and alt alleles in the order listed">
          |##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
          |##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
          |##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Normalized, Phred-scaled likelihoods for genotypes as defined in the VCF specification">""".stripMargin)
      sb += '\n'

      vds.metadata.filters.map { case (key, desc) =>
        sb.append(s"""##FILTER=<ID=$key,Description="$desc">\n""")
      }

      val infoHeader: Option[Array[(String, VCFSignature)]] = vds.metadata.vaSignatures
        .getOption(List("info")) match {
        case Some(sigs: StructSignature) =>
          Some(sigs.m.toArray
            .sortBy { case (key, (i, s)) => i }
            .flatMap { case (key, (i, s)) => s match {
              case vcfSig: VCFSignature => Some(key, vcfSig)
              case _ => None
            }
            })
        case _ => None
      }
      infoHeader.foreach { i =>
        i.foreach { case (key, value) =>
          val sig = value.asInstanceOf[VCFSignature]
          sb.append(
            s"""##INFO=<ID=$key,Number=${sig.number},Type=${sig.vcfType},Description="${sig.description}">\n""")
        }
      }

      if (options.append != null) {
        readFile(options.append, state.hadoopConf) { s =>
          Source.fromInputStream(s)
            .getLines()
            .filterNot(_.isEmpty)
            .foreach { line =>
              sb.append(line)
              sb += '\n'
            }
        }
      }

      sb.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
      val sampleIds: Array[String] = vds.localSamples.map(vds.sampleIds)
      sampleIds.foreach { id =>
        sb += '\t'
        sb.append(id)
      }
      sb.result()
    }

    def printInfo(a: Any): String = {
      a match {
        case iter: Iterable[_] => iter.map(_.toString).mkString(",")
        case _ => a.toString
      }
    }

    val infoF: (StringBuilder, Annotation) => Unit = {
      vds.metadata.vaSignatures.getOption(List("info")) match {
        case Some(signatures: StructSignature) =>
          val keys = signatures.m.map { case (k, (i, v)) => (k, i, v.dType == expr.TBoolean) }
            .toArray
            .sortBy { case (key, index, isBoolean) => index }
            .map { case (key, index, isBoolean) => (key, isBoolean) }
          val querier = vds.metadata.vaSignatures.query(List("info"))

          (sb, ad) => {
            val infoRow = querier(ad).map(_.asInstanceOf[Row])
            var first = true
            infoRow match {
              case Some(r) =>
                var appended = 0
                keys.iterator.zipWithIndex.map { case ((s, isBoolean), i) =>
                  (s, isBoolean, r.getOption(i))
                }
                  .flatMap { case (s, isBoolean, opt) =>
                    opt match {
                      case Some(o) => Some(s, isBoolean, o)
                      case None => None
                    }
                  }
                  .filter { case (s, isBoolean, v) => !(isBoolean && (v == false)) }
                  .foreachBetween({ case (s, isBoolean, a) =>
                    sb.append(s)
                    if (!isBoolean) {
                      sb += '='
                      sb.append(printInfo(a))
                    }
                    appended += 1
                  }) { unit => sb += ';' }
                if (appended == 0)
                  sb += '.'

              case None => sb.append(".")
            }
          }
        case _ =>
          (sb, ad) => sb.append(".")
      }
    }

    val idQuery: Querier = if (vds.vaSignatures.getOption("rsid").forall(sig => sig.dType == expr.TString))
      vds.queryVA("rsid")
    else
      a => None

    val qualQuery: Querier = if (vds.vaSignatures.getOption("qual").forall(sig => sig.dType == expr.TDouble))
      vds.queryVA("rsid")
    else
      a => None

    val filterQuery: Querier = if (vds.vaSignatures.getOption("filters").forall(sig =>
      sig.dType == expr.TSet(expr.TString)))
      vds.queryVA("rsid")
    else
      a => None

    def appendRow(sb: StringBuilder, v: Variant, a: Annotation, gs: Iterable[Genotype],
      infoF: (StringBuilder, Annotation) => Unit) {

      sb.append(v.contig)
      sb += '\t'
      sb.append(v.start)
      sb += '\t'

      val id = idQuery(a).getOrElse(".")
      sb.append(id)

      sb += '\t'
      sb.append(v.ref)
      sb += '\t'
      v.altAlleles.foreachBetween(aa =>
        sb.append(aa.alt))(_ => sb += ',')
      sb += '\t'

      val qual = qualQuery(a).map(_.asInstanceOf[Double].formatted("%.2f"))
        .getOrElse(".")
      sb.append(qual)
      sb += '\t'

      val filters = filterQuery(a).map(_.asInstanceOf[mutable.WrappedArray[String]]) match {
        case Some(f) =>
          if (f.nonEmpty)
            f.foreachBetween(s => sb.append(s))(_ => sb += ',')
          else
            sb += '.'
        case None => sb += '.'
      }

      sb += '\t'

      infoF(sb, a)

      sb += '\t'
      sb.append("GT:AD:DP:GQ:PL")

      gs.foreach { g =>
        sb += '\t'
        sb.append(g)
      }
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
          appendRow(sb, v, va, gs, infoF)
          sb.result()
        }
      }.writeTable(options.output, Some(header), deleteTmpFiles = true)
    kvRDD.unpersist()
    state
  }

}
