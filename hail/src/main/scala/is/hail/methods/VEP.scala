package is.hail.methods

import com.fasterxml.jackson.core.JsonParseException
import is.hail.annotations._
import is.hail.expr._
import is.hail.expr.ir.{TableLiteral, TableValue}
import is.hail.expr.types._
import is.hail.rvd.{RVD, RVDContext}
import is.hail.sparkextras.ContextRDD
import is.hail.table.Table
import is.hail.utils._
import is.hail.variant.{Locus, RegionValueVariant, VariantMethods}
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.json4s.jackson.JsonMethods
import org.apache.hadoop

import scala.collection.JavaConverters._

case class VEPConfiguration(
  command: Array[String],
  env: Map[String, String],
  vep_json_schema: TStruct)

object VEP {
  def readConfiguration(hadoopConf: hadoop.conf.Configuration, path: String): VEPConfiguration = {
    val jv = hadoopConf.readFile(path) { in =>
      JsonMethods.parse(in)
    }
    implicit val formats = defaultJSONFormats + new TStructSerializer
    jv.extract[VEPConfiguration]
  }

  def printContext(w: (String) => Unit) {
    w("##fileformat=VCFv4.1")
    w("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
  }

  def printElement(w: (String) => Unit, v: (Locus, IndexedSeq[String])) {
    val (locus, alleles) = v

    val sb = new StringBuilder()
    sb.append(locus.contig)
    sb += '\t'
    sb.append(locus.position)
    sb.append("\t.\t")
    sb.append(alleles(0))
    sb += '\t'
    sb.append(alleles.tail.filter(_ != "*").mkString(","))
    sb.append("\t.\t.\tGT")
    w(sb.result())
  }

  def variantFromInput(input: String): (Locus, IndexedSeq[String]) = {
    try {
      val a = input.split("\t")
      (Locus(a(0), a(1).toInt), a(3) +: a(4).split(","))
    } catch {
      case e: Throwable => fatal(s"VEP returned invalid variant '$input'", e)
    }
  }

  def getCSQHeaderDefinition(cmd: Array[String], confEnv: Map[String, String]): Option[String] = {
    val csqHeaderRegex = "ID=CSQ[^>]+Description=\"([^\"]+)".r
    val pb = new ProcessBuilder(cmd.toList.asJava)
    val env = pb.environment()
    confEnv.foreach { case (key, value) => env.put(key, value) }
    
    val (jt, proc) = List((Locus("1", 13372), IndexedSeq("G", "C"))).iterator.pipe(pb,
      printContext,
      printElement,
      _ => ())

    val csqHeader = jt.flatMap(s => csqHeaderRegex.findFirstMatchIn(s).map(m => m.group(1)))
    val rc = proc.waitFor()
    if (rc != 0)
      fatal(s"VEP command failed with non-zero exit status $rc")

    if (csqHeader.hasNext)
      Some(csqHeader.next())
    else {
      warn("could not get VEP CSQ header")
      None
    }
  }
  
  def annotate(ht: Table, config: String, csq: Boolean, blockSize: Int): Table = {
    assert(ht.key == FastIndexedSeq("locus", "alleles"))
    assert(ht.typ.rowType.size == 2)

    val conf = readConfiguration(ht.hc.hadoopConf, config)
    val vepSignature = conf.vep_json_schema

    val cmd = conf.command.map(s =>
      if (s == "__OUTPUT_FORMAT_FLAG__")
        if (csq) "--vcf" else "--json"
      else
        s)

    val csqHeader = if (csq) getCSQHeaderDefinition(cmd, conf.env) else None
    
    val inputQuery = vepSignature.query("input")

    val csqRegex = "CSQ=[^;^\\t]+".r

    val localBlockSize = blockSize

    val localRowType = ht.typ.rowType
    val rowKeyOrd = ht.typ.keyType.ordering

    val prev = ht.value.rvd
    val annotations = prev
      .mapPartitions { it =>
        val pb = new ProcessBuilder(cmd.toList.asJava)
        val env = pb.environment()
        conf.env.foreach { case (key, value) =>
            env.put(key, value)
        }

        val rvv = new RegionValueVariant(localRowType)
        it
          .map { rv =>
            rvv.setRegion(rv)
            (rvv.locus(), rvv.alleles(): IndexedSeq[String])
          }
          .grouped(localBlockSize)
          .flatMap { block =>
            val (jt, proc) = block.iterator.pipe(pb,
              printContext,
              printElement,
              _ => ())

            val nonStarToOriginalVariant = block.map { case v@(locus, alleles) =>
              (locus, alleles.filter(_ != "*")) -> v
            }.toMap

            val kt = jt
              .filter(s => !s.isEmpty && s(0) != '#')
              .map { s =>
                if (csq) {
                  val vepv@(vepLocus, vepAlleles) = variantFromInput(s)
                  nonStarToOriginalVariant.get(vepv) match {
                    case Some(v@(locus, alleles)) =>
                      val x = csqRegex.findFirstIn(s)
                      val a = x match {
                        case Some(value) =>
                          value.substring(4).split(",").toFastIndexedSeq
                        case None =>
                          warn(s"No CSQ INFO field for VEP output variant ${ VariantMethods.locusAllelesToString(vepLocus, vepAlleles) }.\nVEP output: $s.")
                          null
                      }
                      (Annotation(locus, alleles), a)
                    case None =>
                      fatal(s"VEP output variant ${ VariantMethods.locusAllelesToString(vepLocus, vepAlleles) } not found in original variants.\nVEP output: $s")
                  }
                } else {
                  val jv = JsonMethods.parse(s)
                  val a = JSONAnnotationImpex.importAnnotation(jv, vepSignature)
                  val variantString = inputQuery(a).asInstanceOf[String]
                  if (variantString == null)
                    fatal(s"VEP generated null variant string" +
                      s"\n  json:   $s" +
                      s"\n  parsed: $a")
                  val vepv@(vepLocus, vepAlleles) = variantFromInput(variantString)

                  nonStarToOriginalVariant.get(vepv) match {
                    case Some(v@(locus, alleles)) =>
                      (Annotation(locus, alleles), a)
                    case None =>
                      fatal(s"VEP output variant ${ VariantMethods.locusAllelesToString(vepLocus, vepAlleles) } not found in original variants.\nVEP output: $s")
                  }
                }
              }

            val r = kt.toArray
              .sortBy(_._1)(rowKeyOrd.toOrdering)

            val rc = proc.waitFor()
            if (rc != 0)
              fatal(s"vep command '${ cmd.mkString(" ") }' failed with non-zero exit status $rc")

            r
          }
      }

    val vepType: Type = if (csq) TArray(TString()) else vepSignature

    val vepRVDType = prev.typ.copy(rowType = prev.rowType ++ TStruct("vep" -> vepType))

    val vepRowType = vepRVDType.rowType

    val vepRVD: RVD = RVD(
      vepRVDType,
      prev.partitioner,
      ContextRDD.weaken[RVDContext](annotations).cmapPartitions { (ctx, it) =>
        val region = ctx.region
        val rvb = ctx.rvb
        val rv = RegionValue(region)

        it.map { case (v, vep) =>
          rvb.start(vepRowType)
          rvb.startStruct()
          rvb.addAnnotation(vepRowType.types(0), v.asInstanceOf[Row].get(0))
          rvb.addAnnotation(vepRowType.types(1), v.asInstanceOf[Row].get(1))
          rvb.addAnnotation(vepRowType.types(2), vep)
          rvb.endStruct()
          rv.setOffset(rvb.end())
          rv
        }
      }).persist(StorageLevel.MEMORY_AND_DISK)

    val (globalValue, globalType) =
      if (csq)
        (Row(csqHeader.getOrElse("")), TStruct("vep_csq_header" -> TString()))
      else
        (Row(), TStruct())
    
    new Table(ht.hc, TableLiteral(TableValue(
      TableType(vepRowType, FastIndexedSeq("locus", "alleles"), globalType),
      BroadcastRow(globalValue, globalType, ht.hc.sc),
      vepRVD)))
  }

  def apply(ht: Table, config: String, csq: Boolean = false, blockSize: Int = 1000): Table =
    annotate(ht, config, csq, blockSize)
}
