package is.hail.methods

import java.io.BufferedInputStream

import com.fasterxml.jackson.core.JsonParseException
import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr._
import is.hail.expr.ir.{ExecuteContext, TableValue}
import is.hail.expr.ir.functions.TableToTableFunction
import is.hail.expr.types._
import is.hail.expr.types.physical.{PStruct, PType}
import is.hail.expr.types.virtual._
import is.hail.methods.VEP._
import is.hail.rvd.{RVD, RVDContext, RVDType}
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.variant.{Locus, RegionValueVariant, VariantMethods}
import is.hail.io.fs.FS
import org.apache.spark.sql.Row
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._
import scala.io.Source

case class VEPConfiguration(
  command: Array[String],
  env: Map[String, String],
  vep_json_schema: TStruct)

object VEP {
  def readConfiguration(fs: FS, path: String): VEPConfiguration = {
    val jv = fs.readFile(path) { in =>
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

  def waitFor(proc: Process, cmd: Array[String]): Unit = {
    val rc = proc.waitFor()

    if (rc != 0) {
      val errorLines = Source.fromInputStream(new BufferedInputStream(proc.getErrorStream)).getLines().mkString("\n")

      fatal(s"VEP command '${ cmd.mkString(" ") }' failed with non-zero exit status $rc\n" +
        "  VEP Error output:\n" + errorLines)
    }
  }

  def getCSQHeaderDefinition(cmd: Array[String], confEnv: Map[String, String]): Option[String] = {
    val csqHeaderRegex = "ID=CSQ[^>]+Description=\"([^\"]+)".r
    val pb = new ProcessBuilder(cmd.toList.asJava)
    val env = pb.environment()
    confEnv.foreach { case (key, value) => env.put(key, value) }

    val (jt, proc) = List((Locus("1", 13372), FastIndexedSeq("G", "C"))).iterator.pipe(pb,
      printContext,
      printElement,
      _ => ())

    val csqHeader = jt.flatMap(s => csqHeaderRegex.findFirstMatchIn(s).map(m => m.group(1)))
    waitFor(proc, cmd)

    if (csqHeader.hasNext)
      Some(csqHeader.next())
    else {
      warn("could not get VEP CSQ header")
      None
    }
  }
}

case class VEP(config: String, csq: Boolean, blockSize: Int) extends TableToTableFunction {
  private lazy val conf = VEP.readConfiguration(HailContext.sFS, config)
  private lazy val vepSignature = conf.vep_json_schema

  override def preservesPartitionCounts: Boolean = false

  override def typ(childType: TableType): TableType = {
    val vepType = if (csq) TArray(TString()) else vepSignature
    TableType(childType.rowType ++ TStruct("vep" -> vepType), childType.key, childType.globalType)
  }

  override def execute(ctx: ExecuteContext, tv: TableValue): TableValue = {
    assert(tv.typ.key == FastIndexedSeq("locus", "alleles"))
    assert(tv.typ.rowType.size == 2)

    val conf = readConfiguration(HailContext.sFS, config)
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

    val localRowType = tv.rvd.rowPType
    val rowKeyOrd = tv.typ.keyType.ordering

    val prev = tv.rvd
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
              .flatMap { s =>
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
                      Some((Annotation(locus, alleles), a))
                    case None =>
                      fatal(s"VEP output variant ${ VariantMethods.locusAllelesToString(vepLocus, vepAlleles) } not found in original variants.\nVEP output: $s")
                  }
                } else {
                  try {
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
                        Some((Annotation(locus, alleles), a))
                      case None =>
                        fatal(s"VEP output variant ${ VariantMethods.locusAllelesToString(vepLocus, vepAlleles) } not found in original variants.\nVEP output: $s")
                    }
                  } catch {
                    case e: JsonParseException =>
                      log.warn(s"VEP failed to produce parsable JSON!\n  json: $s\n  error: $e")
                      None
                  }
                }
              }

            val r = kt.toArray
              .sortBy(_._1)(rowKeyOrd.toOrdering)

            waitFor(proc, cmd)

            r
          }
      }

    val vepType: Type = if (csq) TArray(TString()) else vepSignature

    val vepRVDType = prev.typ.copy(rowType = PType.canonical(prev.rowType ++ TStruct("vep" -> vepType)).asInstanceOf[PStruct])

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
          rvb.addAnnotation(vepRowType.types(0).virtualType, v.asInstanceOf[Row].get(0))
          rvb.addAnnotation(vepRowType.types(1).virtualType, v.asInstanceOf[Row].get(1))
          rvb.addAnnotation(vepRowType.types(2).virtualType, vep)
          rvb.endStruct()
          rv.setOffset(rvb.end())
          rv
        }
      })

    val (globalValue, globalType) =
      if (csq)
        (Row(csqHeader.getOrElse("")), TStruct("vep_csq_header" -> TString()))
      else
        (Row(), TStruct())

    TableValue(
      TableType(vepRowType.virtualType, FastIndexedSeq("locus", "alleles"), globalType),
      BroadcastRow(ctx, globalValue, globalType),
      vepRVD)
  }
}
