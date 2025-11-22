package is.hail.methods

import is.hail.annotations._
import is.hail.backend.ExecuteContext
import is.hail.expr._
import is.hail.expr.ir.TableValue
import is.hail.expr.ir.functions.{RelationalFunctions, TableToTableFunction}
import is.hail.io.fs.FS
import is.hail.rvd.RVD
import is.hail.sparkextras.ContextRDD
import is.hail.types.physical.PType
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.utils.compat.immutable.ArraySeq
import is.hail.variant.{Locus, RegionValueVariant}
import is.hail.variant.VariantMethods.locusAllelesToString

import scala.collection.mutable
import scala.jdk.CollectionConverters._

import com.fasterxml.jackson.core.JsonParseException
import org.apache.spark.sql.Row
import org.json4s.{Formats, JValue}
import org.json4s.jackson.JsonMethods

case class VEPConfiguration(
  command: Array[String],
  env: Map[String, String],
  vep_json_schema: TStruct,
)

object VEP {

  def readConfiguration(fs: FS, path: String): VEPConfiguration = {
    val jv = using(fs.open(path))(in => JsonMethods.parse(in))
    implicit val formats: Formats = defaultJSONFormats + new TStructSerializer
    jv.extract[VEPConfiguration]
  }

  def apply(fs: FS, params: VEPParameters): VEP =
    new VEP(params, VEP.readConfiguration(fs, params.config))

  def apply(fs: FS, config: String, csq: Boolean, blockSize: Int, tolerateParseError: Boolean)
    : VEP =
    VEP(fs, VEPParameters(config, csq, blockSize, tolerateParseError))

  def fromJValue(fs: FS, jv: JValue): VEP = {
    implicit val formats: Formats = RelationalFunctions.formats
    VEP(fs, jv.extract[VEPParameters])
  }
}

case class VEPParameters(config: String, csq: Boolean, blockSize: Int, tolerateParseError: Boolean)

class VEP(private val params: VEPParameters, conf: VEPConfiguration)
    extends TableToTableFunction with Logging with Serializable {

  override def preservesPartitionCounts: Boolean = false

  override def typ(childType: TableType): TableType = {
    val vepType = if (params.csq) TArray(TString) else conf.vep_json_schema
    val globType = if (params.csq) TStruct("vep_csq_header" -> TString) else TStruct.empty
    val procIdType = TStruct("part_idx" -> TInt32, "block_idx" -> TInt32)
    TableType(
      childType.rowType ++ TStruct("vep" -> vepType, "vep_proc_id" -> procIdType),
      childType.key,
      globType,
    )
  }

  override def execute(ctx: ExecuteContext, tv: TableValue): TableValue = {
    val inType = tv.typ
    assert(inType.key == FastSeq("locus", "alleles"))
    assert(inType.rowType.size == 2)

    val inRvd = tv.rvd

    val cmd =
      conf.command.map { s =>
        if (s == "__OUTPUT_FORMAT_FLAG__")
          if (params.csq) "--vcf" else "--json"
        else s
      }

    val localRowType = inRvd.rowPType

    val csqPattern = "CSQ=[^;^\\t]+".r
    val annotations =
      inRvd.mapPartitionsWithIndex { (partIdx, _, it) =>
        val pb = new ProcessBuilder(cmd.toList.asJava)
        val env = pb.environment()
        for { (k, v) <- conf.env } env.put(k, v)

        val warnContext = new mutable.HashSet[String]

        val rvv = new RegionValueVariant(localRowType)
        it
          .map { ptr =>
            rvv.set(ptr)
            (rvv.locus(), rvv.alleles().to(ArraySeq): IndexedSeq[String])
          }
          .grouped(params.blockSize)
          .zipWithIndex
          .flatMap { case (block, blockIdx) =>
            val procID = Annotation(partIdx, blockIdx)
            val (jt, err, proc) = block.iterator.pipe(pb, printContext, printElement, _ => ())

            val nonStarToOriginalVariant =
              block
                .map(v => (v._1, v._2.filter(_ != "*")) -> v)
                .toMap

            val kt: Map[Annotation, Annotation] =
              jt
                .filter(s => s.nonEmpty && s.head != '#')
                .flatMap { s =>
                  if (params.csq) {
                    val vepv = variantFromInput(s)

                    val (locus, alleles) =
                      nonStarToOriginalVariant.getOrElse(
                        vepv,
                        fatal(
                          s"""VEP output variant ${locusAllelesToString(vepv)} not found in original variants.
                             |VEP output: $s""".stripMargin
                        ),
                      )

                    val a: Annotation =
                      csqPattern
                        .findFirstIn(s)
                        .map(_.substring(4).split(",").to(ArraySeq))
                        .getOrElse {
                          logger.warn(
                            s"""No CSQ INFO field for VEP output variant ${locusAllelesToString(vepv)}.
                               |VEP output: $s.""".stripMargin
                          )
                          null
                        }

                    Some((Annotation(locus, alleles), a))
                  } else {
                    val jsonOpt =
                      try Some(JsonMethods.parse(s))
                      catch {
                        case e: JsonParseException if params.tolerateParseError =>
                          logger.warn(
                            s"""VEP failed to produce parsable JSON!
                               |   json: $s
                               |  error: $e""".stripMargin
                          )
                          None
                      }

                    jsonOpt.map { json =>
                      val a =
                        JSONAnnotationImpex.importAnnotation(
                          json,
                          conf.vep_json_schema,
                          warnContext = warnContext,
                        )

                      val vepv =
                        Option(conf.vep_json_schema.query("input")(a))
                          .map { case input: String => variantFromInput(input) }
                          .getOrElse {
                            fatal(
                              s"""VEP generated null variant string
                                 |   json: $s
                                 | parsed: $a
                                 |""".stripMargin
                            )
                          }

                      val (locus, alleles) =
                        nonStarToOriginalVariant.getOrElse(
                          vepv,
                          fatal(
                            s"""VEP output variant ${locusAllelesToString(vepv)} not found in original variants.
                               |VEP output: $s""".stripMargin
                          ),
                        )

                      (Annotation(locus, alleles), a)
                    }
                  }
                }
                .toMap

            waitFor(proc, err, cmd)

            block.map { case (locus, alleles) =>
              val variant = Annotation(locus, alleles)
              val vepAnnotation = kt.get(variant).orNull
              (variant, vepAnnotation, procID)
            }
          }
      }

    val outType = typ(inType)

    val vepRowType = {
      val newVep = PType.canonical(outType.rowType.fieldType("vep"))
      val newProcId = PType.canonical(outType.rowType.fieldType("vep_proc_id"), true, true)
      inRvd.rowPType.appendKey("vep", newVep).appendKey("vep_proc_id", newProcId)
    }

    val vepRVD: RVD =
      RVD(
        inRvd.typ.copy(rowType = vepRowType),
        inRvd.partitioner,
        ContextRDD.weaken(annotations).cmapPartitions { (ctx, it) =>
          val rvb = ctx.rvb

          it.map { case (v, vep, proc) =>
            rvb.start(vepRowType)
            rvb.startStruct()
            rvb.addAnnotation(vepRowType.types(0).virtualType, v.asInstanceOf[Row].get(0))
            rvb.addAnnotation(vepRowType.types(1).virtualType, v.asInstanceOf[Row].get(1))
            rvb.addAnnotation(vepRowType.types(2).virtualType, vep)
            rvb.addAnnotation(vepRowType.types(3).virtualType, proc)
            rvb.endStruct()

            rvb.end()
          }
        },
      )

    val globalValue =
      if (params.csq)
        Row(getCSQHeaderDefinition(cmd, conf.env).getOrElse {
          logger.warn(f"Could not get VEP CSQ header, cmd = ${cmd.mkString(" ")}")
          ""
        })
      else
        Row()

    TableValue(ctx, outType, BroadcastRow(ctx, globalValue, outType.globalType), vepRVD)
  }

  def printContext(w: String => Unit): Unit = {
    w("##fileformat=VCFv4.1")
    w("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
  }

  def printElement(w: (String) => Unit, v: (Locus, IndexedSeq[String])): Unit = {
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

  def variantFromInput(input: String): (Locus, IndexedSeq[String]) =
    try {
      val a = input.split("\t")
      (Locus(a(0), a(1).toInt), a(3) +: a(4).split(","))
    } catch {
      case e: Throwable => fatal(s"VEP returned invalid variant '$input'", e)
    }

  def waitFor(proc: Process, err: StringBuilder, cmd: Array[String]): Unit = {
    val rc = proc.waitFor()
    if (rc != 0) {
      fatal(
        s"""VEP command '${cmd.mkString(" ")}' failed with non-zero exit status $rc.
           |VEP Error output:
           |$err""".stripMargin
      )
    }
  }

  def getCSQHeaderDefinition(cmd: Array[String], confEnv: Map[String, String]): Option[String] = {
    val csqHeaderRegex = "ID=CSQ[^>]+Description=\"([^\"]+)".r
    val pb = new ProcessBuilder(cmd.toList.asJava)
    val env = pb.environment()
    confEnv.foreach { case (key, value) => env.put(key, value) }

    val (jt, err, proc) =
      Iterator
        .single((Locus("1", 13372), FastSeq("G", "C")))
        .pipe(pb, printContext, printElement, _ => ())

    val csqHeader = jt.flatMap(csqHeaderRegex.findFirstMatchIn(_).map(_.group(1)))
    waitFor(proc, err, cmd)
    csqHeader.headOption
  }

  override def toJValue: JValue =
    decomposeWithName(params, "VEP")(RelationalFunctions.formats)

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: VEP => params == that.params
    case _ => false
  }
}
