package is.hail.io.bgen

import is.hail.annotations._
import is.hail.asm4s.AsmFunction4
import is.hail.expr.types._
import is.hail.io.HadoopFSDataBinaryReader
import is.hail.rvd._
import is.hail.sparkextras._
import is.hail.variant.ReferenceGenome
import org.apache.spark.rdd.RDD
import org.apache.spark.{Partition, SparkContext, TaskContext}

sealed trait EntriesSetting
final case object NoEntries extends EntriesSetting
final case class EntriesWithFields (
  gt: Boolean,
  gp: Boolean,
  dosage: Boolean
) extends EntriesSetting

sealed case class RowFields (
  varid: Boolean,
  rsid: Boolean,
  fileRowIndex: Boolean
)

case class BgenSettings(
  nSamples: Int,
  entries: EntriesSetting,
  rowFields: RowFields,
  rg: Option[ReferenceGenome],
  private val userContigRecoding: Map[String, String],
  skipInvalidLoci: Boolean
) {
  val (includeGT, includeGP, includeDosage) = entries match {
    case NoEntries => (false, false, false)
    case EntriesWithFields(gt, gp, dosage) => (gt, gp, dosage)
  }

  val matrixType = MatrixBGENReader.getMatrixType(
    rg,
    rowFields.rsid,
    rowFields.varid,
    rowFields.fileRowIndex,
    includeGT,
    includeGP,
    includeDosage
  )

  val typ: TStruct = entries match {
    case NoEntries =>
      matrixType.rowType
    case _: EntriesWithFields =>
      matrixType.rvRowType
  }

  def recodeContig(bgenContig: String): String = {
    val hailContig = bgenContig match {
      case "23" => "X"
      case "24" => "Y"
      case "25" => "X"
      case "26" => "MT"
      case x => x
    }
    userContigRecoding.getOrElse(hailContig, hailContig)
  }
}

object BgenRDD {
  def apply(
    sc: SparkContext,
    files: Seq[BgenHeader],
    fileNPartitions: Array[Int],
    includedVariantsPerFile: Map[String, Seq[Int]],
    settings: BgenSettings
  ): ContextRDD[RVDContext, RegionValue] =
    ContextRDD(
      new BgenRDD(sc, files, fileNPartitions, includedVariantsPerFile, settings))

  private[bgen] def decompress(
    input: Array[Byte],
    uncompressedSize: Int
  ): Array[Byte] = is.hail.utils.decompress(input, uncompressedSize)
}

private class BgenRDD(
  sc: SparkContext,
  files: Seq[BgenHeader],
  fileNPartitions: Array[Int],
  includedVariantsPerFile: Map[String, Seq[Int]],
  settings: BgenSettings
) extends RDD[RVDContext => Iterator[RegionValue]](sc, Nil) {
  private[this] val f = CompileDecoder(settings)
  private[this] val parts = BgenRDDPartitions(
    sc,
    files,
    fileNPartitions,
    includedVariantsPerFile,
    settings)

  protected def getPartitions: Array[Partition] = parts

  def compute(split: Partition, context: TaskContext): Iterator[RVDContext => Iterator[RegionValue]] =
    Iterator.single { (ctx: RVDContext) =>
      new BgenRecordIterator(ctx, split.asInstanceOf[BgenPartition], settings, f()).flatten }
}

private class BgenRecordIterator(
  ctx: RVDContext,
  p: BgenPartition,
  settings: BgenSettings,
  f: AsmFunction4[Region, BgenPartition, HadoopFSDataBinaryReader, BgenSettings, Long]
) extends Iterator[Option[RegionValue]] {
  private[this] val bfis = p.makeInputStream
  private[this] val rv = RegionValue(ctx.region)
  def next(): Option[RegionValue] = {
    val maybeOffset = f(ctx.region, p, bfis, settings)
    if (maybeOffset == -1) {
      None
    } else {
      rv.setOffset(maybeOffset)
      Some(rv)
    }
  }

  def hasNext(): Boolean =
    p.hasNext(bfis)
}
