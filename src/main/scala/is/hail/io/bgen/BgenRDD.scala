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
  offset: Boolean
)

case class BgenSettings(
  nSamples: Int,
  entries: EntriesSetting,
  rowFields: RowFields,
  rg: Option[ReferenceGenome],
  private val userContigRecoding: Map[String, String],
  skipInvalidLoci: Boolean,
  createIndex: Boolean
) {
  private[this] val typedRowFields = Array(
    (true, "locus" -> TLocus.schemaFromRG(rg)),
    (true, "alleles" -> TArray(TString())),
    (rowFields.rsid, "rsid" -> TString()),
    (rowFields.varid, "varid" -> TString()),
    (rowFields.offset, "offset" -> TInt64()))
    .withFilter(_._1).map(_._2)

  private[this] val typedEntryFields: Array[(String, Type)] = entries match {
    case NoEntries => Array.empty
    case EntriesWithFields(gt, gp, dosage) => Array(
      (gt, "GT" -> TCall()),
      (gp, "GP" -> +TArray(+TFloat64())),
      (dosage, "dosage" -> +TFloat64()))
        .withFilter(_._1).map(_._2)
  }

  val matrixType: MatrixType =
    MatrixType.fromParts(
      globalType = TStruct.empty(),
      colKey = Array("s"),
      colType = TStruct("s" -> TString()),
      rowType = TStruct(typedRowFields: _*),
      rowKey = Array("locus", "alleles"),
      rowPartitionKey = Array("locus"),
      entryType = TStruct(typedEntryFields: _*))

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
    includedOffsetsPerFile: Map[String, Array[Long]],
    settings: BgenSettings
  ): ContextRDD[RVDContext, RegionValue] =
    ContextRDD(
      new BgenRDD(sc, files, fileNPartitions, includedOffsetsPerFile, settings))

  private[bgen] def decompress(
    input: Array[Byte],
    uncompressedSize: Int
  ): Array[Byte] = is.hail.utils.decompress(input, uncompressedSize)
}

private class BgenRDD(
  sc: SparkContext,
  files: Seq[BgenHeader],
  fileNPartitions: Array[Int],
  includedOffsetsPerFile: Map[String, Array[Long]],
  settings: BgenSettings
) extends RDD[RVDContext => Iterator[RegionValue]](sc, Nil) {
  private[this] val f = CompileDecoder(settings)
  private[this] val parts = BgenRDDPartitions(
    sc,
    files,
    fileNPartitions,
    includedOffsetsPerFile,
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
