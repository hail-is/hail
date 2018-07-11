package is.hail.io.bgen

import is.hail.expr.types._
import is.hail.utils.SerializableHadoopConfiguration
import is.hail.variant.{ Call2, ReferenceGenome }
import org.apache.hadoop.conf.Configuration
import org.apache.spark.{ Partition, SparkContext, TaskContext }
import org.apache.spark.rdd.RDD
import is.hail.annotations._
import is.hail.io._
import is.hail.rvd._
import is.hail.sparkextras._
import is.hail.utils._

import org.apache.hadoop.fs._

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

private case class BgenSettings (
  nSamples: Int,
  nVariants: Int,
  entries: EntriesSetting,
  rowFields: RowFields,
  rg: Option[ReferenceGenome],
  private val userContigRecoding: Map[String, String],
  skipInvalidLoci: Boolean
) {
  private[this] val typedRowFields = Array(
    (true, "locus" -> TLocus.schemaFromRG(rg)),
    (true, "alleles" -> TArray(TString())),
    (rowFields.rsid, "rsid" -> TString()),
    (rowFields.varid, "varid" -> TString()),
    (rowFields.fileRowIndex, "file_row_idx" -> TInt64()))
    .withFilter(_._1).map(_._2)

  private[this] val typedEntryFields: Array[(String, Type)] = entries match {
    case NoEntries => Array.empty
    case EntriesWithFields(gt, gp, dosage) => Array(
      (gt, "GT" -> TCall()),
      (gp, "GP" -> +TArray(+TFloat64())),
      (dosage, "dosage" -> +TFloat64()))
        .withFilter(_._1).map(_._2)
  }

  val matrixType: MatrixType = MatrixType.fromParts(
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
    minPartitions: Option[Int],
    includedVariantsPerFile: Map[String, Seq[Int]],
    settings: BgenSettings
  ): ContextRDD[RVDContext, RegionValue] =
    ContextRDD(
      new BgenRDD(sc, files, minPartitions, includedVariantsPerFile, settings))
}

private class BgenRDD(
  sc: SparkContext,
  files: Seq[BgenHeader],
  minPartitions: Option[Int],
  includedVariantsPerFile: Map[String, Seq[Int]],
  settings: BgenSettings
) extends RDD[RVDContext => Iterator[RegionValue]](sc, Nil) {
  private[this] val defaultMinPartitions =
    sc.defaultMinPartitions
  private[this] val parts = BgenRDDPartitions(
    sc,
    files,
    minPartitions.getOrElse(defaultMinPartitions),
    includedVariantsPerFile,
    settings)

  protected def getPartitions: Array[Partition] = parts

  def compute(split: Partition, context: TaskContext): Iterator[RVDContext => Iterator[RegionValue]] =
    Iterator.single { (ctx: RVDContext) =>
      FlipbookIterator(
        new BgenRecordStateMachine(
          ctx, split.asInstanceOf[BgenPartition], settings)) }
}

private class BgenRecordStateMachine (
  ctx: RVDContext,
  p: BgenPartition,
  settings: BgenSettings
) extends StateMachine[RegionValue] {
  private[this] val bfis = p.makeInputStream
  private[this] var rv = RegionValue(ctx.region)
  private[this] val rvb = ctx.rvb

  def isValid: Boolean = rv != null
  def value: RegionValue = rv
  def advance() {
    if (!p.hasNext(bfis)) {
      rv = null
      return
    }

    val rowFieldIndex = p.advance(bfis)

    val varid = if (settings.rowFields.varid) {
      bfis.readLengthAndString(2)
    } else {
      bfis.readLengthAndSkipString(2)
      null
    }
    val rsid = if (settings.rowFields.rsid) {
      bfis.readLengthAndString(2)
    } else {
      bfis.readLengthAndSkipString(2)
      null
    }
    val contig = bfis.readLengthAndString(2)
    val contigRecoded = settings.recodeContig(contig)
    val position = bfis.readInt()

    if (settings.skipInvalidLoci && !settings.rg.forall(_.isValidLocus(contigRecoded, position))) {
      val nAlleles = bfis.readShort()
      var i = 0
      while (i < nAlleles) {
        bfis.readLengthAndSkipString(4)
        i += 1
      }
      val dataSize = bfis.readInt()
      bfis.skipBytes(dataSize)
      advance()
    } else {
      rvb.start(settings.typ)
      rvb.startStruct() // record
      rvb.startStruct() // locus
      settings.rg.foreach(_.checkLocus(contigRecoded, position))
      rvb.addString(contigRecoded)
      rvb.addInt(position)
      rvb.endStruct()

      val nAlleles = bfis.readShort()
      if (nAlleles != 2)
        fatal(s"Only biallelic variants supported, found variant with $nAlleles")

      // alleles
      rvb.startArray(nAlleles)
      var i = 0
      while (i < nAlleles) {
        rvb.addString(bfis.readLengthAndString(4))
        i += 1
      }
      rvb.endArray()

      if (settings.rowFields.rsid)
        rvb.addString(rsid)
      if (settings.rowFields.varid)
        rvb.addString(varid)
      if (settings.rowFields.fileRowIndex)
        rvb.addLong(rowFieldIndex)

      val dataSize = bfis.readInt()

      readGenotypes(rvb, dataSize, nAlleles, varid)

      rvb.endStruct()
      rv.setOffset(rvb.end())
    }
  }

  private[this] val readGenotypes: (RegionValueBuilder, Int, Int, String) => Unit =
    settings.entries match {
      case NoEntries =>
        (rvb, dataSize, nAlleles, varid) => bfis.skipBytes(dataSize)

      case EntriesWithFields(gt, gp, dosage)
          if !(gt || gp || dosage) =>
        { (rvb, dataSize, nAlleles, varid) =>
          rvb.startArray(settings.nSamples)
          assert(rvb.currentType().byteSize == 0)
          rvb.unsafeAdvance(settings.nSamples)
          rvb.endArray()
          bfis.skipBytes(dataSize)
        }

      case EntriesWithFields(includeGT, includeGP, includeDosage) =>
        { (rvb, dataSize, nAlleles, varid) =>
          val data = if (p.compressed) {
            val uncompressedSize = bfis.readInt()
            val input = bfis.readBytes(dataSize - 4)
            decompress(input, uncompressedSize)
          } else {
            bfis.readBytes(dataSize)
          }
          val reader = new ByteArrayReader(data)

          val nRow = reader.readInt()
          if (nRow != settings.nSamples)
            fatal("row nSamples is not equal to header nSamples $nRow, $nSamples")

          val nAlleles2 = reader.readShort()
          if (nAlleles != nAlleles2)
            fatal(s"""Value for `nAlleles' in genotype probability data storage is
                 |not equal to value in variant identifying data. Expected
                 |$nAlleles but found $nAlleles2 at $varid.""".stripMargin)

          val minPloidy = reader.read()
          val maxPloidy = reader.read()

          if (minPloidy != 2 || maxPloidy != 2)
            fatal(s"Hail only supports diploid genotypes. Found min ploidy equals `$minPloidy' and max ploidy equals `$maxPloidy'.")

          var i = 0
          while (i < settings.nSamples) {
            val ploidy = reader.read()
            if ((ploidy & 0x3f) != 2)
              fatal(s"Ploidy value must equal to 2. Found $ploidy.")
            i += 1
          }

          val phase = reader.read()
          if (phase != 0 && phase != 1)
            fatal(s"Value for phase must be 0 or 1. Found $phase.")
          val isPhased = phase == 1

          if (isPhased)
            fatal("Hail does not support phased genotypes.")

          val nBitsPerProb = reader.read()
          if (nBitsPerProb < 1 || nBitsPerProb > 32)
            fatal(s"Value for nBits must be between 1 and 32 inclusive. Found $nBitsPerProb.")
          if (nBitsPerProb != 8)
            fatal(s"Only 8-bit probabilities supported, found $nBitsPerProb")

          val nGenotypes = triangle(nAlleles)

          val nExpectedBytesProbs = (settings.nSamples * (nGenotypes - 1) * nBitsPerProb + 7) / 8
          if (reader.length != nExpectedBytesProbs + settings.nSamples + 10)
            fatal(s"""Number of uncompressed bytes `${ reader.length }' does not
                 |match the expected size `$nExpectedBytesProbs'.""".stripMargin)

          val c0 = Call2.fromUnphasedDiploidGtIndex(0)
          val c1 = Call2.fromUnphasedDiploidGtIndex(1)
          val c2 = Call2.fromUnphasedDiploidGtIndex(2)

          rvb.startArray(settings.nSamples) // gs
          i = 0
          while (i < settings.nSamples) {
            val sampleMissing = (data(8 + i) & 0x80) != 0
            if (sampleMissing)
              rvb.setMissing()
            else {
              rvb.startStruct() // g

              val off = settings.nSamples + 10 + 2 * i
              val d0 = data(off) & 0xff
              val d1 = data(off + 1) & 0xff
              val d2 = 255 - d0 - d1

              if (includeGT) {
                if (d0 > d1) {
                  if (d0 > d2)
                    rvb.addInt(c0)
                  else if (d2 > d0)
                    rvb.addInt(c2)
                  else {
                    // d0 == d2
                    rvb.setMissing()
                  }
                } else {
                  // d0 <= d1
                  if (d2 > d1)
                    rvb.addInt(c2)
                  else {
                    // d2 <= d1
                    if (d1 == d0 || d1 == d2)
                      rvb.setMissing()
                    else
                      rvb.addInt(c1)
                  }
                }
              }

              if (includeGP) {
                rvb.startArray(3) // GP
                rvb.addDouble(d0 / 255.0)
                rvb.addDouble(d1 / 255.0)
                rvb.addDouble(d2 / 255.0)
                rvb.endArray()
              }

              if (includeDosage) {
                val dosage = (d1 + (d2 << 1)) / 255.0
                rvb.addDouble(dosage)
              }

              rvb.endStruct() // g
            }
            i += 1
          }
          rvb.endArray()
        }
    }

  advance() // make sure iterator is initialized in first valid state
}
