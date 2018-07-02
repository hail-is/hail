package is.hail.io.bgen

import is.hail.asm4s._
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
  skipInvalidLoci: Boolean,
  checkPloidy: Boolean
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

  private[bgen] def decompress(
    input: Array[Byte],
    uncompressedSize: Int
  ): Array[Byte] = is.hail.utils.decompress(input, uncompressedSize)

  private[bgen] def triangle(
    x: Int
  ): Int = is.hail.utils.triangle(x)
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
      new BgenRecordIterator(ctx, split.asInstanceOf[BgenPartition], settings).flatten }
}

private class BgenRecordIterator(
  ctx: RVDContext,
  p: BgenPartition,
  settings: BgenSettings
) extends Iterator[Option[RegionValue]] {
  private[this] val bfis = p.makeInputStream
  private[this] var read: Boolean = false
  private[this] val rv = RegionValue(ctx.region)
  private[this] val fb = new Function4Builder[Region, BgenPartition, HadoopFSDataBinaryReader, BgenSettings, Long]
  private[this] val mb = fb.apply_method
  private[this] val cp = mb.getArg[BgenPartition](2).load()
  private[this] val cbfis = mb.getArg[HadoopFSDataBinaryReader](3).load()
  private[this] val csettings = mb.getArg[BgenSettings](4).load()
  private[this] val srvb = new StagedRegionValueBuilder(mb, settings.typ)
  private[this] val fileRowIndex = mb.newLocal[Long]
  private[this] val varid = mb.newLocal[String]
  private[this] val rsid = mb.newLocal[String]
  private[this] val contig = mb.newLocal[String]
  private[this] val contigRecoded = mb.newLocal[String]
  private[this] val position = mb.newLocal[Int]
  private[this] val nAlleles = mb.newLocal[Int]
  private[this] val i = mb.newLocal[Int]
  private[this] val dataSize = mb.newLocal[Int]
  private[this] val invalidLocus = mb.newLocal[Boolean]
  private[this] val data = mb.newLocal[Array[Byte]]
  private[this] val uncompressedSize = mb.newLocal[Int]
  private[this] val input = mb.newLocal[Array[Byte]]
  private[this] val reader = mb.newLocal[ByteArrayReader]
  private[this] val nRow = mb.newLocal[Int]
  private[this] val nAlleles2 = mb.newLocal[Int]
  private[this] val minPloidy = mb.newLocal[Int]
  private[this] val maxPloidy = mb.newLocal[Int]
  private[this] val longPloidy = mb.newLocal[Long]
  private[this] val ploidy = mb.newLocal[Int]
  private[this] val phase = mb.newLocal[Int]
  private[this] val nBitsPerProb = mb.newLocal[Int]
  private[this] val nGenotypes = mb.newLocal[Int]
  private[this] val nExpectedBytesProbs = mb.newLocal[Int]
  private[this] val c0 = mb.newLocal[Int]
  private[this] val c1 = mb.newLocal[Int]
  private[this] val c2 = mb.newLocal[Int]
  private[this] val off = mb.newLocal[Int]
  private[this] val d0 = mb.newLocal[Int]
  private[this] val d1 = mb.newLocal[Int]
  private[this] val d2 = mb.newLocal[Int]
  private[this] val c = Code(
    fileRowIndex := cp.invoke[HadoopFSDataBinaryReader, Long]("advance", cbfis),
    if (settings.rowFields.varid) {
      varid := cbfis.invoke[Int, String]("readLengthAndString", 2)
    } else {
      cbfis.invoke[Int, Unit]("readLengthAndSkipString", 2)
    },
    if (settings.rowFields.rsid) {
      rsid := cbfis.invoke[Int, String]("readLengthAndString", 2)
    } else {
      cbfis.invoke[Int, Unit]("readLengthAndSkipString", 2)
    },
    contig := cbfis.invoke[Int, String]("readLengthAndString", 2),
    contigRecoded := csettings.invoke[String, String]("recodeContig", contig),
    position := cbfis.invoke[Int]("readInt"),
    if (settings.skipInvalidLoci) {
      Code(
        invalidLocus :=
          (if (settings.rg.nonEmpty) {
            !csettings.invoke[Option[ReferenceGenome]]("rg")
              .invoke[ReferenceGenome]("get")
              .invoke[String, Int, Boolean]("isValidLocus", contigRecoded, position)
          } else false),
        invalidLocus.mux(
          Code(
            nAlleles := cbfis.invoke[Int]("readShort"),
            i := 0,
            Code.whileLoop(i < nAlleles,
              cbfis.invoke[Int, Unit]("readLengthAndSkipString", 4),
              i := i + 1
            ),
            dataSize := cbfis.invoke[Int]("readInt"),
            Code.toUnit(cbfis.invoke[Long, Long]("skipBytes", dataSize.toL)),
            Code._return(-1L) // return -1 to indicate we are skipping this variant
          ),
          Code._empty // if locus is valid, continue
        ))
    } else {
      if (settings.rg.nonEmpty) {
        // verify the locus is valid before continuing
        csettings.invoke[Option[ReferenceGenome]]("rg")
          .invoke[ReferenceGenome]("get")
          .invoke[String, Int, Unit]("checkLocus", contigRecoded, position)
      } else {
        Code._empty // if locus is valid continue
      }
    },
    srvb.start(),
    srvb.addBaseStruct(settings.typ.types(settings.typ.fieldIdx("locus")).fundamentalType.asInstanceOf[TBaseStruct], { srvb =>
      Code(
        srvb.start(),
        srvb.addString(contigRecoded),
        srvb.advance(),
        srvb.addInt(position))
    }),
    srvb.advance(),
    nAlleles := cbfis.invoke[Int]("readShort"),
    nAlleles.cne(2).mux(
      Code._fatal(
        const("Only biallelic variants supported, found variant with ")
          .concat(nAlleles.toS)),
      Code._empty),
    srvb.addArray(settings.typ.types(settings.typ.fieldIdx("alleles")).asInstanceOf[TArray], { srvb =>
      Code(
        srvb.start(nAlleles),
        i := 0,
        Code.whileLoop(i < nAlleles,
          srvb.addString(cbfis.invoke[Int, String]("readLengthAndString", 4)),
          srvb.advance(),
          i := i + 1))
    }),
    srvb.advance(),
    if (settings.rowFields.rsid)
      Code(srvb.addString(rsid), srvb.advance())
    else Code._empty,
    if (settings.rowFields.varid)
      Code(srvb.addString(varid), srvb.advance())
    else Code._empty,
    if (settings.rowFields.fileRowIndex)
      Code(srvb.addLong(fileRowIndex), srvb.advance())
    else Code._empty,
    dataSize := cbfis.invoke[Int]("readInt"),
    settings.entries match {
      case NoEntries =>
        Code.toUnit(cbfis.invoke[Long, Long]("skipBytes", dataSize.toL))

      case EntriesWithFields(gt, gp, dosage)
          if !(gt || gp || dosage) =>
        assert(settings.matrixType.entryType.byteSize == 0)
        Code(
          srvb.addArray(settings.matrixType.entryArrayType, { srvb =>
            Code(
              srvb.start(settings.nSamples),
              i := 0,
              Code.whileLoop(i < settings.nSamples,
                srvb.advance(),
                i := i + 1))
          }),
          srvb.advance(),
          Code.toUnit(cbfis.invoke[Long, Long]("skipBytes", dataSize.toL)))

      case EntriesWithFields(includeGT, includeGP, includeDosage) =>
        Code(
          if (p.compressed) {
            Code(
              uncompressedSize := cbfis.invoke[Int]("readInt"),
              input := cbfis.invoke[Int, Array[Byte]]("readBytes", dataSize - 4),
              data := Code.invokeScalaObject[Array[Byte], Int, Array[Byte]](
                BgenRDD.getClass, "decompress", input, uncompressedSize))
          } else {
            data := cbfis.invoke[Int, Array[Byte]]("readBytes", dataSize)
          },
          reader := Code.newInstance[ByteArrayReader, Array[Byte]](data),
          nRow := reader.invoke[Int]("readInt"),
          (nRow.cne(settings.nSamples)).mux(
            Code._fatal(
              const("row nSamples is not equal to header nSamples ")
                .concat(nRow.toS)
                .concat(", ")
                .concat(settings.nSamples.toString)),
            Code._empty),
          nAlleles2 := reader.invoke[Int]("readShort"),
          (nAlleles.cne(nAlleles2)).mux(
            Code._fatal(
              const("""Value for `nAlleles' in genotype probability data storage is
                        |not equal to value in variant identifying data. Expected""".stripMargin)
                .concat(nAlleles.toS)
                .concat(" but found ")
                .concat(nAlleles2.toS)
                .concat(" at ")
                .concat(contig)
                .concat(":")
                .concat(position.toS)
                .concat(".")),
            Code._empty),
          minPloidy := reader.invoke[Int]("read"),
          maxPloidy := reader.invoke[Int]("read"),
          (minPloidy.cne(2) || maxPloidy.cne(2)).mux(
            Code._fatal(
              const("Hail only supports diploid genotypes. Found min ploidy equals `")
                .concat(minPloidy.toS)
                .concat("' and max ploidy equals `")
                .concat(maxPloidy.toS)
                .concat("'.")),
            Code._empty),
          i := 0,
          if (settings.checkPloidy) {
            Code(
              Code.whileLoop(i < (settings.nSamples - 8),
                longPloidy := reader.invoke[Long]("readLong"),
                ((longPloidy & 0x3f3f3f3f3f3f3f3fL).cne(0x0202020202020202L)).mux(
                  Code._fatal(
                    const("Ploidy value must equal to 2. Found ")
                      .concat(longPloidy.toS)
                      .concat(" somewhere between sample ")
                      .concat(i.toS)
                      .concat(" and ")
                      .concat((i + 8).toS)
                      .concat(" at ")
                      .concat(contig)
                      .concat(":")
                      .concat(position.toS)
                      .concat(".")),
                  Code._empty),
                i += 8
              ),
              Code.whileLoop(i < settings.nSamples,
                ploidy := reader.invoke[Int]("read"),
                ((ploidy & 0x3f).cne(2)).mux(
                  Code._fatal(
                    const("Ploidy value must equal to 2. Found ")
                      .concat(ploidy.toS)
                      .concat(".")),
                  Code._empty),
                i += 1
              ))
          } else {
            Code.toUnit(reader.invoke[Long, Long]("skipBytes", settings.nSamples))
          },
          phase := reader.invoke[Int]("read"),
          (phase.cne(0) && phase.cne(1)).mux(
            Code._fatal(
              const("Value for phase must be 0 or 1. Found ")
                .concat(phase.toS)
                .concat(".")),
            Code._empty),
          phase.ceq(1).mux(
            Code._fatal("Hail does not support phased genotypes."),
            Code._empty),
          nBitsPerProb := reader.invoke[Int]("read"),
          (nBitsPerProb < 1 || nBitsPerProb > 32).mux(
            Code._fatal(
              const("Value for nBits must be between 1 and 32 inclusive. Found ")
                .concat(nBitsPerProb.toS)
                .concat(".")),
            Code._empty),
          (nBitsPerProb.cne(8)).mux(
            Code._fatal(
              const("Only 8-bit probabilities supported, found ")
                .concat(nBitsPerProb.toS)
                .concat(".")),
            Code._empty),
          nGenotypes := Code.invokeScalaObject[Int, Int](
            BgenRDD.getClass, "triangle", nAlleles),
          nExpectedBytesProbs := (const(settings.nSamples) * (nGenotypes - 1) * nBitsPerProb + 7) / 8,
          (reader.invoke[Int]("length").cne(nExpectedBytesProbs + settings.nSamples + 10)).mux(
            Code._fatal(
              const("Number of uncompressed bytes `")
                .concat(reader.invoke[Int]("length").toS)
                .concat("' does not match the expected size `")
                .concat(nExpectedBytesProbs.toS)
                .concat("'.")),
            Code._empty),
          c0 := Call2.fromUnphasedDiploidGtIndex(0),
          c1 := Call2.fromUnphasedDiploidGtIndex(1),
          c2 := Call2.fromUnphasedDiploidGtIndex(2),
          srvb.addArray(settings.matrixType.entryArrayType, { srvb =>
            Code(
              srvb.start(settings.nSamples),
              i := 0,
              Code.whileLoop(i < settings.nSamples,
                (data(i + 8) & 0x80).cne(0).mux(
                  srvb.setMissing(),
                  srvb.addBaseStruct(settings.matrixType.entryType, { srvb =>
                    Code(
                      srvb.start(),
                      off := const(settings.nSamples + 10) + i * 2,
                      d0 := data(off) & 0xff,
                      d1 := data(off + 1) & 0xff,
                      d2 := const(255) - d0 - d1,
                      if (includeGT) {
                        Code(
                          (d0 > d1).mux(
                            (d0 > d2).mux(
                              srvb.addInt(c0),
                              (d2 > d0).mux(
                                srvb.addInt(c2),
                                // d0 == d2
                                srvb.setMissing())),
                            // d0 <= d1
                            (d2 > d1).mux(
                              srvb.addInt(c2),
                              // d2 <= d1
                              (d1.ceq(d0) || d1.ceq(d2)).mux(
                                srvb.setMissing(),
                                srvb.addInt(c1)))),
                          srvb.advance())
                      } else Code._empty,
                      if (includeGP) {
                        Code(
                          srvb.addArray(settings.matrixType.entryType.types(settings.matrixType.entryType.fieldIdx("GP")).asInstanceOf[TArray], { srvb =>
                            Code(
                              srvb.start(3),
                              srvb.addDouble(d0.toD / 255.0),
                              srvb.advance(),
                              srvb.addDouble(d1.toD / 255.0),
                              srvb.advance(),
                              srvb.addDouble(d2.toD / 255.0),
                              srvb.advance())
                          }),
                          srvb.advance())
                      } else Code._empty,
                      if (includeDosage) {
                        val dosage = (d1 + (d2 << 1)).toD / 255.0
                        Code(
                          srvb.addDouble(dosage),
                          srvb.advance())
                      } else Code._empty)
                    })),
                srvb.advance(),
                i := i + 1))
          }))
    },
    srvb.end())

  mb.emit(c)
  private[this] val compiledNext = fb.result()()
  def next(): Option[RegionValue] = {
    val maybeOffset = compiledNext(ctx.region, p, bfis, settings)
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
