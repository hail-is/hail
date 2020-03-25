package is.hail.io.bgen

import is.hail.HailContext
import is.hail.annotations.{Region, _}
import is.hail.asm4s.{coerce, _}
import is.hail.expr.ir.{EmitFunctionBuilder, EmitMethodBuilder, EmitRegion}
import is.hail.expr.ir.functions.StringFunctions
import is.hail.expr.types._
import is.hail.expr.types.physical.{PArray, PStruct, PType}
import is.hail.expr.types.virtual.{TArray, TInterval, Type}
import is.hail.io.index.{IndexReader, IndexReaderBuilder}
import is.hail.io.{ByteArrayReader, HadoopFSDataBinaryReader}
import is.hail.utils._
import is.hail.variant.{Call2, ReferenceGenome}
import is.hail.io.fs.FS
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.{Partition, SparkContext}

trait BgenPartition extends Partition {
  def path: String

  def compressed: Boolean

  def skipInvalidLoci: Boolean

  def contigRecoding: Map[String, String]

  def bcFS: Broadcast[FS]

  def makeInputStream: HadoopFSDataBinaryReader = {
    val fileSystem = bcFS.value.fileSystem(path)
    val bfis = new HadoopFSDataBinaryReader(fileSystem.open)
    bfis
  }

  def recodeContig(contig: String): String = contigRecoding.getOrElse(contig, contig)
}

private case class LoadBgenPartition(
  path: String,
  indexPath: String,
  filterPartition: Partition,
  compressed: Boolean,
  skipInvalidLoci: Boolean,
  contigRecoding: Map[String, String],
  partitionIndex: Int,
  startIndex: Long,
  endIndex: Long,
  bcFS: Broadcast[FS]
) extends BgenPartition {
  assert(startIndex <= endIndex)

  def index = partitionIndex
}

object BgenRDDPartitions extends Logging {
  def checkFilesDisjoint(fs: FS, fileMetadata: Seq[BgenFileMetadata], keyType: Type): Array[Interval] = {
    assert(fileMetadata.nonEmpty)
    val pord = keyType.ordering
    val bounds = fileMetadata.map(md => (md.path, md.rangeBounds))

    val overlappingBounds = new ArrayBuilder[(String, Interval, String, Interval)]
    var i = 0
    while (i < bounds.length) {
      var j = 0
      while (j < i) {
        val b1 = bounds(i)
        val b2 = bounds(j)
        if (!b1._2.isDisjointFrom(pord, b2._2))
          overlappingBounds += ((b1._1, b1._2, b2._1, b2._2))
        j += 1
      }
      i += 1
    }

    if (!overlappingBounds.isEmpty)
      fatal(
        s"""Each BGEN file must contain a region of the genome disjoint from other files. Found the following overlapping files:
          |  ${ overlappingBounds.result().map { case (f1, i1, f2, i2) =>
          s"file1: $f1\trangeBounds1: $i1\tfile2: $f2\trangeBounds2: $i2"
        }.mkString("\n  ") })""".stripMargin)

    bounds.map(_._2).toArray
  }

  def apply(
    rg: Option[ReferenceGenome],
    files: Seq[BgenFileMetadata],
    blockSizeInMB: Option[Int],
    nPartitions: Option[Int],
    keyType: Type
  ): (Array[Partition], Array[Interval]) = {
    val hc = HailContext.get
    val fs = hc.sFS
    val bcFS = hc.bcFS

    val fileRangeBounds = checkFilesDisjoint(fs, files, keyType)
    val intervalOrdering = TInterval(keyType).ordering

    val sortedFiles = files.zip(fileRangeBounds)
      .sortWith { case ((_, i1), (_, i2)) => intervalOrdering.lt(i1, i2) }
      .map(_._1)

    val totalSize = sortedFiles.map(_.header.fileByteSize).sum

    val fileNPartitions = (blockSizeInMB, nPartitions) match {
      case (Some(blockSizeInMB), _) =>
        val blockSizeInB = blockSizeInMB * 1024 * 1024
        sortedFiles.map { md =>
          val size = md.header.fileByteSize
          ((size + blockSizeInB - 1) / blockSizeInB).toInt
        }
      case (_, Some(nParts)) =>
        sortedFiles.map { md =>
          val size = md.header.fileByteSize
          ((size * nParts + totalSize - 1) / totalSize).toInt
        }
      case (None, None) => fatal(s"Must specify either of 'blockSizeInMB' or 'nPartitions'.")
    }

    val nonEmptyFilesAfterFilter = sortedFiles.filter(_.nVariants > 0)

    val indexReaderBuilder = {
      val (leafCodec, internalNodeCodec) = BgenSettings.indexCodecSpecs(rg)
      val (leafPType: PStruct, leafDec) = leafCodec.buildDecoder(leafCodec.encodedVirtualType)
      val (intPType: PStruct, intDec) = internalNodeCodec.buildDecoder(internalNodeCodec.encodedVirtualType)
      IndexReaderBuilder.withDecoders(leafDec, intDec, BgenSettings.indexKeyType(rg), BgenSettings.indexAnnotationType, leafPType, intPType)
    }
    if (nonEmptyFilesAfterFilter.isEmpty) {
      (Array.empty, Array.empty)
    } else {
      val partitions = new ArrayBuilder[Partition]()
      val rangeBounds = new ArrayBuilder[Interval]()
      var fileIndex = 0
      while (fileIndex < nonEmptyFilesAfterFilter.length) {
        val file = nonEmptyFilesAfterFilter(fileIndex)
        using(indexReaderBuilder(fs, file.indexPath, 8)) { index =>
          val nPartitions = math.min(fileNPartitions(fileIndex), file.nVariants.toInt)
          val partNVariants = partition(file.nVariants.toInt, nPartitions)
          val partFirstVariantIndex = partNVariants.scan(0)(_ + _).init
          var i = 0
          while (i < nPartitions) {
            val firstVariantIndex = partFirstVariantIndex(i)
            val lastVariantIndex = firstVariantIndex + partNVariants(i)
            val partitionIndex = partitions.length

            partitions += LoadBgenPartition(
              file.path,
              file.indexPath,
              filterPartition = null,
              file.header.compressed,
              file.skipInvalidLoci,
              file.contigRecoding,
              partitionIndex,
              firstVariantIndex,
              lastVariantIndex,
              bcFS
            )

            rangeBounds += Interval(
                index.queryByIndex(firstVariantIndex).key,
                index.queryByIndex(lastVariantIndex - 1).key,
                includesStart = true,
                includesEnd = true) // this must be true -- otherwise boundaries with duplicates will have the wrong range bounds

            i += 1
          }
        }
        fileIndex += 1
      }
      (partitions.result(), rangeBounds.result())
    }
  }
}

object CompileDecoder {
  def apply(
    settings: BgenSettings
  ): (Int, Region) => AsmFunction4[Region, BgenPartition, HadoopFSDataBinaryReader, BgenSettings, Long] = {
    val fb = EmitFunctionBuilder[Region, BgenPartition, HadoopFSDataBinaryReader, BgenSettings, Long]("bgen_rdd_decoder")
    val mb = fb.apply_method
    val region = mb.getArg[Region](1)
    val cp = mb.getArg[BgenPartition](2)
    val cbfis = mb.getArg[HadoopFSDataBinaryReader](3)
    val csettings = mb.getArg[BgenSettings](4)

    val regionField = mb.genFieldThisRef[Region]("region")
    val srvb = new StagedRegionValueBuilder(mb, settings.rowPType, regionField)

    val offset = mb.newLocal[Long]("offset")
    val fileIdx = mb.newLocal[Int]("fileIdx")
    val varid = mb.newLocal[String]("varid")
    val rsid = mb.newLocal[String]("rsid")
    val contig = mb.newLocal[String]("contig")
    val contigRecoded = mb.newLocal[String]("contigRecoded")
    val position = mb.newLocal[Int]("position")
    val nAlleles = mb.newLocal[Int]("nAlleles")
    val i = mb.newLocal[Int]("i")
    val dataSize = mb.newLocal[Int]("dataSize")
    val invalidLocus = mb.newLocal[Boolean]("invalidLocus")
    val data = mb.newLocal[Array[Byte]]("data")
    val uncompressedSize = mb.newLocal[Int]("uncompressedSize")
    val input = mb.newLocal[Array[Byte]]("input")
    val reader = mb.newLocal[ByteArrayReader]("reader")
    val nRow = mb.newLocal[Int]("nRow")
    val nAlleles2 = mb.newLocal[Int]("nAlleles2")
    val minPloidy = mb.newLocal[Int]("minPloidy")
    val maxPloidy = mb.newLocal[Int]("maxPloidy")
    val longPloidy = mb.newLocal[Long]("longPloidy")
    val ploidy = mb.newLocal[Int]("ploidy")
    val phase = mb.newLocal[Int]("phase")
    val nBitsPerProb = mb.newLocal[Int]("nBitsPerProb")
    val nExpectedBytesProbs = mb.newLocal[Int]("nExpectedBytesProbs")
    val c0 = mb.genFieldThisRef[Int]("c0")
    val c1 = mb.genFieldThisRef[Int]("c1")
    val c2 = mb.genFieldThisRef[Int]("c2")
    val off = mb.newLocal[Int]("off")
    val d0 = mb.newLocal[Int]("d0")
    val d1 = mb.newLocal[Int]("d1")
    val d2 = mb.newLocal[Int]("d2")
    val c = Code(Code(FastIndexedSeq(
      offset := cbfis.invoke[Long]("getPosition"),
      fileIdx := cp.invoke[Int]("index"),
      if (settings.hasField("varid")) {
        varid := cbfis.invoke[Int, String]("readLengthAndString", 2)
      } else {
        cbfis.invoke[Int, Unit]("readLengthAndSkipString", 2)
      },
      if (settings.hasField("rsid")) {
        rsid := cbfis.invoke[Int, String]("readLengthAndString", 2)
      } else {
        cbfis.invoke[Int, Unit]("readLengthAndSkipString", 2)
      },
      contig := cbfis.invoke[Int, String]("readLengthAndString", 2),
      contigRecoded := cp.invoke[String, String]("recodeContig", contig),
      position := cbfis.invoke[Int]("readInt"),
      cp.invoke[Boolean]("skipInvalidLoci").mux(
        Code(
          invalidLocus :=
            (if (settings.rgBc.nonEmpty) {
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
          )),
        if (settings.rgBc.nonEmpty) {
          // verify the locus is valid before continuing
          csettings.invoke[Option[ReferenceGenome]]("rg")
            .invoke[ReferenceGenome]("get")
            .invoke[String, Int, Unit]("checkLocus", contigRecoded, position)
        } else {
          Code._empty // if locus is valid continue
        }
      ),
      regionField := region,
      srvb.start(),
      if (settings.hasField("locus"))
        Code(
        srvb.addBaseStruct(settings.rowPType.field("locus").typ.fundamentalType.asInstanceOf[PStruct], { srvb =>
          Code(
            srvb.start(),
            srvb.addString(contigRecoded),
            srvb.advance(),
            srvb.addInt(position))
        }),
        srvb.advance())
      else Code._empty,
      nAlleles := cbfis.invoke[Int]("readShort"),
      nAlleles.cne(2).mux(
        Code._fatal[Unit](
          const("Only biallelic variants supported, found variant with ")
            .concat(nAlleles.toS)),
        Code._empty),
      if (settings.hasField("alleles"))
        Code(srvb.addArray(settings.rowPType.field("alleles").typ.fundamentalType.asInstanceOf[PArray], { srvb =>
          Code(
            srvb.start(nAlleles),
            i := 0,
            Code.whileLoop(i < nAlleles,
              srvb.addString(cbfis.invoke[Int, String]("readLengthAndString", 4)),
              srvb.advance(),
              i := i + 1))
        }),
          srvb.advance())
       else Code._empty,
      if (settings.hasField("rsid"))
        Code(srvb.addString(rsid), srvb.advance())
      else Code._empty,
      if (settings.hasField("varid"))
        Code(srvb.addString(varid), srvb.advance())
      else Code._empty,
      if (settings.hasField("offset"))
        Code(srvb.addLong(offset), srvb.advance())
      else Code._empty,
      if (settings.hasField("file_idx"))
        Code(srvb.addInt(fileIdx), srvb.advance())
      else Code._empty,
      dataSize := cbfis.invoke[Int]("readInt"),
      settings.entryType match {
        case None =>
          Code.toUnit(cbfis.invoke[Long, Long]("skipBytes", dataSize.toL))

        case Some(t) =>
          val entriesArrayType = settings.rowPType.field(MatrixType.entriesIdentifier).typ.asInstanceOf[PArray]
          val entryType = entriesArrayType.elementType.asInstanceOf[PStruct]

          val includeGT = t.hasField("GT")
          val includeGP = t.hasField("GP")
          val includeDosage = t.hasField("dosage")

          val alreadyMemoized = mb.genFieldThisRef[Boolean]("alreadyMemoized")
          val memoizedEntryData = mb.genFieldThisRef[Long]("memoizedEntryData")

          val memoTyp = PArray(entryType.setRequired(true), required = true)
          val memoizeAllValues: Code[Unit] = {
            val memoMB = mb.genEmitMethod("memoizeEntries", Array[TypeInfo[_]](), UnitInfo)

            val d0 = memoMB.newLocal[Int]("memoize_entries_d0")
            val d1 = memoMB.newLocal[Int]("memoize_entries_d1")
            val d2 = memoMB.newLocal[Int]("memoize_entries_d2")

            val srvb = new StagedRegionValueBuilder(memoMB, memoTyp, fb.partitionRegion)

            memoMB.emit(
              alreadyMemoized.mux(
                Code._empty,
                Code(
                  srvb.start(1 << 16),
                  d0 := 0,
                  Code.whileLoop(d0 < 256,
                    d1 := 0,
                    Code.whileLoop(d1 < 256,
                      d2 := const(255) - d0 - d1,
                      srvb.addBaseStruct(entryType, { srvb =>
                        val addGT: Code[Unit] = if (includeGT) {

                          val addGtMB = mb.genEmitMethod("bgen_add_gt",
                            Array[TypeInfo[_]](IntInfo, IntInfo, IntInfo),
                            UnitInfo)
                          val d0arg = addGtMB.getArg[Int](1)
                          val d1arg = addGtMB.getArg[Int](2)
                          val d2arg = addGtMB.getArg[Int](3)

                          addGtMB.emit(
                            Code(
                              (d0arg > d1arg).mux(
                                (d0arg > d2arg).mux(
                                  srvb.addInt(c0),
                                  (d2arg > d0arg).mux(
                                    srvb.addInt(c2),
                                    // d0 == d2
                                    srvb.setMissing())),
                                // d0 <= d1
                                (d2arg > d1arg).mux(
                                  srvb.addInt(c2),
                                  // d2 <= d1
                                  (d1arg.ceq(d0arg) || d1arg.ceq(d2arg)).mux(
                                    srvb.setMissing(),
                                    srvb.addInt(c1)))),
                              if (includeGP || includeDosage) srvb.advance() else Code._empty))
                          addGtMB.invoke(d0, d1, d2)
                        } else Code._empty

                        val addGP: Code[Unit] = if (includeGP) {
                          val addGpMB = mb.genEmitMethod("bgen_add_gp",
                            Array[TypeInfo[_]](IntInfo, IntInfo, IntInfo),
                            UnitInfo)

                          val d0arg = addGpMB.getArg[Int](1)
                          val d1arg = addGpMB.getArg[Int](2)
                          val d2arg = addGpMB.getArg[Int](3)

                          val divisor = addGpMB.newLocal[Double]("divisor")

                          addGpMB.emit(Code(
                            srvb.addArray(entryType.field("GP").typ.asInstanceOf[PArray], { srvb =>
                              Code(
                                divisor := 255.0,
                                srvb.start(3),
                                srvb.addDouble(d0arg.toD / divisor),
                                srvb.advance(),
                                srvb.addDouble(d1arg.toD / divisor),
                                srvb.advance(),
                                srvb.addDouble(d2arg.toD / divisor))
                            }),
                            if (includeDosage) srvb.advance() else Code._empty))
                          addGpMB.invoke(d0, d1, d2)
                        } else Code._empty

                        val addDosage: Code[Unit] = if (includeDosage) {
                          val addDosageMB = mb.genEmitMethod("bgen_add_dosage",
                            Array[TypeInfo[_]](IntInfo, IntInfo),
                            UnitInfo)

                          val d1arg = addDosageMB.getArg[Int](1)
                          val d2arg = addDosageMB.getArg[Int](2)

                          addDosageMB.emit(srvb.addDouble((d1arg + (d2arg << 1)).toD / 255.0))
                          addDosageMB.invoke(d1, d2)
                        } else Code._empty

                        Code(srvb.start(), addGT, addGP, addDosage)
                      }),
                      srvb.advance(),
                      d1 := d1 + 1
                    ),
                    d0 := d0 + 1
                ),
                memoizedEntryData := srvb.end(),
                alreadyMemoized := true
              )
            ))
            memoMB.invoke()
          }

          val lookupEntry: (Code[Int], Code[Int]) => Code[Long] = {
            val lookupMB = mb.genEmitMethod("bgen_look_up_add_entry", Array[TypeInfo[_]](IntInfo, IntInfo), LongInfo)

            val d0 = lookupMB.getArg[Int](1)
            val d1 = lookupMB.getArg[Int](2)
            lookupMB.emit(Code(
              Code._empty,
              memoTyp.elementOffset(memoizedEntryData, settings.nSamples, (d0 << 8) | d1)
            ))
            lookupMB.invoke(_, _)
          }

          val addEntries: Code[Array[Byte]] => Code[Unit] = {
            val addEntriesMB = mb.genEmitMethod("bgen_add_entries", Array[TypeInfo[_]](typeInfo[Array[Byte]]), UnitInfo)
            val data = addEntriesMB.getArg[Array[Byte]](1)
            val i = addEntriesMB.newLocal[Int]("i")
            val off = addEntriesMB.newLocal[Int]("off")
            val d0 = addEntriesMB.newLocal[Int]("d0")
            val d1 = addEntriesMB.newLocal[Int]("d1")
            addEntriesMB.emit(
              srvb.addArray(entriesArrayType,
                { srvb =>
                  Code(
                    srvb.start(settings.nSamples),
                    i := 0,
                    Code.whileLoop(i < settings.nSamples,
                      (data(i + 8) & 0x80).cne(0).mux(
                        srvb.setMissing(),
                        Code(
                          off := const(settings.nSamples + 10) + i * 2,
                          d0 := data(off) & 0xff,
                          d1 := data(off + 1) & 0xff,
                          srvb.addIRIntermediate(entryType)(lookupEntry(d0, d1))
                        )),
                      srvb.advance(),
                      i := i + 1))
                })
            )
            addEntriesMB.invoke(_)
          }

          Code(FastIndexedSeq(
            cp.invoke[Boolean]("compressed").mux(
              Code(
                uncompressedSize := cbfis.invoke[Int]("readInt"),
                input := cbfis.invoke[Int, Array[Byte]]("readBytes", dataSize - 4),
                data := Code.invokeScalaObject[Array[Byte], Int, Array[Byte]](
                  BgenRDD.getClass, "decompress", input, uncompressedSize)),
              data := cbfis.invoke[Int, Array[Byte]]("readBytes", dataSize)),
            reader := Code.newInstance[ByteArrayReader, Array[Byte]](data),
            nRow := reader.invoke[Int]("readInt"),
            (nRow.cne(settings.nSamples)).mux(
              Code._fatal[Unit](
                const("Row nSamples is not equal to header nSamples: ")
                  .concat(nRow.toS)
                  .concat(", ")
                  .concat(settings.nSamples.toString)),
              Code._empty),
            nAlleles2 := reader.invoke[Int]("readShort"),
            (nAlleles.cne(nAlleles2)).mux(
              Code._fatal[Unit](
                const("""Value for 'nAlleles' in genotype probability data storage is
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
              Code._fatal[Unit](
                const("Hail only supports diploid genotypes. Found min ploidy '")
                  .concat(minPloidy.toS)
                  .concat("' and max ploidy '")
                  .concat(maxPloidy.toS)
                  .concat("'.")),
              Code._empty),
            i := 0,
            Code.whileLoop(i < settings.nSamples,
              ploidy := reader.invoke[Int]("read"),
              ((ploidy & 0x3f).cne(2)).mux(
                Code._fatal[Unit](
                  const("Ploidy value must equal to 2. Found ")
                    .concat(ploidy.toS)
                    .concat(".")),
                Code._empty),
              i += 1
            ),
            phase := reader.invoke[Int]("read"),
            (phase.cne(0) && phase.cne(1)).mux(
              Code._fatal[Unit](
                const("Phase value must be 0 or 1. Found ")
                  .concat(phase.toS)
                  .concat(".")),
              Code._empty),
            phase.ceq(1).mux(
              Code._fatal[Unit]("Hail does not support phased genotypes."),
              Code._empty),
            nBitsPerProb := reader.invoke[Int]("read"),
            (nBitsPerProb < 1 || nBitsPerProb > 32).mux(
              Code._fatal[Unit](
                const("nBits value must be between 1 and 32 inclusive. Found ")
                  .concat(nBitsPerProb.toS)
                  .concat(".")),
              Code._empty),
            (nBitsPerProb.cne(8)).mux(
              Code._fatal[Unit](
                const("Hail only supports 8-bit probabilities, found ")
                  .concat(nBitsPerProb.toS)
                  .concat(".")),
              Code._empty),
            nExpectedBytesProbs := settings.nSamples * 2,
            (reader.invoke[Int]("length").cne(nExpectedBytesProbs + settings.nSamples + 10)).mux(
              Code._fatal[Unit](
                const("Number of uncompressed bytes '")
                  .concat(reader.invoke[Int]("length").toS)
                  .concat("' does not match the expected size '")
                  .concat(nExpectedBytesProbs.toS)
                  .concat("'.")),
              Code._empty),
            c0 := Call2.fromUnphasedDiploidGtIndex(0),
            c1 := Call2.fromUnphasedDiploidGtIndex(1),
            c2 := Call2.fromUnphasedDiploidGtIndex(2),
            memoizeAllValues,
            addEntries(data)
            ))
      })),
      srvb.end())

    mb.emit(c)
    fb.resultWithIndex()
  }
}
