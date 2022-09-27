package is.hail.io.bgen

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.{BroadcastValue, ExecuteContext}
import is.hail.expr.ir.{EmitCode, EmitFunctionBuilder, IEmitCode, ParamType, TableReader}
import is.hail.io.fs.FS
import is.hail.io.index.IndexReaderBuilder
import is.hail.io.{ByteArrayReader, HadoopFSDataBinaryReader}
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.{SCanonicalCallValue, SStackStruct, SStringPointer}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual.{TInterval, Type}
import is.hail.utils._
import is.hail.variant.{Call2, ReferenceGenome}
import org.apache.spark.Partition

trait BgenPartition extends Partition {
  def path: String

  def compressed: Boolean

  def skipInvalidLoci: Boolean

  def contigRecoding: Map[String, String]

  def fsBc: BroadcastValue[FS]

  def makeInputStream: HadoopFSDataBinaryReader = {
    new HadoopFSDataBinaryReader(fsBc.value.openNoCompression(path))
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
  fsBc: BroadcastValue[FS]
) extends BgenPartition {
  assert(startIndex <= endIndex)

  def index = partitionIndex
}

object BgenRDDPartitions extends Logging {
  def checkFilesDisjoint(fs: FS, fileMetadata: Seq[BgenFileMetadata], keyType: Type): Array[Interval] = {
    assert(fileMetadata.nonEmpty)
    val pord = keyType.ordering
    val bounds = fileMetadata.map(md => (md.path, md.rangeBounds))

    val overlappingBounds = new BoxedArrayBuilder[(String, Interval, String, Interval)]
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
           |  ${
          overlappingBounds.result().map { case (f1, i1, f2, i2) =>
            s"file1: $f1\trangeBounds1: $i1\tfile2: $f2\trangeBounds2: $i2"
          }.mkString("\n  ")
        })""".stripMargin)

    bounds.map(_._2).toArray
  }

  def apply(
    ctx: ExecuteContext,
    rg: Option[ReferenceGenome],
    files: Seq[BgenFileMetadata],
    blockSizeInMB: Option[Int],
    nPartitions: Option[Int],
    keyType: Type
  ): (Array[Partition], Array[Interval]) = {
    val fs = ctx.fs
    val fsBc = fs.broadcast

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
      val (leafPType: PStruct, leafDec) = leafCodec.buildDecoder(ctx, leafCodec.encodedVirtualType)
      val (intPType: PStruct, intDec) = internalNodeCodec.buildDecoder(ctx, internalNodeCodec.encodedVirtualType)
      IndexReaderBuilder.withDecoders(leafDec, intDec, BgenSettings.indexKeyType(rg), BgenSettings.indexAnnotationType, leafPType, intPType)
    }
    if (nonEmptyFilesAfterFilter.isEmpty) {
      (Array.empty, Array.empty)
    } else {
      val partitions = new BoxedArrayBuilder[Partition]()
      val rangeBounds = new BoxedArrayBuilder[Interval]()
      var fileIndex = 0
      while (fileIndex < nonEmptyFilesAfterFilter.length) {
        val file = nonEmptyFilesAfterFilter(fileIndex)
        // TODO Not sure I should be using ctx's pool here.
        using(indexReaderBuilder(ctx.theHailClassLoader, fs, file.indexPath, 8, ctx.r.pool)) { index =>
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
              fsBc
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
    ctx: ExecuteContext,
    settings: BgenSettings
  ): (HailClassLoader, FS, Int, Region) => AsmFunction4[Region, BgenPartition, HadoopFSDataBinaryReader, BgenSettings, Long] = {
    val fb = EmitFunctionBuilder[Region, BgenPartition, HadoopFSDataBinaryReader, BgenSettings, Long](ctx, "bgen_rdd_decoder")
    val mb = fb.apply_method
    val rowType = settings.rowPType
    mb.emitWithBuilder[Long] { cb =>

      val region = mb.getCodeParam[Region](1)
      val cp = mb.getCodeParam[BgenPartition](2)
      val cbfis = mb.getCodeParam[HadoopFSDataBinaryReader](3)
      val csettings = mb.getCodeParam[BgenSettings](4)

      val offset = cb.newLocal[Long]("offset")
      val fileIdx = cb.newLocal[Int]("fileIdx")
      val varid = cb.newLocal[String]("varid")
      val rsid = cb.newLocal[String]("rsid")
      val contig = cb.newLocal[String]("contig")
      val contigRecoded = cb.newLocal[String]("contigRecoded")
      val position = cb.newLocal[Int]("position")
      val nAlleles = cb.newLocal[Int]("nAlleles")
      val i = cb.newLocal[Int]("i")
      val dataSize = cb.newLocal[Int]("dataSize")
      val invalidLocus = cb.newLocal[Boolean]("invalidLocus")
      val data = cb.newLocal[Array[Byte]]("data")
      val uncompressedSize = cb.newLocal[Int]("uncompressedSize")
      val input = cb.newLocal[Array[Byte]]("input")
      val reader = cb.newLocal[ByteArrayReader]("reader")
      val nRow = cb.newLocal[Int]("nRow")
      val nAlleles2 = cb.newLocal[Int]("nAlleles2")
      val minPloidy = cb.newLocal[Int]("minPloidy")
      val maxPloidy = cb.newLocal[Int]("maxPloidy")
      val longPloidy = cb.newLocal[Long]("longPloidy")
      val ploidy = cb.newLocal[Int]("ploidy")
      val phase = cb.newLocal[Int]("phase")
      val nBitsPerProb = cb.newLocal[Int]("nBitsPerProb")
      val nExpectedBytesProbs = cb.newLocal[Int]("nExpectedBytesProbs")
      val c0 = mb.genFieldThisRef[Int]("c0")
      val c1 = mb.genFieldThisRef[Int]("c1")
      val c2 = mb.genFieldThisRef[Int]("c2")

      cb.assign(c0, Call2.fromUnphasedDiploidGtIndex(0))
      cb.assign(c1, Call2.fromUnphasedDiploidGtIndex(1))
      cb.assign(c2, Call2.fromUnphasedDiploidGtIndex(2))


      cb.assign(offset, cbfis.invoke[Long]("getPosition"))
      cb.assign(fileIdx, cp.invoke[Int]("index"))

      if (settings.hasField("varid"))
        cb.assign(varid, cbfis.invoke[Int, String]("readLengthAndString", 2))
      else
        cb += cbfis.invoke[Int, Unit]("readLengthAndSkipString", 2)

      if (settings.hasField("rsid"))
        cb.assign(rsid, cbfis.invoke[Int, String]("readLengthAndString", 2))
      else
        cb += cbfis.invoke[Int, Unit]("readLengthAndSkipString", 2)

      cb.assign(contig, cbfis.invoke[Int, String]("readLengthAndString", 2))
      cb.assign(contigRecoded, cp.invoke[String, String]("recodeContig", contig))
      cb.assign(position, cbfis.invoke[Int]("readInt"))


      cb.ifx(cp.invoke[Boolean]("skipInvalidLoci"),
        {
          if (settings.rgBc.nonEmpty) {
            cb.assign(invalidLocus, !csettings.invoke[Option[ReferenceGenome]]("rg")
              .invoke[ReferenceGenome]("get")
              .invoke[String, Int, Boolean]("isValidLocus", contigRecoded, position))
          }
          cb.ifx(invalidLocus,
            {
              cb.assign(nAlleles, cbfis.invoke[Int]("readShort"))
              cb.assign(i, 0)
              cb.whileLoop(i < nAlleles,
                {
                  cb += cbfis.invoke[Int, Unit]("readLengthAndSkipString", 4)
                  cb.assign(i, i + 1)
                })
              cb.assign(dataSize, cbfis.invoke[Int]("readInt"))
              cb += Code.toUnit(cbfis.invoke[Long, Long]("skipBytes", dataSize.toL))
              cb += Code._return(-1L)
            }
          )
        },
        {
          if (settings.rgBc.nonEmpty)
            cb += csettings.invoke[Option[ReferenceGenome]]("rg")
              .invoke[ReferenceGenome]("get")
              .invoke[String, Int, Unit]("checkLocus", contigRecoded, position)
        })

      val structFieldCodes = new BoxedArrayBuilder[EmitCode]()

      if (settings.hasField("locus")) {
        // double-allocates the locus struct, but this is a very minor performance regression
        // and will be removed soon
        val pc = settings.rowPType.field("locus").typ match {
          case t: PCanonicalLocus =>
            t.constructFromPositionAndString(cb, region, contigRecoded, position)
          case t: PCanonicalStruct =>
            val strT = t.field("contig").typ.asInstanceOf[PCanonicalString]
            val contigPC = strT.sType.constructFromString(cb, region, contigRecoded)
            t.constructFromFields(cb, region,
              FastIndexedSeq(EmitCode.present(cb.emb, contigPC), EmitCode.present(cb.emb, primitive(position))),
              deepCopy = false)
        }
        structFieldCodes += EmitCode.present(cb.emb, pc)
      }

      cb.assign(nAlleles, cbfis.invoke[Int]("readShort"))

      cb.ifx(nAlleles.cne(2),
        cb._fatal("Only biallelic variants supported, found variant with ", nAlleles.toS, " alleles: ",
          contigRecoded, ":", position.toS))

      if (settings.hasField("alleles")) {
        val allelesType = settings.rowPType.field("alleles").typ.asInstanceOf[PCanonicalArray]
        val alleleStringType = allelesType.elementType.asInstanceOf[PCanonicalString]
        val (pushElement, finish) = allelesType.constructFromFunctions(cb, region, nAlleles, deepCopy = false)

        cb.whileLoop(i < nAlleles, {
          val st = SStringPointer(alleleStringType)
          val strCode = st.constructFromString(cb, region, cbfis.invoke[Int, String]("readLengthAndString", 4))
          pushElement(cb, IEmitCode.present(cb, strCode))
          cb.assign(i, i + 1)
        })

        val allelesArr = finish(cb)
        structFieldCodes += EmitCode.present(cb.emb, allelesArr)
      }

      if (settings.hasField("rsid"))
        structFieldCodes += EmitCode.present(cb.emb, SStringPointer(PCanonicalString(false)).constructFromString(cb, region, rsid))
      if (settings.hasField("varid"))
        structFieldCodes += EmitCode.present(cb.emb, SStringPointer(PCanonicalString(false)).constructFromString(cb, region, varid))
      if (settings.hasField("offset"))
        structFieldCodes += EmitCode.present(cb.emb, primitive(offset))
      if (settings.hasField("file_idx"))
        structFieldCodes += EmitCode.present(cb.emb, primitive(fileIdx))

      cb.assign(dataSize, cbfis.invoke[Int]("readInt"))
      settings.entryType match {
        case None =>
          cb += Code.toUnit(cbfis.invoke[Long, Long]("skipBytes", dataSize.toL))
        case Some(t) =>
          val entriesArrayType = settings.rowPType.field(MatrixType.entriesIdentifier).typ.asInstanceOf[PCanonicalArray]
          val entryType = entriesArrayType.elementType.asInstanceOf[PCanonicalStruct]

          val includeGT = t.hasField("GT")
          val includeGP = t.hasField("GP")
          val includeDosage = t.hasField("dosage")

          val alreadyMemoized = mb.genFieldThisRef[Boolean]("alreadyMemoized")
          val memoizedEntryData = mb.genFieldThisRef[Long]("memoizedEntryData")

          val memoTyp = PCanonicalArray(entryType.setRequired(true), required = true)

          val memoMB = mb.genEmitMethod("memoizeEntries", FastIndexedSeq[ParamType](), UnitInfo)
          memoMB.voidWithBuilder { cb =>

            val partRegion = mb.partitionRegion

            val LnoOp = CodeLabel()
            cb.ifx(alreadyMemoized, cb.goto(LnoOp))

            val (push, finish) = memoTyp.constructFromFunctions(cb, partRegion, 1 << 16, false)

            val d0 = cb.newLocal[Int]("memoize_entries_d0", 0)
            cb.whileLoop(d0 < 256, {
              val d1 = cb.newLocal[Int]("memoize_entries_d1", 0)
              cb.whileLoop(d1 < 256, {
                val d2 = cb.newLocal[Int]("memoize_entries_d2", const(255) - d0 - d1)

                val entryFieldCodes = new BoxedArrayBuilder[EmitCode]()

                if (includeGT)
                  entryFieldCodes += EmitCode.fromI(cb.emb) { cb =>
                    val Lmissing = CodeLabel()
                    val Lpresent = CodeLabel()
                    val value = cb.newLocal[Int]("bgen_gt_value")

                    cb.ifx(d0 > d1,
                      cb.ifx(d0 > d2,
                        {
                          cb.assign(value, c0)
                          cb.goto(Lpresent)
                        },
                        cb.ifx(d2 > d0,
                          {
                            cb.assign(value, c2)
                            cb.goto(Lpresent)
                          },
                          // d0 == d2
                          cb.goto(Lmissing))),
                      // d0 <= d1
                      cb.ifx(d2 > d1,
                        {
                          cb.assign(value, c2)
                          cb.goto(Lpresent)
                        },
                        // d2 <= d1
                        cb.ifx(d1.ceq(d0) || d1.ceq(d2),
                          cb.goto(Lmissing),
                          {
                            cb.assign(value, c1)
                            cb.goto(Lpresent)
                          })))

                    IEmitCode(Lmissing, Lpresent, new SCanonicalCallValue(value), false)
                  }

                if (includeGP)
                  entryFieldCodes += EmitCode.fromI(cb.emb) { cb =>

                    val divisor = cb.newLocal[Double]("divisor", 255.0)

                    val gpType = entryType.field("GP").typ.asInstanceOf[PCanonicalArray]

                    val (pushElement, finish) = gpType.constructFromFunctions(cb, partRegion, 3, deepCopy = false)
                    pushElement(cb, IEmitCode.present(cb, primitive(cb.memoize(d0.toD / divisor))))
                    pushElement(cb, IEmitCode.present(cb, primitive(cb.memoize(d1.toD / divisor))))
                    pushElement(cb, IEmitCode.present(cb, primitive(cb.memoize(d2.toD / divisor))))

                    IEmitCode.present(cb, finish(cb))
                  }


                if (includeDosage)
                  entryFieldCodes += EmitCode.fromI(cb.emb) { cb =>
                    IEmitCode.present(cb, primitive(cb.memoize((d1 + (d2 << 1)).toD / 255.0)))
                  }

                push(cb, IEmitCode.present(cb,
                  SStackStruct.constructFromArgs(cb, partRegion, entryType.virtualType, entryFieldCodes.result(): _*)))

                cb.assign(d1, d1 + 1)
              })

              cb.assign(d0, d0 + 1)
            })

            cb.assign(memoizedEntryData, finish(cb).a)
            cb.assign(alreadyMemoized, true)

            cb.define(LnoOp)
          }

          cb.ifx(cp.invoke[Boolean]("compressed"), {
            cb.assign(uncompressedSize, cbfis.invoke[Int]("readInt"))
            cb.assign(input, cbfis.invoke[Int, Array[Byte]]("readBytes", dataSize - 4))
            cb.assign(data, Code.invokeScalaObject2[Array[Byte], Int, Array[Byte]](
              BgenRDD.getClass, "decompress", input, uncompressedSize))
          }, {
            cb.assign(data, cbfis.invoke[Int, Array[Byte]]("readBytes", dataSize))
          })

          cb.assign(reader, Code.newInstance[ByteArrayReader, Array[Byte]](data))
          cb.assign(nRow, reader.invoke[Int]("readInt"))
          cb.ifx(nRow.cne(settings.nSamples), cb._fatal(
            const("Row nSamples is not equal to header nSamples: ")
              .concat(nRow.toS)
              .concat(", ")
              .concat(settings.nSamples.toString)
          ))

          cb.assign(nAlleles2, reader.invoke[Int]("readShort"))
          cb.ifx(nAlleles.cne(nAlleles2),
            cb._fatal(const(
              """Value for 'nAlleles' in genotype probability data storage is
                |not equal to value in variant identifying data. Expected""".stripMargin)
              .concat(nAlleles.toS)
              .concat(" but found ")
              .concat(nAlleles2.toS)
              .concat(" at ")
              .concat(contig)
              .concat(":")
              .concat(position.toS)
              .concat(".")))

          cb.assign(minPloidy, reader.invoke[Int]("read"))
          cb.assign(maxPloidy, reader.invoke[Int]("read"))

          cb.ifx(minPloidy.cne(2) || maxPloidy.cne(2),
            cb._fatal(const("Hail only supports diploid genotypes. Found min ploidy '")
              .concat(minPloidy.toS)
              .concat("' and max ploidy '")
              .concat(maxPloidy.toS)
              .concat("'.")))

          cb.assign(i, 0)
          cb.whileLoop(i < settings.nSamples, {
            cb.assign(ploidy, reader.invoke[Int]("read"))
            cb.ifx((ploidy & 0x3f).cne(2),
              cb._fatal(const("Ploidy value must equal to 2. Found ")
                .concat(ploidy.toS)
                .concat(".")))
            cb.assign(i, i + 1)
          })

          cb.assign(phase, reader.invoke[Int]("read"))
          cb.ifx(phase.cne(0) && (phase.cne(1)),
            cb._fatal(const("Phase value must be 0 or 1. Found ")
              .concat(phase.toS)
              .concat(".")))

          cb.ifx(phase.ceq(1), cb._fatal("Hail does not support phased genotypes in 'import_bgen'."))

          cb.assign(nBitsPerProb, reader.invoke[Int]("read"))
          cb.ifx(nBitsPerProb < 1 || nBitsPerProb > 32,
            cb._fatal(const("nBits value must be between 1 and 32 inclusive. Found ")
              .concat(nBitsPerProb.toS)
              .concat(".")))
          cb.ifx(nBitsPerProb.cne(8),
            cb._fatal(const("Hail only supports 8-bit probabilities, found ")
              .concat(nBitsPerProb.toS)
              .concat(".")))

          cb.assign(nExpectedBytesProbs, settings.nSamples * 2)
          cb.ifx(reader.invoke[Int]("length").cne(nExpectedBytesProbs + settings.nSamples + 10),
            cb._fatal(const("Number of uncompressed bytes '")
              .concat(reader.invoke[Int]("length").toS)
              .concat("' does not match the expected size '")
              .concat(nExpectedBytesProbs.toS)
              .concat("'.")))

          cb.invokeVoid(memoMB)

          val (pushElement, finish) = entriesArrayType.constructFromFunctions(cb, region, settings.nSamples, deepCopy = false)

          cb.assign(i, 0)
          cb.whileLoop(i < settings.nSamples, {

            val Lmissing = CodeLabel()
            val Lpresent = CodeLabel()

            cb.ifx((data(i + 8) & 0x80).cne(0), cb.goto(Lmissing))
            val dataOffset = cb.newLocal[Int]("bgen_add_entries_offset", const(settings.nSamples + 10) + i * 2)
            val d0 = data(dataOffset) & 0xff
            val d1 = data(dataOffset + 1) & 0xff
            val pc = entryType.loadCheapSCode(cb, memoTyp.loadElement(memoizedEntryData, settings.nSamples, (d0 << 8) | d1))
            cb.goto(Lpresent)
            val iec = IEmitCode(Lmissing, Lpresent, pc, false)
            pushElement(cb, iec)

            cb.assign(i, i + 1)
          })

          val pc = finish(cb)

          structFieldCodes += EmitCode.fromI(cb.emb)(cb => IEmitCode.present(cb, pc))
      }

      if (settings.hasField(TableReader.uidFieldName))
        structFieldCodes += EmitCode.present(cb.emb, primitive(offset))

      rowType.constructFromFields(cb, region, structFieldCodes.result(), deepCopy = false).a
    }

    fb.resultWithIndex()
  }
}

