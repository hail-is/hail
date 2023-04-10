package is.hail.io.bgen

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.{BroadcastValue, ExecuteContext}
import is.hail.expr.ir.functions.{RegistryFunctions, StringFunctions}
import is.hail.expr.ir.streams.StreamUtils
import is.hail.expr.ir.{ArraySorter, EmitCode, EmitCodeBuilder, EmitFunctionBuilder, EmitSettable, IEmitCode, LowerMatrixIR, ParamType, StagedArrayBuilder, uuid4}
import is.hail.io._
import is.hail.io.fs.SeekableDataInputStream
import is.hail.io.index.{StagedIndexReader, StagedIndexWriter}
import is.hail.lir
import is.hail.types.physical._
import is.hail.types.physical.stypes.SingleCodeType
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces.{NoBoxLongIterator, SBaseStructValue, primitive}
import is.hail.types.physical.stypes.primitives.SInt64
import is.hail.types.virtual._
import is.hail.types.{RStruct, TableType, TypeWithRequiredness}
import is.hail.utils.{BoxedArrayBuilder, CompressionUtils, FastIndexedSeq}
import is.hail.variant.{Call2, ReferenceGenome}
import org.objectweb.asm.Opcodes._

object StagedBGENReader {
  def decompress(
    input: Array[Byte],
    uncompressedSize: Int
  ): Array[Byte] = is.hail.utils.decompress(input, uncompressedSize)


  def recodeContig(contig: String, contigMap: Map[String, String]): String = contigMap.getOrElse(contig, contig)

  def rowRequiredness(requested: TStruct): RStruct = {
    val t = TypeWithRequiredness(requested).asInstanceOf[RStruct]
    t.fieldOption(LowerMatrixIR.entriesFieldName)
      .foreach { t => t.fromPType(entryArrayPType(requested.field(LowerMatrixIR.entriesFieldName).typ))
      }
    t
  }

  def entryArrayPType(requested: Type) = {

    val entryType = requested.asInstanceOf[TArray].elementType.asInstanceOf[TStruct]

    PCanonicalArray(PCanonicalStruct(false,
      Array(
        "GT" -> PCanonicalCall(),
        "GP" -> PCanonicalArray(PFloat64Required, required = true),
        "dosage" -> PFloat64Required
      ).filter { case (name, _) => entryType.hasField(name)
      }: _*), true)
  }

  def decodeRow(cb: EmitCodeBuilder,
    region: Value[Region],
    cbfis: Value[HadoopFSDataBinaryReader],
    nSamples: Value[Int],
    fileIdx: Value[Int],
    compression: Value[Int],
    skipInvalidLoci: Value[Boolean],
    contigRecoding: Value[Map[String, String]],
    requestedType: TStruct,
    rg: Option[String]
  ): EmitCode = {
    var out: EmitSettable = null // defined and assigned inside method
    val emb = cb.emb.ecb.genEmitMethod("decode_bgen_row", IndexedSeq[ParamType](classInfo[Region], classInfo[HadoopFSDataBinaryReader], IntInfo, IntInfo, IntInfo, BooleanInfo, classInfo[Map[String, String]]), UnitInfo)
    emb.voidWithBuilder { cb =>

      val rgBc = rg.map { rg => cb.memoize(emb.getReferenceGenome(rg)) }
      val region = emb.getCodeParam[Region](1)
      val cbfis = emb.getCodeParam[HadoopFSDataBinaryReader](2)
      val nSamples = emb.getCodeParam[Int](3)
      val fileIdx = emb.getCodeParam[Int](4)
      val compression = emb.getCodeParam[Int](5)
      val skipInvalidLoci = emb.getCodeParam[Boolean](6)
      val contigRecoding = emb.getCodeParam[Map[String, String]](7)

      val LreturnMissing = CodeLabel()

      val offset = cb.newLocal[Long]("offset")
      val varid = cb.newLocal[String]("varid")
      val rsid = cb.newLocal[String]("rsid")
      val contig = cb.newLocal[String]("contig")
      val contigRecoded = cb.newLocal[String]("contigRecoded")
      val position = cb.newLocal[Int]("position")
      val nAlleles = cb.newLocal[Int]("nAlleles")
      val i = cb.newLocal[Int]("i")
      val dataSize = cb.newLocal[Int]("dataSize")
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
      val c0 = emb.genFieldThisRef[Int]("c0")
      val c1 = emb.genFieldThisRef[Int]("c1")
      val c2 = emb.genFieldThisRef[Int]("c2")

      cb.assign(c0, Call2.fromUnphasedDiploidGtIndex(0))
      cb.assign(c1, Call2.fromUnphasedDiploidGtIndex(1))
      cb.assign(c2, Call2.fromUnphasedDiploidGtIndex(2))


      cb.assign(offset, cbfis.invoke[Long]("getPosition"))

      if (requestedType.hasField("varid"))
        cb.assign(varid, cbfis.invoke[Int, String]("readLengthAndString", 2))
      else
        cb += cbfis.invoke[Int, Unit]("readLengthAndSkipString", 2)

      if (requestedType.hasField("rsid"))
        cb.assign(rsid, cbfis.invoke[Int, String]("readLengthAndString", 2))
      else
        cb += cbfis.invoke[Int, Unit]("readLengthAndSkipString", 2)

      cb.assign(contig, cbfis.invoke[Int, String]("readLengthAndString", 2))
      cb.assign(contigRecoded, Code.invokeScalaObject2[String, Map[String, String], String](StagedBGENReader.getClass, "recodeContig", contig, contigRecoding))
      cb.assign(position, cbfis.invoke[Int]("readInt"))


      cb.ifx(skipInvalidLoci, {
        rgBc.foreach { rg =>
          cb.ifx(!rg.invoke[String, Int, Boolean]("isValidLocus", contigRecoded, position),
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
              cb.goto(LreturnMissing)
            })
        }
      }, {
        rgBc.foreach { rg =>
          cb += rg.invoke[String, Int, Unit]("checkLocus", contigRecoded, position)
        }
      })

      val structFieldCodes = new BoxedArrayBuilder[EmitCode]()

      if (requestedType.hasField("locus")) {
        // double-allocates the locus struct, but this is a very minor performance regression
        // and will be removed soon
        val pc = requestedType.field("locus").typ match {
          case TLocus(rg) =>
            val pt = SCanonicalLocusPointer(PCanonicalLocus(rg))
            pt.pType.constructFromPositionAndString(cb, region, contigRecoded, position)
          case t: TStruct =>
            val contig = SJavaString.constructFromString(cb, region, contigRecoded)
            SStackStruct.constructFromArgs(cb, region, t,
              EmitCode.present(cb.emb, contig), EmitCode.present(cb.emb, primitive(position)))
        }
        structFieldCodes += EmitCode.present(cb.emb, pc)
      }

      cb.assign(nAlleles, cbfis.invoke[Int]("readShort"))

      cb.ifx(nAlleles.cne(2),
        cb._fatal("Only biallelic variants supported, found variant with ", nAlleles.toS, " alleles: ",
          contigRecoded, ":", position.toS))

      if (requestedType.hasField("alleles")) {
        val allelesType = SJavaArrayString(true)

        val a = cb.newLocal[Array[String]]("alleles", Code.newArray[String](nAlleles))
        cb.whileLoop(i < nAlleles, {
          cb += a.update(i, cbfis.invoke[Int, String]("readLengthAndString", 4))
          cb.assign(i, i + 1)
        })


        structFieldCodes += EmitCode.present(cb.emb, allelesType.construct(cb, a))
      }

      if (requestedType.hasField("rsid"))
        structFieldCodes += EmitCode.present(cb.emb, SStringPointer(PCanonicalString(false)).constructFromString(cb, region, rsid))
      if (requestedType.hasField("varid"))
        structFieldCodes += EmitCode.present(cb.emb, SStringPointer(PCanonicalString(false)).constructFromString(cb, region, varid))
      if (requestedType.hasField("offset"))
        structFieldCodes += EmitCode.present(cb.emb, primitive(offset))
      if (requestedType.hasField("file_idx"))
        structFieldCodes += EmitCode.present(cb.emb, primitive(fileIdx))

      cb.assign(dataSize, cbfis.invoke[Int]("readInt"))
      requestedType.fieldOption(LowerMatrixIR.entriesFieldName) match {
        case None =>
          cb += Code.toUnit(cbfis.invoke[Long, Long]("skipBytes", dataSize.toL))
        case Some(t) =>
          val entriesArrayType = entryArrayPType(t.typ)
          val entryType = entriesArrayType.elementType.asInstanceOf[PCanonicalStruct]
          val entryVType = entryType.virtualType

          val includeGT = entryVType.hasField("GT")
          val includeGP = entryVType.hasField("GP")
          val includeDosage = entryVType.hasField("dosage")

          val alreadyMemoized = emb.genFieldThisRef[Boolean]("alreadyMemoized")
          val memoizedEntryData = emb.genFieldThisRef[Long]("memoizedEntryData")

          val memoTyp = PCanonicalArray(entryType.setRequired(true), required = true)

          val memoMB = emb.genEmitMethod("memoizeEntries", FastIndexedSeq[ParamType](), UnitInfo)
          memoMB.voidWithBuilder { cb =>

            val partRegion = emb.partitionRegion

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

          cb.ifx(compression ceq BgenSettings.UNCOMPRESSED, {
            cb.assign(data, cbfis.invoke[Int, Array[Byte]]("readBytes", dataSize))
          }, {
            cb.assign(uncompressedSize, cbfis.invoke[Int]("readInt"))
            cb.assign(input, cbfis.invoke[Int, Array[Byte]]("readBytes", dataSize - 4))
            cb.ifx(compression ceq BgenSettings.ZLIB_COMPRESSION, {
              cb.assign(data,
                Code.invokeScalaObject2[Array[Byte], Int, Array[Byte]](
                  CompressionUtils.getClass, "decompressZlib", input, uncompressedSize))
            }, {
              // zstd
              cb.assign(data, Code.invokeScalaObject2[Array[Byte], Int, Array[Byte]](
                CompressionUtils.getClass, "decompressZstd", input, uncompressedSize))
            })
          })

          cb.assign(reader, Code.newInstance[ByteArrayReader, Array[Byte]](data))
          cb.assign(nRow, reader.invoke[Int]("readInt"))
          cb.ifx(nRow.cne(nSamples), cb._fatal(
            const("Row nSamples is not equal to header nSamples: ")
              .concat(nRow.toS)
              .concat(", ")
              .concat(nSamples.toString)
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
          cb.whileLoop(i < nSamples, {
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

          cb.assign(nExpectedBytesProbs, nSamples * 2)
          cb.ifx(reader.invoke[Int]("length").cne(nExpectedBytesProbs + nSamples.get + 10),
            cb._fatal(const("Number of uncompressed bytes '")
              .concat(reader.invoke[Int]("length").toS)
              .concat("' does not match the expected size '")
              .concat(nExpectedBytesProbs.toS)
              .concat("'.")))

          cb.invokeVoid(memoMB)

          val (pushElement, finish) = entriesArrayType.constructFromFunctions(cb, region, nSamples, deepCopy = false)

          cb.assign(i, 0)
          cb.whileLoop(i < nSamples, {

            val Lmissing = CodeLabel()
            val Lpresent = CodeLabel()

            cb.ifx((data(i + 8) & 0x80).cne(0), cb.goto(Lmissing))
            val dataOffset = cb.newLocal[Int]("bgen_add_entries_offset", (nSamples.get + const(10).get) + i * 2)
            val d0 = data(dataOffset) & 0xff
            val d1 = data(dataOffset + 1) & 0xff
            val pc = entryType.loadCheapSCode(cb, memoTyp.loadElement(memoizedEntryData, nSamples, (d0 << 8) | d1))
            cb.goto(Lpresent)
            val iec = IEmitCode(Lmissing, Lpresent, pc, false)
            pushElement(cb, iec)

            cb.assign(i, i + 1)
          })

          val pc = finish(cb)

          structFieldCodes += EmitCode.fromI(cb.emb)(cb => IEmitCode.present(cb, pc))
      }

      val ss = SStackStruct.constructFromArgs(cb, region, requestedType, structFieldCodes.result(): _*)

      out = emb.ecb.newEmitField("bgen_row", ss.st, false)
      cb.assign(out, EmitCode.present(emb, ss))
      val Lfinish = CodeLabel()
      cb.goto(Lfinish)

      cb.define(LreturnMissing)
      cb.assign(out, EmitCode.missing(emb, ss.st))

      cb.define(Lfinish)
    }
    cb.invokeVoid(emb, region, cbfis, nSamples, fileIdx, compression, skipInvalidLoci, contigRecoding)
    out
  }

  def queryIndexByPosition(ctx: ExecuteContext, leafSpec: AbstractTypedCodecSpec, internalSpec: AbstractTypedCodecSpec): (String, Array[Long]) => Array[AnyRef] = {
    val fb = EmitFunctionBuilder[String, Array[Long], Array[AnyRef]](ctx, "bgen_query_index")

    fb.emitWithBuilder { cb =>
      val mb = fb.apply_method
      val path = mb.getCodeParam[String](1)
      val indices = mb.getCodeParam[Array[Long]](2)
      val index = new StagedIndexReader(mb, leafSpec, internalSpec)
      index.initialize(cb, path)

      val len = cb.memoize(indices.length())
      val boxed = cb.memoize(Code.newArray[AnyRef](len))
      val i = cb.newLocal[Int]("i", 0)
      cb.whileLoop(i < len, {

        val r = index.queryIndex(cb, mb.partitionRegion, cb.memoize(indices(i))).loadField(cb, "key").get(cb)
        cb += boxed.update(i, StringFunctions.svalueToJavaValue(cb, mb.partitionRegion, r, safe = true))
        cb.assign(i, i + 1)
      })
      index.close(cb)
      boxed
    }

    val res = fb.resultWithIndex();
    { (path: String, indices: Array[Long]) =>
      ctx.r.pool.scopedRegion { r =>
        res.apply(ctx.theHailClassLoader, ctx.fs, ctx.taskContext, r)
          .apply(path, indices)
      }
    }
  }

}

object BGENFunctions extends RegistryFunctions {

  def uuid(): String = uuid4()

  override def registerAll(): Unit = {
    registerSCode("index_bgen", Array(TString, TString, TDict(TString, TString), TBoolean, TInt32), TInt64, (_, _) => SInt64, Array(TVariable("locusType"))) {
      case (er, cb, Seq(locType), _, Array(_path, _idxPath, _recoding, _skipInvalidLoci, _bufferSize), err) =>
        val mb = cb.emb

        val ctx = cb.emb.ecb.ctx
        val localTmpBase = ExecuteContext.createTmpPathNoCleanup(ctx.localTmpdir, "index_bgen_")

        val path = _path.asString.loadString(cb)
        val idxPath = _idxPath.asString.loadString(cb)
        val recoding = cb.memoize(coerce[Map[String, String]](svalueToJavaValue(cb, er.region, _recoding)))
        val skipInvalidLoci = _skipInvalidLoci.asBoolean.value
        val bufferSize = _bufferSize.asInt.value

        val cbfis = cb.memoize(Code.newInstance[HadoopFSDataBinaryReader, SeekableDataInputStream](
          mb.getFS.invoke[String, Boolean, SeekableDataInputStream]("openNoCompression", path, false)))

        val header = cb.memoize(Code.invokeScalaObject3[HadoopFSDataBinaryReader, String, Long, BgenHeader](
          LoadBgen.getClass, "readState", cbfis, path, mb.getFS.invoke[String, Long]("getFileSize", path)))

        cb.ifx(header.invoke[Int]("version") cne 2, {
          cb._fatalWithError(err, "BGEN not version 2: ", path, ", version=", header.invoke[Int]("version").toS)
        })
        val nSamples = cb.memoize(header.invoke[Int]("nSamples"))

        val fileIdx = const(-1) // unused
        val compression = cb.memoize(header.invoke[Int]("compression"))
        val dataStart = cb.memoize(header.invoke[Long]("dataStart"))
        val nVariants = cb.memoize(header.invoke[Int]("nVariants").toL)

        val rg = locType match {
          case TLocus(rg) => Some(rg)
          case _ => None
        }

        val settings: BgenSettings = BgenSettings(
          0, // nSamples not used if there are no entries
          TableType(rowType = TStruct(
            "locus" -> TLocus.schemaFromRG(rg),
            "alleles" -> TArray(TString),
            "offset" -> TInt64),
            key = Array("locus", "alleles"),
            globalType = TStruct.empty),
          rg,
          TStruct()
        )

        val nFilesMax = cb.memoize((nVariants / bufferSize.toL + 1L).toI)
        val groupIndex = cb.newLocal[Int]("nFiles", 0)
        val paths = cb.memoize(Code.newArray[String](nFilesMax), "paths")
        val fileSizes = cb.memoize(Code.newArray[Int](nFilesMax), "fileSizes")

        val rowPType = PCanonicalStruct("locus" -> PType.canonical(locType, true, true),
          "alleles" -> PCanonicalArray(PCanonicalString(true), true),
          "offset" -> PInt64Required)
        val bufferSct = SingleCodeType.fromSType(rowPType.sType)
        val buffer = new StagedArrayBuilder(bufferSct, true, mb, 8)
        val currSize = cb.newLocal[Int]("currSize", 0)

        val spec = TypedCodecSpec(
          PType.canonical(TStruct("locus" -> locType, "alleles" -> TArray(TString), "offset" -> TInt64)),
          BufferSpec.wireSpec
        )

        def dumpBuffer(cb: EmitCodeBuilder) = {
          val sorter = new ArraySorter(er, buffer)
          sorter.sort(cb, er.region, { case (cb, region, l, r) =>
            val lv = bufferSct.loadToSValue(cb, l).asBaseStruct.subset("locus", "alleles")
            val rv = bufferSct.loadToSValue(cb, r).asBaseStruct.subset("locus", "alleles")
            cb.emb.ecb.getOrdering(lv.st, rv.st).ltNonnull(cb, lv, rv)
          })

          val path = cb.newLocal[String]("currFile", const(localTmpBase).concat(groupIndex.toS)
            .concat("-").concat(Code.invokeScalaObject0[String](BGENFunctions.getClass, "uuid")))
          val ob = cb.newLocal[OutputBuffer]("currFile", spec.buildCodeOutputBuffer(mb.create(path)))

          val i = cb.newLocal[Int]("i", 0)
          cb.whileLoop(i < currSize, {
            val k = bufferSct.loadToSValue(cb, cb.memoizeAny(buffer.apply(i), buffer.ti))
            spec.encodedType.buildEncoder(k.st, mb.ecb).apply(cb, k, ob)
            cb.assign(i, i + 1)
          })
          cb += paths.update(groupIndex, path)
          cb += fileSizes.update(groupIndex, currSize)
          cb += ob.invoke[Unit]("close")

          cb.assign(groupIndex, groupIndex + 1)
          cb += buffer.clear
          cb.assign(currSize, 0)
        }

        cb += cbfis.invoke[Long, Unit]("seek", dataStart)

        val nRead = cb.newLocal[Long]("nRead", 0L)
        val nWritten = cb.newLocal[Long]("nWritten", 0L)
        cb.whileLoop(nRead < nVariants, {
          StagedBGENReader.decodeRow(cb, er.region, cbfis, nSamples, fileIdx, compression, skipInvalidLoci, recoding,
            TStruct("locus" -> locType, "alleles" -> TArray(TString), "offset" -> TInt64), rg).toI(cb).consume(cb, {
            // do nothing if missing (invalid locus)
          }, { case row: SBaseStructValue =>
            cb.ifx(currSize ceq bufferSize, {
              dumpBuffer(cb)
            })
            cb += buffer.add(bufferSct.coerceSCode(cb, row, er.region, false).code)
            cb.assign(currSize, currSize + 1)
            cb.assign(nWritten, nWritten + 1)
          })
          cb.assign(nRead, nRead + 1)
        })
        cb.ifx(currSize > 0, dumpBuffer(cb))


        val ecb = cb.emb.genEmitClass[Unit]("buffer_stream")
        ecb.cb.addInterface(typeInfo[NoBoxLongIterator].iname)

        val ctor = ecb.newEmitMethod("<init>", FastIndexedSeq[ParamType](typeInfo[String], typeInfo[Int]), UnitInfo)
        val ib = ecb.genFieldThisRef[InputBuffer]("ib")
        val iterSize = ecb.genFieldThisRef[Int]("size")
        val iterCurrIdx = ecb.genFieldThisRef[Int]("currIdx")
        val iterEltRegion = ecb.genFieldThisRef[Region]("eltRegion")
        val iterEOS = ecb.genFieldThisRef[Boolean]("eos")
        ctor.voidWithBuilder { cb =>
          val L = new lir.Block()
          L.append(
            lir.methodStmt(INVOKESPECIAL,
              "java/lang/Object",
              "<init>",
              "()V",
              false,
              UnitInfo,
              FastIndexedSeq(lir.load(ctor.mb._this.asInstanceOf[LocalRef[_]].l))))
          cb += new VCode(L, L, null)

          val path = cb.memoize(ctor.getCodeParam[String](1))
          val _size = cb.memoize(ctor.getCodeParam[Int](2))

          cb.assign(ib, spec.buildCodeInputBuffer(mb.open(path, false)))
          cb.assign(iterSize, _size)
          cb.assign(iterCurrIdx, 0)
        }

        val next = ecb.newEmitMethod("next", FastIndexedSeq[ParamType](), LongInfo)

        val init = ecb.newEmitMethod("init", FastIndexedSeq[ParamType](typeInfo[Region], typeInfo[Region]), UnitInfo)
        init.voidWithBuilder { cb =>
          val eltRegion = init.getCodeParam[Region](2)

          cb.assign(iterEltRegion, eltRegion)
        }

        next.emitWithBuilder { cb =>
          val ret = cb.newLocal[Long]("ret")
          cb.ifx(iterCurrIdx < iterSize, {
            cb.assign(ret, rowPType.store(cb, iterEltRegion,
              spec.encodedType.buildDecoder(rowPType.virtualType, ecb).apply(cb, iterEltRegion, ib), false))
            cb.assign(iterCurrIdx, iterCurrIdx + 1)
          }, {
            cb.assign(iterEOS, true)
            cb.assign(ret, 0L)
          })
          ret
        }

        val isEOS = ecb.newEmitMethod("eos", FastIndexedSeq[ParamType](), BooleanInfo)
        isEOS.emitWithBuilder[Boolean](cb => iterEOS)

        val close = ecb.newEmitMethod("close", FastIndexedSeq[ParamType](), UnitInfo)
        close.voidWithBuilder(cb => cb += ib.invoke[Unit]("close"))

        val iters = mb.genFieldThisRef[Array[NoBoxLongIterator]]("iters")
        cb.assign(iters, Code.newArray[NoBoxLongIterator](groupIndex))
        val i = cb.newLocal[Int]("i")
        cb.whileLoop(i < groupIndex, {
          cb += iters.update(i, coerce[NoBoxLongIterator](Code.newInstance(ecb.cb, ctor.mb, FastIndexedSeq(paths(i), fileSizes(i)))))
          cb.assign(i, i + 1)
        })

        val mergedStream = StreamUtils.multiMergeIterators(cb, Right(true), iters, FastIndexedSeq("locus", "alleles"), rowPType)

        val iw = StagedIndexWriter.withDefaults(settings.indexKeyType, mb.ecb, annotationType = +PCanonicalStruct())
        iw.init(cb, idxPath, cb.memoize(Code.invokeScalaObject3[String, Map[String, String], Boolean, Map[String, Any]](
          BGENFunctions.getClass, "wrapAttrs", mb.getObject(rg.orNull), recoding, skipInvalidLoci)))

        val nAdded = cb.newLocal[Long]("nAdded", 0)
        mergedStream.memoryManagedConsume(er.region, cb) { cb =>
          val row = mergedStream.element.toI(cb).get(cb).asBaseStruct
          val key = row.subset("locus", "alleles")
          val offset = row.loadField(cb, "offset").get(cb).asInt64.value
          cb.assign(nAdded, nAdded + 1)
          iw.add(cb, IEmitCode.present(cb, key), offset, IEmitCode.present(cb, SStackStruct.constructFromArgs(cb, er.region, TStruct())))
        }
        cb.ifx(nWritten cne nAdded, cb._fatal(s"nWritten != nAdded - ", nWritten.toS, ", ", nAdded.toS))

        iw.close(cb)
        cb += cbfis.invoke[Unit]("close")
        primitive(nWritten)
    }
  }

  def wrapAttrs(rg: String, recoding: Map[String, String], skipInvalidLoci: Boolean): Map[String, Any] = {
    Map("reference_genome" -> rg,
      "contig_recoding" -> recoding,
      "skip_invalid_loci" -> skipInvalidLoci)
  }
}
