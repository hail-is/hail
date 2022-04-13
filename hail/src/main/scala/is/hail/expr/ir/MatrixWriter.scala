package is.hail.expr.ir

import scala.language.existentials
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.functions.MatrixWriteBlockMatrix
import is.hail.expr.ir.lowering.{LowererUnsupportedOperation, RVDToTableStage, TableStage}
import is.hail.expr.ir.streams.StreamProducer
import is.hail.expr.{JSONAnnotationImpex, Nat}
import is.hail.io._
import is.hail.io.fs.FS
import is.hail.io.gen.{ExportBGEN, ExportGen}
import is.hail.io.index.StagedIndexWriter
import is.hail.io.plink.ExportPlink
import is.hail.io.vcf.{ExportVCF, TabixVCF}
import is.hail.linalg.BlockMatrix
import is.hail.rvd.{IndexSpec, RVDPartitioner, RVDSpecMaker}
import is.hail.types.encoded.{EBaseStruct, EBlockMatrixNDArray, EType}
import is.hail.types.physical.stypes.{EmitType, SValue}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives._
import is.hail.types.physical.{PBooleanRequired, PCanonicalBaseStruct, PCanonicalString, PCanonicalStruct, PInt64, PStruct, PType}
import is.hail.types.virtual._
import is.hail.types._
import is.hail.types.physical.stypes.concrete.{SJavaString, SJavaArrayString, SJavaArrayStringValue, SStackStruct}
import is.hail.types.physical.stypes.interfaces.{SIndexableValue, SBaseStructValue}
import is.hail.types.physical.stypes.primitives.{SBooleanValue, SInt64Value}
import is.hail.utils._
import is.hail.utils.richUtils.ByteTrackingOutputStream
import is.hail.variant.{ReferenceGenome, Call}
import org.apache.spark.sql.Row
import org.json4s.jackson.JsonMethods
import org.json4s.{DefaultFormats, Formats, ShortTypeHints}

import java.io.OutputStream

object MatrixWriter {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(
      List(classOf[MatrixNativeWriter], classOf[MatrixVCFWriter], classOf[MatrixGENWriter],
        classOf[MatrixBGENWriter], classOf[MatrixPLINKWriter], classOf[WrappedMatrixWriter],
        classOf[MatrixBlockMatrixWriter]), typeHintFieldName = "name")
  }
}

case class WrappedMatrixWriter(writer: MatrixWriter,
  colsFieldName: String,
  entriesFieldName: String,
  colKey: IndexedSeq[String]) extends TableWriter {
  def path: String = writer.path
  def apply(ctx: ExecuteContext, tv: TableValue): Unit = writer(ctx, tv.toMatrixValue(colKey, colsFieldName, entriesFieldName))
  override def lower(ctx: ExecuteContext, ts: TableStage, t: TableIR, r: RTable, relationalLetsAbove: Map[String, IR]): IR =
    writer.lower(colsFieldName, entriesFieldName, colKey, ctx, ts, t, r, relationalLetsAbove)

  override def canLowerEfficiently: Boolean = writer.canLowerEfficiently
}

abstract class MatrixWriter {
  def path: String
  def apply(ctx: ExecuteContext, mv: MatrixValue): Unit
  def lower(colsFieldName: String, entriesFieldName: String, colKey: IndexedSeq[String],
    ctx: ExecuteContext, ts: TableStage, t: TableIR, r: RTable, relationalLetsAbove: Map[String, IR]): IR =
    throw new LowererUnsupportedOperation(s"${ this.getClass } does not have defined lowering!")

  def canLowerEfficiently: Boolean = false
}

case class MatrixNativeWriter(
  path: String,
  overwrite: Boolean = false,
  stageLocally: Boolean = false,
  codecSpecJSONStr: String = null,
  partitions: String = null,
  partitionsTypeStr: String = null,
  checkpointFile: String = null
) extends MatrixWriter {
  def apply(ctx: ExecuteContext, mv: MatrixValue): Unit = mv.write(ctx, path, overwrite, stageLocally, codecSpecJSONStr, partitions, partitionsTypeStr, checkpointFile)

  override def canLowerEfficiently: Boolean = !stageLocally && checkpointFile == null

  override def lower(colsFieldName: String, entriesFieldName: String, colKey: IndexedSeq[String],
    ctx: ExecuteContext, tablestage: TableStage, t: TableIR, r: RTable, relationalLetsAbove: Map[String, IR]): IR = {
    val bufferSpec: BufferSpec = BufferSpec.parseOrDefault(codecSpecJSONStr)
    val tm = MatrixType.fromTableType(t.typ, colsFieldName, entriesFieldName, colKey)
    val rm = r.asMatrixType(colsFieldName, entriesFieldName)

    val lowered =
      if (partitions != null) {
        val partitionsType = IRParser.parseType(partitionsTypeStr)
        val jv = JsonMethods.parse(partitions)
        val rangeBounds = JSONAnnotationImpex.importAnnotation(jv, partitionsType)
          .asInstanceOf[IndexedSeq[Interval]]
        tablestage.repartitionNoShuffle(new RVDPartitioner(tm.rowKey.toArray, tm.rowKeyStruct, rangeBounds))
      } else tablestage

    if (checkpointFile != null) {
      warn(s"lowered execution does not support checkpoint files")
    }

    val rowSpec = TypedCodecSpec(EType.fromTypeAndAnalysis(tm.rowType, rm.rowType), tm.rowType, bufferSpec)
    val entrySpec = TypedCodecSpec(EType.fromTypeAndAnalysis(tm.entriesRVType, rm.entriesRVType), tm.entriesRVType, bufferSpec)
    val colSpec = TypedCodecSpec(EType.fromTypeAndAnalysis(tm.colType, rm.colType), tm.colType, bufferSpec)
    val globalSpec = TypedCodecSpec(EType.fromTypeAndAnalysis(tm.globalType, rm.globalType), tm.globalType, bufferSpec)
    val emptySpec = TypedCodecSpec(EBaseStruct(FastIndexedSeq(), required = true), TStruct.empty, bufferSpec)

    // write out partitioner key, which may be stricter than table key
    val partitioner = lowered.partitioner
    val pKey: PStruct = coerce[PStruct](rowSpec.decodedPType(partitioner.kType))

    val emptyWriter = PartitionNativeWriter(emptySpec, IndexedSeq(), s"$path/globals/globals/parts/", None, None)
    val globalWriter = PartitionNativeWriter(globalSpec, IndexedSeq(), s"$path/globals/rows/parts/", None, None)
    val colWriter = PartitionNativeWriter(colSpec, IndexedSeq(), s"$path/cols/rows/parts/", None, None)
    val rowWriter = SplitPartitionNativeWriter(
      rowSpec, s"$path/rows/rows/parts/",
      entrySpec, s"$path/entries/rows/parts/",
      pKey.virtualType.fieldNames, Some(s"$path/index/" -> pKey), if (stageLocally) Some(ctx.localTmpdir) else None)

    lowered.mapContexts { oldCtx =>
      val d = digitsNeeded(lowered.numPartitions)
      val partFiles = Array.tabulate(lowered.numPartitions)(i => s"${ partFile(d, i) }-")

      zip2(oldCtx, ToStream(Literal(TArray(TString), partFiles.toFastIndexedSeq)), ArrayZipBehavior.AssertSameLength) { (ctxElt, pf) =>
        MakeStruct(FastSeq("oldCtx" -> ctxElt, "writeCtx" -> pf))
      }
    }(GetField(_, "oldCtx")).mapCollectWithContextsAndGlobals(relationalLetsAbove) { (rows, ctx) =>
      WritePartition(rows, GetField(ctx, "writeCtx") + UUID4(), rowWriter)
    } { (parts, globals) =>
      val writeEmpty = WritePartition(MakeStream(FastSeq(makestruct()), TStream(TStruct.empty)), Str(partFile(1, 0)), emptyWriter)
      val writeCols = WritePartition(ToStream(GetField(globals, colsFieldName)), Str(partFile(1, 0)), colWriter)
      val writeGlobals = WritePartition(MakeStream(FastSeq(SelectFields(globals, tm.globalType.fieldNames)), TStream(tm.globalType)),
        Str(partFile(1, 0)), globalWriter)

      val globalTableWriter = TableSpecWriter(s"$path/globals", TableType(tm.globalType, FastIndexedSeq(), TStruct.empty), "rows", "globals", "../references", log = false)
      val colTableWriter = TableSpecWriter(s"$path/cols", tm.colsTableType.copy(key = FastIndexedSeq[String]()), "rows", "../globals/rows", "../references", log = false)
      val rowTableWriter = TableSpecWriter(s"$path/rows", tm.rowsTableType, "rows", "../globals/rows", "../references", log = false)
      val entriesTableWriter = TableSpecWriter(s"$path/entries", TableType(tm.entriesRVType, FastIndexedSeq(), tm.globalType), "rows", "../globals/rows", "../references", log = false)

      val matrixWriter = MatrixSpecWriter(path, tm, "rows/rows", "globals/rows", "cols/rows", "entries/rows", "references", log = true)

      val rowsIndexSpec = IndexSpec.defaultAnnotation("../../index", coerce[PStruct](pKey))
      val entriesIndexSpec = IndexSpec.defaultAnnotation("../../index", coerce[PStruct](pKey), withOffsetField = true)

      RelationalWriter.scoped(path, overwrite = overwrite, Some(t.typ))(
        RelationalWriter.scoped(s"$path/globals", overwrite = false, None)(
          RelationalWriter.scoped(s"$path/cols", overwrite = false, None)(
            RelationalWriter.scoped(s"$path/rows", overwrite = false, None)(
              RelationalWriter.scoped(s"$path/entries", overwrite = false, None)(
                bindIR(writeCols) { colInfo =>
                  bindIR(parts) { partInfo =>
                    Begin(FastIndexedSeq(
                      WriteMetadata(MakeArray(GetField(writeEmpty, "filePath")),
                        RVDSpecWriter(s"$path/globals/globals", RVDSpecMaker(emptySpec, RVDPartitioner.unkeyed(1)))),
                      WriteMetadata(MakeArray(GetField(writeGlobals, "filePath")),
                        RVDSpecWriter(s"$path/globals/rows", RVDSpecMaker(globalSpec, RVDPartitioner.unkeyed(1)))),
                      WriteMetadata(MakeArray(MakeStruct(Seq("partitionCounts" -> I64(1), "distinctlyKeyed" -> True(), "firstKey" -> MakeStruct(Seq()), "lastKey" -> MakeStruct(Seq())))), globalTableWriter),
                      WriteMetadata(MakeArray(GetField(colInfo, "filePath")),
                        RVDSpecWriter(s"$path/cols/rows", RVDSpecMaker(colSpec, RVDPartitioner.unkeyed(1)))),
                      WriteMetadata(MakeArray(SelectFields(colInfo, IndexedSeq("partitionCounts", "distinctlyKeyed", "firstKey", "lastKey"))), colTableWriter),
                      bindIR(ToArray(mapIR(ToStream(partInfo)) { fc => GetField(fc, "filePath") })) { files =>
                        Begin(FastIndexedSeq(
                          WriteMetadata(files, RVDSpecWriter(s"$path/rows/rows", RVDSpecMaker(rowSpec, lowered.partitioner, rowsIndexSpec))),
                          WriteMetadata(files, RVDSpecWriter(s"$path/entries/rows", RVDSpecMaker(entrySpec, RVDPartitioner.unkeyed(lowered.numPartitions), entriesIndexSpec)))))
                      },
                      bindIR(ToArray(mapIR(ToStream(partInfo)) { fc => SelectFields(fc, Seq("partitionCounts", "distinctlyKeyed", "firstKey", "lastKey")) })) { countsAndKeyInfo =>
                        Begin(FastIndexedSeq(
                            WriteMetadata(countsAndKeyInfo, rowTableWriter),
                            WriteMetadata(
                              ToArray(mapIR(ToStream(countsAndKeyInfo)){countAndKeyInfo =>
                                InsertFields(SelectFields(countAndKeyInfo, IndexedSeq("partitionCounts", "distinctlyKeyed")), IndexedSeq("firstKey" -> MakeStruct(Seq()), "lastKey" -> MakeStruct(Seq())))
                              }),
                              entriesTableWriter),
                            WriteMetadata(
                              makestruct(
                                "cols" -> GetField(colInfo, "partitionCounts"),
                                "rows" -> ToArray(mapIR(ToStream(countsAndKeyInfo)) {countAndKey => GetField(countAndKey, "partitionCounts")})),
                              matrixWriter)))
                      }))
                  }
                })))))
    }
  }
}

case class SplitPartitionNativeWriter(
  spec1: AbstractTypedCodecSpec, partPrefix1: String,
  spec2: AbstractTypedCodecSpec, partPrefix2: String,
  keyFieldNames: IndexedSeq[String],
  index: Option[(String, PStruct)], localDir: Option[String]) extends PartitionWriter {
  def stageLocally: Boolean = localDir.isDefined
  def hasIndex: Boolean = index.isDefined
  val filenameType = PCanonicalString(required = true)
  def pContextType = PCanonicalString()

  val keyType = spec1.encodedVirtualType.asInstanceOf[TStruct].select(keyFieldNames)._1

  def ctxType: Type = TString
  def returnType: Type = TStruct("filePath" -> TString, "partitionCounts" -> TInt64, "distinctlyKeyed" -> TBoolean, "firstKey" -> keyType, "lastKey" -> keyType)
  def unionTypeRequiredness(r: TypeWithRequiredness, ctxType: TypeWithRequiredness, streamType: RIterable): Unit = {
    val rs = r.asInstanceOf[RStruct]
    val rKeyType = streamType.elementType.asInstanceOf[RStruct].select(keyFieldNames.toArray)
    rs.field("firstKey").union(false)
    rs.field("firstKey").unionFrom(rKeyType)
    rs.field("lastKey").union(false)
    rs.field("lastKey").unionFrom(rKeyType)
    r.union(ctxType.required)
    r.union(streamType.required)
  }

  if (stageLocally)
    throw new LowererUnsupportedOperation("stageLocally option not yet implemented")
  def ifIndexed[T >: Null](obj: => T): T = if (hasIndex) obj else null

  def consumeStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    stream: StreamProducer,
    context: EmitCode,
    region: Value[Region]): IEmitCode = {
    val iAnnotationType = PCanonicalStruct(required = true, "entries_offset" -> PInt64())
    val mb = cb.emb

    val indexWriter = ifIndexed { StagedIndexWriter.withDefaults(index.get._2, mb.ecb, annotationType = iAnnotationType) }

    context.toI(cb).map(cb) { pctx =>
      val filename1 = mb.newLocal[String]("filename1")
      val os1 = mb.newLocal[ByteTrackingOutputStream]("write_os1")
      val ob1 = mb.newLocal[OutputBuffer]("write_ob1")
      val filename2 = mb.newLocal[String]("filename2")
      val os2 = mb.newLocal[ByteTrackingOutputStream]("write_os2")
      val ob2 = mb.newLocal[OutputBuffer]("write_ob2")
      val n = mb.newLocal[Long]("partition_count")
      val distinctlyKeyed = mb.newLocal[Boolean]("distinctlyKeyed")
      cb.assign(distinctlyKeyed, !keyFieldNames.isEmpty) // True until proven otherwise, if there's a key to care about all.

      val keyEmitType = EmitType(spec1.decodedPType(keyType).sType, false)

      val firstSeenSettable =  mb.newEmitLocal("pnw_firstSeen", keyEmitType)
      val lastSeenSettable =  mb.newEmitLocal("pnw_lastSeen", keyEmitType)
      // Start off missing, we will use this to determine if we haven't processed any rows yet.
      cb.assign(firstSeenSettable, EmitCode.missing(cb.emb, keyEmitType.st))
      cb.assign(lastSeenSettable, EmitCode.missing(cb.emb, keyEmitType.st))


      def writeFile(cb: EmitCodeBuilder, codeRow: EmitCode): Unit = {
        val row = codeRow.toI(cb).get(cb, "row can't be missing").asBaseStruct

        if (hasIndex) {
          indexWriter.add(cb, {
            val indexKeyPType = index.get._2
            IEmitCode.present(cb, indexKeyPType.asInstanceOf[PCanonicalBaseStruct]
              .constructFromFields(cb, stream.elementRegion,
                indexKeyPType.fields.map(f => EmitCode.fromI(cb.emb)(cb => row.loadField(cb, f.name))),
                deepCopy = false))
          }, ob1.invoke[Long]("indexOffset"), {
            IEmitCode.present(cb,
              iAnnotationType.constructFromFields(cb, stream.elementRegion,
                FastIndexedSeq(EmitCode.present(cb.emb, primitive(cb.memoize(ob2.invoke[Long]("indexOffset"))))),
                deepCopy = false))
          })
        }

        val key = SStackStruct.constructFromArgs(cb, stream.elementRegion, keyType, keyType.fields.map { f =>
          EmitCode.fromI(cb.emb)(cb => row.loadField(cb, f.name))
        }:_*)

        if (!keyFieldNames.isEmpty) {
          cb.ifx(distinctlyKeyed, {
            lastSeenSettable.loadI(cb).consume(cb, {
              // If there's no last seen, we are in the first row.
              cb.assign(firstSeenSettable, EmitValue.present(key.copyToRegion(cb, region, firstSeenSettable.st)))
            }, { lastSeen =>
              val comparator = EQ(lastSeenSettable.emitType.virtualType).codeOrdering(cb.emb.ecb, lastSeenSettable.st, key.st)
              val equalToLast = comparator(cb, lastSeenSettable, EmitValue.present(key))
              cb.ifx(equalToLast.asInstanceOf[Value[Boolean]], {
                cb.assign(distinctlyKeyed, false)
              })
            })
          })
          cb.assign(lastSeenSettable, IEmitCode.present(cb, key.copyToRegion(cb, region, lastSeenSettable.st)))
        }

        cb += ob1.writeByte(1.asInstanceOf[Byte])

        spec1.encodedType.buildEncoder(row.st, cb.emb.ecb)
          .apply(cb, row, ob1)

        cb += ob2.writeByte(1.asInstanceOf[Byte])

        spec2.encodedType.buildEncoder(row.st, cb.emb.ecb)
          .apply(cb, row, ob2)
        cb.assign(n, n + 1L)
      }

      cb.assign(filename1, pctx.asString.loadString(cb))
      if (hasIndex) {
        val indexFile = cb.newLocal[String]("indexFile")
        cb.assign(indexFile, const(index.get._1).concat(filename1).concat(".idx"))
        indexWriter.init(cb, indexFile)
      }
      cb.assign(filename2, const(partPrefix2).concat(filename1))
      cb.assign(filename1, const(partPrefix1).concat(filename1))
      cb.assign(os1, Code.newInstance[ByteTrackingOutputStream, OutputStream](mb.create(filename1)))
      cb.assign(os2, Code.newInstance[ByteTrackingOutputStream, OutputStream](mb.create(filename2)))
      cb.assign(ob1, spec1.buildCodeOutputBuffer(Code.checkcast[OutputStream](os1)))
      cb.assign(ob2, spec2.buildCodeOutputBuffer(Code.checkcast[OutputStream](os2)))
      cb.assign(n, 0L)

      stream.memoryManagedConsume(region, cb) { cb =>
        writeFile(cb, stream.element)
      }

      cb += ob1.writeByte(0.asInstanceOf[Byte])
      cb += ob2.writeByte(0.asInstanceOf[Byte])
      if (hasIndex)
        indexWriter.close(cb)
      cb += ob1.flush()
      cb += ob2.flush()
      cb += os1.invoke[Unit]("close")
      cb += os2.invoke[Unit]("close")


      SStackStruct.constructFromArgs(cb, region, returnType.asInstanceOf[TBaseStruct],
        EmitCode.present(mb, pctx),
        EmitCode.present(mb, new SInt64Value(n)),
        EmitCode.present(mb, new SBooleanValue(distinctlyKeyed)),
        firstSeenSettable,
        lastSeenSettable
      )
    }
  }
}

class MatrixSpecHelper(
  path: String, rowRelPath: String, globalRelPath: String, colRelPath: String, entryRelPath: String, refRelPath: String, typ: MatrixType, log: Boolean
) extends Serializable {
  def write(fs: FS, nCols: Long, partCounts: Array[Long]): Unit = {
    val spec = MatrixTableSpecParameters(
      FileFormat.version.rep,
      is.hail.HAIL_PRETTY_VERSION,
      "references",
      typ,
      Map("globals" -> RVDComponentSpec(globalRelPath),
        "cols" -> RVDComponentSpec(colRelPath),
        "rows" -> RVDComponentSpec(rowRelPath),
        "entries" -> RVDComponentSpec(entryRelPath),
        "partition_counts" -> PartitionCountsComponentSpec(partCounts)))

    spec.write(fs, path)

    val nRows = partCounts.sum
    info(s"wrote matrix table with $nRows ${ plural(nRows, "row") } " +
      s"and ${ nCols } ${ plural(nCols, "column") } " +
      s"in ${ partCounts.length } ${ plural(partCounts.length, "partition") } " +
      s"to $path")
  }
}

case class MatrixSpecWriter(path: String, typ: MatrixType, rowRelPath: String, globalRelPath: String, colRelPath: String, entryRelPath: String, refRelPath: String, log: Boolean) extends MetadataWriter {
  def annotationType: Type = TStruct("cols" -> TInt64, "rows" -> TArray(TInt64))

  def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region]): Unit = {
    cb += cb.emb.getFS.invoke[String, Unit]("mkDir", path)
    val c = writeAnnotations.get(cb, "write annotations can't be missing!").asBaseStruct
    val partCounts = cb.newLocal[Array[Long]]("partCounts")
    val a = c.loadField(cb, "rows").get(cb).asIndexable

    val n = cb.newLocal[Int]("n", a.loadLength())
    val i = cb.newLocal[Int]("i", 0)
    cb.assign(partCounts, Code.newArray[Long](n))
    cb.whileLoop(i < n, {
      val count = a.loadElement(cb, i).get(cb, "part count can't be missing!")
      cb += partCounts.update(i, count.asInt64.value)
      cb.assign(i, i + 1)
    })
    cb += cb.emb.getObject(new MatrixSpecHelper(path, rowRelPath, globalRelPath, colRelPath, entryRelPath, refRelPath, typ, log))
      .invoke[FS, Long, Array[Long], Unit]("write", cb.emb.getFS, c.loadField(cb, "cols").get(cb).asInt64.value, partCounts)
  }
}

case class MatrixVCFWriter(
  path: String,
  append: Option[String] = None,
  exportType: String = ExportType.CONCATENATED,
  metadata: Option[VCFMetadata] = None,
  tabix: Boolean = false
) extends MatrixWriter {
  def apply(ctx: ExecuteContext, mv: MatrixValue): Unit = {
    val appendStr = getAppendHeaderValue(ctx.fs)
    val tv = mv.toTableValue
    val ts = RVDToTableStage(tv.rvd, tv.globals.toEncodedLiteral(ctx.theHailClassLoader))
    val tl = TableLiteral(tv, ctx.theHailClassLoader)
    CompileAndEvaluate(ctx,
      lower(LowerMatrixIR.colsFieldName, MatrixType.entriesIdentifier, mv.typ.colKey,
        ctx, ts, tl, BaseTypeWithRequiredness(tv.typ).asInstanceOf[RTable], Map()))
  }

  override def canLowerEfficiently: Boolean = true
  override def lower(colsFieldName: String, entriesFieldName: String, colKey: IndexedSeq[String],
      ctx: ExecuteContext, ts: TableStage, t: TableIR, r: RTable, relationalLetsAbove: Map[String, IR]): IR = {
    require(exportType != ExportType.PARALLEL_COMPOSABLE)

    val tm = MatrixType.fromTableType(t.typ, colsFieldName, entriesFieldName, colKey)
    tm.requireRowKeyVariant()
    tm.requireColKeyString()
    ExportVCF.checkFormatSignature(tm.entryType)

    val ext = ctx.fs.getCodecExtension(path)

    val folder = if (exportType == ExportType.CONCATENATED)
      ctx.createTmpPath("write-vcf-concatenated")
    else
      path

    val appendStr = getAppendHeaderValue(ctx.fs)

    val writeHeader = exportType == ExportType.PARALLEL_HEADER_IN_SHARD
    val partAppend = appendStr.filter(_ => writeHeader)
    val partMetadata = metadata.filter(_ => writeHeader)
    val lineWriter = VCFPartitionWriter(tm, entriesFieldName, writeHeader = exportType == ExportType.PARALLEL_HEADER_IN_SHARD,
      partAppend, partMetadata, tabix && exportType != ExportType.CONCATENATED)

    ts.mapContexts { oldCtx =>
      val d = digitsNeeded(ts.numPartitions)
      val partFiles = Literal(TArray(TString), Array.tabulate(ts.numPartitions)(i => s"$folder/${ partFile(d, i) }$ext").toFastIndexedSeq)

      zip2(oldCtx, ToStream(partFiles), ArrayZipBehavior.AssertSameLength) { (ctxElt, pf) =>
        MakeStruct(FastSeq(
          "oldCtx" -> ctxElt,
          "partFile" -> pf))
      }
    }(GetField(_, "oldCtx")).mapCollectWithContextsAndGlobals(relationalLetsAbove) { (rows, ctxRef) =>
      val ctx = MakeStruct(FastSeq(
        "cols" -> GetField(ts.globals, colsFieldName),
        "partFile" -> GetField(ctxRef, "partFile")))
      WritePartition(rows, ctx, lineWriter)
    }{ (parts, globals) =>
      val ctx = MakeStruct(FastSeq("cols" -> GetField(globals, colsFieldName), "partFiles" -> parts))
      val commit = VCFExportFinalizer(tm, path, appendStr, metadata, exportType, tabix)
      Begin(FastIndexedSeq(WriteMetadata(ctx, commit)))
    }
  }

  private def getAppendHeaderValue(fs: FS): Option[String] = append.map { f =>
    using(fs.open(f)) { s =>
      val sb = new StringBuilder
      scala.io.Source.fromInputStream(s)
        .getLines()
        .filterNot(_.isEmpty)
        .foreach { line =>
          sb.append(line)
          sb += '\n'
        }
      sb.result()
    }
  }
}

case class VCFPartitionWriter(typ: MatrixType, entriesFieldName: String, writeHeader: Boolean,
    append: Option[String], metadata: Option[VCFMetadata], tabix: Boolean) extends PartitionWriter {
  val ctxType: Type = TStruct("cols" -> TArray(typ.colType), "partFile" -> TString)

  if (typ.rowType.hasField("info")) {
    typ.rowType.field("info").typ match {
      case _: TStruct =>
      case t =>
        warn(s"export_vcf found row field 'info' of type $t, but expected type 'Struct'. Emitting no INFO fields.")
    }
  } else {
    warn(s"export_vcf found no row field 'info'. Emitting no INFO fields.")
  }

  val formatFieldOrder: Array[Int] = typ.entryType.fieldIdx.get("GT") match {
    case Some(i) => (i +: typ.entryType.fields.filter(fd => fd.name != "GT").map(_.index)).toArray
    case None => typ.entryType.fields.indices.toArray
  }
  val formatFieldString = formatFieldOrder.map(i => typ.entryType.fields(i).name).mkString(":")
  val missingFormatStr = if (typ.entryType.size > 0 && typ.entryType.types(formatFieldOrder(0)) == TCall)
    "./."
    else
      "."

  val locusIdx = typ.rowType.fieldIdx("locus")
  val allelesIdx = typ.rowType.fieldIdx("alleles")
  val (idExists, idIdx) = ExportVCF.lookupVAField(typ.rowType, "rsid", "ID", Some(TString))
  val (qualExists, qualIdx) = ExportVCF.lookupVAField(typ.rowType, "qual", "QUAL", Some(TFloat64))
  val (filtersExists, filtersIdx) = ExportVCF.lookupVAField(typ.rowType, "filters", "FILTERS", Some(TSet(TString)))
  val (infoExists, infoIdx) = ExportVCF.lookupVAField(typ.rowType, "info", "INFO", None)

  def returnType: Type = TString
  def unionTypeRequiredness(r: TypeWithRequiredness, ctxType: TypeWithRequiredness, streamType: RIterable): Unit = {
    r.union(ctxType.required)
    r.union(streamType.required)
  }

  final def consumeStream(ctx: ExecuteContext, cb: EmitCodeBuilder, stream: StreamProducer,
      context: EmitCode, region: Value[Region]): IEmitCode = {
    val mb = cb.emb
    context.toI(cb).map(cb) { case ctx: SBaseStructValue =>
      val filename = ctx.loadField(cb, "partFile").get(cb, "partFile can't be missing").asString.loadString(cb)

      val os = cb.memoize(cb.emb.create(filename))
      if (writeHeader) {
        val sampleIds = ctx.loadField(cb, "cols").get(cb).asIndexable
        val stringSampleIds = cb.memoize(Code.newArray[String](sampleIds.loadLength()))
        sampleIds.forEachDefined(cb) { case (cb, i, colv: SBaseStructValue) =>
          val s = colv.subset(typ.colKey: _*).loadField(cb, 0).get(cb).asString
          cb += (stringSampleIds(i) = s.loadString(cb))
        }

        val headerStr = Code.invokeScalaObject6[TStruct, TStruct, ReferenceGenome, Option[String], Option[VCFMetadata], Array[String], String](
          ExportVCF.getClass, "makeHeader",
          mb.getType[TStruct](typ.rowType), mb.getType[TStruct](typ.entryType),
          mb.getReferenceGenome(typ.referenceGenome), mb.getObject(append),
          mb.getObject(metadata), stringSampleIds)
        cb += os.invoke[Array[Byte], Unit]("write", headerStr.invoke[Array[Byte]]("getBytes"))
        cb += os.invoke[Int, Unit]("write", '\n')
      }

      stream.memoryManagedConsume(region, cb) { cb =>
        consumeElement(cb, stream.element, os, stream.elementRegion)
      }

      cb += os.invoke[Unit]("close")

      if (tabix) {
          cb += Code.invokeScalaObject2[FS, String, Unit](TabixVCF.getClass, "apply", cb.emb.getFS, filename)
      }

      SJavaString.construct(cb, filename)
    }
  }

  def consumeElement(cb: EmitCodeBuilder, element: EmitCode, os: Value[OutputStream], region: Value[Region]): Unit = {
    def _writeC(cb: EmitCodeBuilder, code: Code[Int]) = { cb += os.invoke[Int, Unit]("write", code) }
    def _writeB(cb: EmitCodeBuilder, code: Code[Array[Byte]]) = { cb += os.invoke[Array[Byte], Unit]("write", code) }
    def _writeS(cb: EmitCodeBuilder, code: Code[String]) = { _writeB(cb, code.invoke[Array[Byte]]("getBytes")) }
    def writeValue(cb: EmitCodeBuilder, value: SValue) = value match {
      case v: SInt32Value => _writeS(cb, v.value.toS)
      case v: SInt64Value =>
        cb.ifx(v.value > Int.MaxValue || v.value <  Int.MinValue, cb._fatal(
          "Cannot convert Long to Int if value is greater than Int.MaxValue (2^31 - 1) ",
          "or less than Int.MinValue (-2^31). Found ", v.value.toS))
        _writeS(cb, v.value.toS)
      case v: SFloat32Value =>
        cb.ifx(Code.invokeStatic1[java.lang.Float, Float, Boolean]("isNaN", v.value),
          _writeC(cb, '.'),
          _writeS(cb, Code.invokeScalaObject2[String, Float, String](ExportVCF.getClass, "fmtFloat", "%.6g", v.value)))
      case v: SFloat64Value =>
        cb.ifx(Code.invokeStatic1[java.lang.Double, Double, Boolean]("isNaN", v.value),
          _writeC(cb, '.'),
          _writeS(cb, Code.invokeScalaObject2[String, Double, String](ExportVCF.getClass, "fmtDouble", "%.6g", v.value)))
      case v: SStringValue =>
        _writeB(cb, v.toBytes(cb).loadBytes(cb))
      case v: SCallValue =>
        val ploidy = v.ploidy(cb)
        val phased = v.isPhased(cb)
        cb.ifx(ploidy.ceq(0), cb._fatal("VCF spec does not support 0-ploid calls."))
        cb.ifx(ploidy.ceq(1) , cb._fatal("VCF spec does not support phased haploid calls."))
        val c = v.canonicalCall(cb)
        _writeS(cb, Code.invokeScalaObject1[Int, String](Call.getClass, "toString", c))
      case _ =>
        fatal(s"VCF does not support ${value.st}")
    }

    def writeIterable(cb: EmitCodeBuilder, it: SIndexableValue, delim: Int) =
      it.forEachDefinedOrMissing(cb)({ (cb, i) =>
        cb.ifx(i.cne(0), _writeC(cb, delim))
        _writeC(cb, '.')
      }, { (cb, i, value) =>
        cb.ifx(i.cne(0), _writeC(cb, delim))
        writeValue(cb, value)
      })

    def writeGenotype(cb: EmitCodeBuilder, gt: SBaseStructValue) = {
      val end = cb.newLocal[Int]("lastDefined", -1)
      val Lend = CodeLabel()
      formatFieldOrder.zipWithIndex.reverse.foreach { case (idx, pos) =>
        cb.ifx(!gt.isFieldMissing(cb, idx), {
          cb.assign(end, pos)
          cb.goto(Lend)
        })
      }

      cb.define(Lend)

      val Lout = CodeLabel()

      cb.ifx(end < 0, {
        _writeS(cb, missingFormatStr)
        cb.goto(Lout)
      })

      formatFieldOrder.zipWithIndex.foreach { case (idx, pos) =>
        if (pos != 0)
          _writeC(cb, ':')

        gt.loadField(cb, idx).consume(cb, {
          if (gt.st.fieldTypes(idx).virtualType == TCall)
            _writeS(cb, "./.")
          else
            _writeC(cb, '.')
        }, {
          case value: SIndexableValue =>
            writeIterable(cb, value, ',')
          case value =>
            writeValue(cb, value)
        })

        cb.ifx(end.ceq(pos), cb.goto(Lout))
      }

      cb.define(Lout)
    }

    def writeC(code: Code[Int]) = _writeC(cb, code)
    def writeB(code: Code[Array[Byte]]) = _writeB(cb, code)
    def writeS(code: Code[String]) = _writeS(cb, code)

    val elt = element.toI(cb).get(cb).asBaseStruct
    val locus = elt.loadField(cb, locusIdx).get(cb).asLocus
    // CHROM
    writeB(locus.contig(cb).toBytes(cb).loadBytes(cb))
    // POS
    writeC('\t')
    writeS(locus.position(cb).toS)

    // ID
    writeC('\t')
    if (idExists)
      elt.loadField(cb, idIdx).consume(cb, writeC('.'), { case id: SStringValue =>
        writeB(id.toBytes(cb).loadBytes(cb))
      })
    else
      writeC('.')

    // REF
    writeC('\t')
    val alleles = elt.loadField(cb, allelesIdx).get(cb).asIndexable
    writeB(alleles.loadElement(cb, 0).get(cb).asString.toBytes(cb).loadBytes(cb))

    // ALT
    writeC('\t')
    cb.ifx(alleles.loadLength() > 1,
      {
        val i = cb.newLocal[Int]("i")
        cb.forLoop(cb.assign(i, 1), i < alleles.loadLength(), cb.assign(i, i + 1), {
          cb.ifx(i.cne(1), writeC(','))
          writeB(alleles.loadElement(cb, i).get(cb).asString.toBytes(cb).loadBytes(cb))
        })
      },
      writeC('.'))

    // QUAL
    writeC('\t')
    if (qualExists)
      elt.loadField(cb, qualIdx).consume(cb, writeC('.'), { qual =>
        writeS(Code.invokeScalaObject2[String, Double, String](ExportVCF.getClass, "fmtDouble", "%.2f", qual.asDouble.value))
      })
    else
      writeC('.')

    // FILTER
    writeC('\t')
    if (filtersExists)
      elt.loadField(cb, filtersIdx).consume(cb, writeC('.'), { case filters: SIndexableValue =>
        cb.ifx(filters.loadLength().ceq(0), writeS("PASS"), {
          writeIterable(cb, filters, ';')
        })
      })
    else
      writeC('.')

    // INFO
    writeC('\t')
    if (infoExists) {
      val wroteInfo = cb.newLocal[Boolean]("wroteInfo", false)

      elt.loadField(cb, infoIdx).consume(cb, { /* do nothing */ }, { case info: SBaseStructValue =>
        var idx = 0
        while (idx < info.st.size) {
          val field = info.st.virtualType.fields(idx)
          info.loadField(cb, idx).consume(cb, { /* do nothing */ }, {
            case infoArray: SIndexableValue if infoArray.st.elementType.virtualType != TBoolean =>
              cb.ifx(infoArray.loadLength() > 0, {
                cb.ifx(wroteInfo, writeC(';'))
                writeS(field.name)
                writeC('=')
                writeIterable(cb, infoArray, ',')
                cb.assign(wroteInfo, true)
              })
            case infoFlag: SBooleanValue =>
              cb.ifx(infoFlag.value, {
                cb.ifx(wroteInfo, writeC(';'))
                writeS(field.name)
                cb.assign(wroteInfo, true)
              })
            case info =>
              cb.ifx(wroteInfo, writeC(';'))
              writeS(field.name)
              writeC('=')
              writeValue(cb, info)
              cb.assign(wroteInfo, true)
          })
          idx += 1
        }
      })

      cb.ifx(!wroteInfo, writeC('.'))
    } else {
      writeC('.')
    }

    // FORMAT
    val genotypes = elt.loadField(cb, entriesFieldName).get(cb).asIndexable
    cb.ifx(genotypes.loadLength() > 0, {
      writeC('\t')
      writeS(formatFieldString)
      genotypes.forEachDefinedOrMissing(cb)({ (cb, _) =>
        _writeC(cb, '\t')
        _writeS(cb, missingFormatStr)
      }, { case (cb, _, gt: SBaseStructValue) =>
        _writeC(cb, '\t')
        writeGenotype(cb, gt)
      })
    })

    writeC('\n')
  }
}

case class VCFExportFinalizer(typ: MatrixType, outputPath: String, append: Option[String],
    metadata: Option[VCFMetadata], exportType: String, tabix: Boolean) extends MetadataWriter {
  def annotationType: Type = TStruct("cols" -> TArray(typ.colType), "partFiles" -> TArray(TString))
  private def header(cb: EmitCodeBuilder, annotations: SBaseStructValue): Code[String] = {
    val mb = cb.emb
    val sampleIds = annotations.loadField(cb, "cols").get(cb).asIndexable
    val stringSampleIds = cb.memoize(Code.newArray[String](sampleIds.loadLength()))
    sampleIds.forEachDefined(cb) { case (cb, i, colv: SBaseStructValue) =>
      val s = colv.subset(typ.colKey: _*).loadField(cb, 0).get(cb).asString
      cb += (stringSampleIds(i) = s.loadString(cb))
    }
    Code.invokeScalaObject6[TStruct, TStruct, ReferenceGenome, Option[String], Option[VCFMetadata], Array[String], String](
      ExportVCF.getClass, "makeHeader",
      mb.getType[TStruct](typ.rowType), mb.getType[TStruct](typ.entryType),
      mb.getReferenceGenome(typ.referenceGenome), mb.getObject(append),
      mb.getObject(metadata), stringSampleIds)
  }

  def writeMetadata(writeAnnotations: => IEmitCode, cb: EmitCodeBuilder, region: Value[Region]): Unit = {
    val ctx: ExecuteContext = cb.emb.ctx
    val ext = ctx.fs.getCodecExtension(outputPath)

    val annotations = writeAnnotations.get(cb).asBaseStruct

    exportType match {
      case ExportType.CONCATENATED =>
        val headerStr = header(cb, annotations)

        val partPaths = annotations.loadField(cb, "partFiles").get(cb)
        val files = partPaths.castTo(cb, region, SJavaArrayString(true), false)
        val headerFilePath = ctx.createTmpPath("header", ext)
        val os = cb.memoize(cb.emb.create(const(headerFilePath)))
        cb += os.invoke[Array[Byte], Unit]("write", headerStr.invoke[Array[Byte]]("getBytes"))
        cb += os.invoke[Int, Unit]("write", '\n')
        cb += os.invoke[Unit]("close")

        val partFiles = files.asInstanceOf[SJavaArrayStringValue].array
        val jFiles = cb.memoize(Code.newArray[String](partFiles.length + 1))
        cb += (jFiles(0) = const(headerFilePath))
        cb += Code.invokeStatic5[System, Any, Int, Any, Int, Int, Unit](
          "arraycopy", partFiles /*src*/, 0 /*srcPos*/, jFiles /*dest*/, 1 /*destPos*/, partFiles.length /*len*/)

        cb += cb.emb.getFS.invoke[Array[String], String, Unit]("concatenateFiles", jFiles, const(outputPath))

        val i = cb.newLocal[Int]("i")
        cb.forLoop(cb.assign(i, 0), i < jFiles.length, cb.assign(i, i + 1), {
          cb += cb.emb.getFS.invoke[String, Boolean, Unit]("delete", jFiles(i), const(false))
        })

        if (tabix) {
          cb += Code.invokeScalaObject2[FS, String, Unit](TabixVCF.getClass, "apply", cb.emb.getFS, const(outputPath))
        }

      case ExportType.PARALLEL_HEADER_IN_SHARD =>
        cb += cb.emb.getFS.invoke[String, Unit]("touch", const(outputPath).concat("/_SUCCESS"))

      case ExportType.PARALLEL_SEPARATE_HEADER =>
        val headerFilePath = s"$outputPath/header$ext"
        val headerStr = header(cb, annotations)

        val os = cb.memoize(cb.emb.create(const(headerFilePath)))
        cb += os.invoke[Array[Byte], Unit]("write", headerStr.invoke[Array[Byte]]("getBytes"))
        cb += os.invoke[Int, Unit]("write", '\n')
        cb += os.invoke[Unit]("close")

        cb += cb.emb.getFS.invoke[String, Unit]("touch", const(outputPath).concat("/_SUCCESS"))
    }
  }
}

case class MatrixGENWriter(
  path: String,
  precision: Int = 4
) extends MatrixWriter {
  def apply(ctx: ExecuteContext, mv: MatrixValue): Unit = ExportGen(ctx, mv, path, precision)
}

case class MatrixBGENWriter(
  path: String,
  exportType: String
) extends MatrixWriter {
  def apply(ctx: ExecuteContext, mv: MatrixValue): Unit = ExportBGEN(ctx, mv, path, exportType)
}

case class MatrixPLINKWriter(
  path: String
) extends MatrixWriter {
  def apply(ctx: ExecuteContext, mv: MatrixValue): Unit = ExportPlink(ctx, mv, path)
}

case class MatrixBlockMatrixWriter(
  path: String,
  overwrite: Boolean,
  entryField: String,
  blockSize: Int
) extends MatrixWriter {
  def apply(ctx: ExecuteContext, mv: MatrixValue): Unit = MatrixWriteBlockMatrix(ctx, mv, entryField, path, overwrite, blockSize)

  override def lower(colsFieldName: String, entriesFieldName: String, colKey: IndexedSeq[String],
    ctx: ExecuteContext, ts: TableStage, t: TableIR, r: RTable, relationalLetsAbove: Map[String, IR]): IR = {

    val tm = MatrixType.fromTableType(t.typ, colsFieldName, entriesFieldName, colKey)
    val rm = r.asMatrixType(colsFieldName, entriesFieldName)

    val countColumnsIR = ArrayLen(GetField(ts.getGlobals(), colsFieldName))
    val numCols: Int = CompileAndEvaluate(ctx, countColumnsIR, true).asInstanceOf[Int]
    val numBlockCols: Int = (numCols - 1) / blockSize + 1
    val lastBlockNumCols = numCols % blockSize

    val rowCountIR = ts.mapCollect(relationalLetsAbove)(paritionIR => StreamLen(paritionIR))
    val inputRowCountPerPartition: IndexedSeq[Int] = CompileAndEvaluate(ctx, rowCountIR).asInstanceOf[IndexedSeq[Int]]
    val inputPartStartsPlusLast = inputRowCountPerPartition.scanLeft(0L)(_ + _)
    val inputPartStarts = inputPartStartsPlusLast.dropRight(1)
    val inputPartStops = inputPartStartsPlusLast.tail

    val numRows = inputPartStartsPlusLast.last
    val numBlockRows: Int = (numRows.toInt - 1) / blockSize + 1

    // Zip contexts with partition starts and ends
    val zippedWithStarts = ts.mapContexts{oldContextsStream => zipIR(IndexedSeq(oldContextsStream, ToStream(Literal(TArray(TInt64), inputPartStarts)), ToStream(Literal(TArray(TInt64), inputPartStops))), ArrayZipBehavior.AssertSameLength){ case IndexedSeq(oldCtx, partStart, partStop) =>
      MakeStruct(Seq[(String, IR)]("mwOld" -> oldCtx, "mwStartIdx" -> Cast(partStart, TInt32), "mwStopIdx" -> Cast(partStop, TInt32)))
    }}(newCtx => GetField(newCtx, "mwOld"))

    // Now label each row with its idx.
    val perRowIdxId = genUID()
    val partsZippedWithIdx = zippedWithStarts.mapPartitionWithContext { (part, ctx) =>
      zip2(part, rangeIR(GetField(ctx, "mwStartIdx"), GetField(ctx, "mwStopIdx")), ArrayZipBehavior.AssertSameLength) { (partRow, idx) =>
        insertIR(partRow, (perRowIdxId, idx))
      }
    }

    // Two steps, make a partitioner that works currently based on row_idx splits, then resplit accordingly.
    val inputRowIntervals = inputPartStarts.zip(inputPartStops).map{ case (intervalStart, intervalEnd) =>
      Interval(Row(intervalStart.toInt), Row(intervalEnd.toInt), true, false)
    }
    val rowIdxPartitioner = RVDPartitioner.generate(TStruct((perRowIdxId, TInt32)), inputRowIntervals)

    val keyedByRowIdx = partsZippedWithIdx.changePartitionerNoRepartition(rowIdxPartitioner)

    // Now create a partitioner that makes appropriately sized blocks
    val desiredRowStarts = (0 until numBlockRows).map(_ * blockSize)
    val desiredRowStops = desiredRowStarts.drop(1) :+ numRows.toInt
    val desiredRowIntervals = desiredRowStarts.zip(desiredRowStops).map{
      case (intervalStart, intervalEnd) =>  Interval(Row(intervalStart), Row(intervalEnd), true, false)
    }

    val blockSizeGroupsPartitioner = RVDPartitioner.generate(TStruct((perRowIdxId, TInt32)), desiredRowIntervals)
    val rowsInBlockSizeGroups: TableStage = keyedByRowIdx.repartitionNoShuffle(blockSizeGroupsPartitioner)

    def createBlockMakingContexts(tablePartsStreamIR: IR): IR = {
      flatten(zip2(tablePartsStreamIR, rangeIR(numBlockRows), ArrayZipBehavior.AssertSameLength) { case (tableSinglePartCtx, blockColIdx)  =>
        mapIR(rangeIR(I32(numBlockCols))){ blockColIdx =>
          MakeStruct(Seq("oldTableCtx" -> tableSinglePartCtx, "blockStart" -> (blockColIdx * I32(blockSize)),
            "blockSize" -> If(blockColIdx ceq I32(numBlockCols - 1), I32(lastBlockNumCols), I32(blockSize)),
            "blockColIdx" -> blockColIdx,
            "blockRowIdx" -> blockColIdx))
        }
      })
    }

    val tableOfNDArrays = rowsInBlockSizeGroups.mapContexts(createBlockMakingContexts)(ir => GetField(ir, "oldTableCtx")).mapPartitionWithContext{ (partIr, ctxRef) =>
      bindIR(GetField(ctxRef, "blockStart")){ blockStartRef =>
        val numColsOfBlock = GetField(ctxRef, "blockSize")
        val arrayOfSlicesAndIndices = ToArray(mapIR(partIr) { singleRow =>
          val mappedSlice = ToArray(mapIR(ToStream(sliceArrayIR(GetField(singleRow, entriesFieldName), blockStartRef, blockStartRef + numColsOfBlock)))(entriesStructRef =>
            GetField(entriesStructRef, entryField)
          ))
          MakeStruct(Seq(
            perRowIdxId -> GetField(singleRow, perRowIdxId),
            "rowOfData" -> mappedSlice
          ))
        })
        bindIR(arrayOfSlicesAndIndices){ arrayOfSlicesAndIndicesRef =>
          val idxOfResult = GetField(ArrayRef(arrayOfSlicesAndIndicesRef, I32(0)), perRowIdxId)
          val ndarrayData = ToArray(flatMapIR(ToStream(arrayOfSlicesAndIndicesRef)){idxAndSlice =>
            ToStream(GetField(idxAndSlice, "rowOfData"))
          })
          val numRowsOfBlock = ArrayLen(arrayOfSlicesAndIndicesRef)
          val shape = maketuple(Cast(numRowsOfBlock, TInt64), Cast(numColsOfBlock, TInt64))
          val ndarray = MakeNDArray(ndarrayData, shape, True(), ErrorIDs.NO_ERROR)
          MakeStream(Seq(MakeStruct(Seq(
            perRowIdxId -> idxOfResult,
            "blockRowIdx" -> GetField(ctxRef, "blockRowIdx"),
            "blockColIdx" -> GetField(ctxRef, "blockColIdx"),
            "ndBlock" -> ndarray))),
            TStream(TStruct(perRowIdxId -> TInt32, "blockRowIdx" -> TInt32, "blockColIdx" -> TInt32, "ndBlock" -> ndarray.typ)))
        }
      }
    }

    val elementType = tm.entryType.fieldType(entryField)
    val etype = EBlockMatrixNDArray(EType.fromTypeAndAnalysis(elementType, rm.entryType.field(entryField)), encodeRowMajor = true, required = true)
    val spec = TypedCodecSpec(etype, TNDArray(tm.entryType.fieldType(entryField), Nat(2)), BlockMatrix.bufferSpec)

    val pathsWithColMajorIndices = tableOfNDArrays.mapCollect(relationalLetsAbove) { partition =>
     ToArray(mapIR(partition) { singleNDArrayTuple =>
       bindIR(GetField(singleNDArrayTuple, "blockRowIdx") + (GetField(singleNDArrayTuple, "blockColIdx") * numBlockRows)) { colMajorIndex =>
         val blockPath =
           Str(s"$path/parts/part-") +
             invoke("str", TString, colMajorIndex) + Str("-") + UUID4()
         maketuple(colMajorIndex, WriteValue(GetField(singleNDArrayTuple, "ndBlock"), blockPath, spec))
       }
      })
    }
    val flatPathsAndIndices = flatMapIR(ToStream(pathsWithColMajorIndices))(ToStream(_))
    val sortedColMajorPairs = sortIR(flatPathsAndIndices){case (l, r) => ApplyComparisonOp(LT(TInt32), GetTupleElement(l, 0), GetTupleElement(r, 0))}
    val flatPaths = ToArray(mapIR(ToStream(sortedColMajorPairs))(GetTupleElement(_, 1)))
    val bmt = BlockMatrixType(elementType, IndexedSeq(numRows, numCols), numRows==1, blockSize, BlockMatrixSparsity.dense)
    RelationalWriter.scoped(path, overwrite, None)(WriteMetadata(flatPaths, BlockMatrixNativeMetadataWriter(path, false, bmt)))
  }
}

object MatrixNativeMultiWriter {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(List(classOf[MatrixNativeMultiWriter]), typeHintFieldName = "name")
  }
}

case class MatrixNativeMultiWriter(
  paths: IndexedSeq[String],
  overwrite: Boolean = false,
  stageLocally: Boolean = false,
  codecSpecJSONStr: String = null
) {
  val bufferSpec: BufferSpec = BufferSpec.parseOrDefault(codecSpecJSONStr)
  def apply(ctx: ExecuteContext, mvs: IndexedSeq[MatrixValue]): Unit = MatrixValue.writeMultiple(ctx, mvs, paths, overwrite, stageLocally, bufferSpec)
}
