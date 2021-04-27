package is.hail.expr.ir

import java.io.OutputStream
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.functions.MatrixWriteBlockMatrix
import is.hail.expr.ir.lowering.{LowererUnsupportedOperation, TableStage}
import is.hail.expr.ir.streams.StreamProducer
import is.hail.io._
import is.hail.io.fs.FS
import is.hail.io.gen.{ExportBGEN, ExportGen}
import is.hail.io.index.StagedIndexWriter
import is.hail.io.plink.ExportPlink
import is.hail.io.vcf.ExportVCF
import is.hail.rvd.{RVDPartitioner, RVDSpecMaker}
import is.hail.types.encoded.{EBaseStruct, EType}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.{PBaseStructCode, PCanonicalBaseStruct, PCanonicalString, PCanonicalStruct, PCode, PIndexableValue, PInt64, PInt64Required, PStream, PStruct, PType}
import is.hail.types.virtual._
import is.hail.types.{MatrixType, RTable, TableType}
import is.hail.utils._
import is.hail.utils.richUtils.ByteTrackingOutputStream
import org.json4s.jackson.JsonMethods
import org.json4s.{DefaultFormats, Formats, ShortTypeHints}

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
}

abstract class MatrixWriter {
  def path: String
  def apply(ctx: ExecuteContext, mv: MatrixValue): Unit
  def lower(colsFieldName: String, entriesFieldName: String, colKey: IndexedSeq[String],
    ctx: ExecuteContext, ts: TableStage, t: TableIR, r: RTable, relationalLetsAbove: Map[String, IR]): IR =
    throw new LowererUnsupportedOperation(s"${ this.getClass } does not have defined lowering!")
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

    val emptyWriter = PartitionNativeWriter(emptySpec, s"$path/globals/globals/parts/", None, None)
    val globalWriter = PartitionNativeWriter(globalSpec, s"$path/globals/rows/parts/", None, None)
    val colWriter = PartitionNativeWriter(colSpec, s"$path/cols/rows/parts/", None, None)
    val rowWriter = SplitPartitionNativeWriter(
      rowSpec, s"$path/rows/rows/parts/",
      entrySpec, s"$path/entries/rows/parts/",
      Some(s"$path/index/" -> pKey), if (stageLocally) Some(ctx.localTmpdir) else None)

    lowered.mapContexts { oldCtx =>
      val d = digitsNeeded(lowered.numPartitions)
      val partFiles = Array.tabulate(lowered.numPartitions)(i => Str(s"${ partFile(d, i) }-"))

      zip2(oldCtx, MakeStream(partFiles, TStream(TString)), ArrayZipBehavior.AssertSameLength) { (ctxElt, pf) =>
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
      val colTableWriter = TableSpecWriter(s"$path/cols", tm.colsTableType, "rows", "../globals/rows", "../references", log = false)
      val rowTableWriter = TableSpecWriter(s"$path/rows", tm.rowsTableType, "rows", "../globals/rows", "../references", log = false)
      val entriesTableWriter = TableSpecWriter(s"$path/entries", TableType(tm.entriesRVType, FastIndexedSeq(), tm.globalType), "rows", "../globals/rows", "../references", log = false)

      val matrixWriter = MatrixSpecWriter(path, tm, "rows/rows", "globals/rows", "cols/rows", "entries/rows", "references", log = true)

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
                      WriteMetadata(MakeArray(I64(1)), globalTableWriter),
                      WriteMetadata(MakeArray(GetField(colInfo, "filePath")),
                        RVDSpecWriter(s"$path/cols/rows", RVDSpecMaker(colSpec, RVDPartitioner.unkeyed(1)))),
                      WriteMetadata(MakeArray(GetField(colInfo, "partitionCounts")), colTableWriter),
                      bindIR(ToArray(mapIR(ToStream(partInfo)) { fc => GetField(fc, "filePath") })) { files =>
                        Begin(FastIndexedSeq(
                          WriteMetadata(files, RVDSpecWriter(s"$path/rows/rows", RVDSpecMaker(rowSpec, lowered.partitioner))),
                          WriteMetadata(files, RVDSpecWriter(s"$path/entries/rows", RVDSpecMaker(entrySpec, RVDPartitioner.unkeyed(lowered.numPartitions))))))
                      },
                      bindIR(ToArray(mapIR(ToStream(partInfo)) { fc => GetField(fc, "partitionCounts") })) { counts =>
                        Begin(FastIndexedSeq(
                            WriteMetadata(counts, rowTableWriter),
                            WriteMetadata(counts, entriesTableWriter),
                            WriteMetadata(makestruct("cols" -> GetField(colInfo, "partitionCounts"), "rows" -> counts), matrixWriter)))
                      }))
                  }
                })))))
    }
  }
}

case class SplitPartitionNativeWriter(
  spec1: AbstractTypedCodecSpec, partPrefix1: String,
  spec2: AbstractTypedCodecSpec, partPrefix2: String,
  index: Option[(String, PStruct)], localDir: Option[String]) extends PartitionWriter {
  def stageLocally: Boolean = localDir.isDefined
  def hasIndex: Boolean = index.isDefined
  val filenameType = PCanonicalString(required = true)
  def pContextType = PCanonicalString()
  def pResultType: PCanonicalStruct =
    PCanonicalStruct(required=true, "filePath" -> filenameType, "partitionCounts" -> PInt64(required=true))

  def ctxType: Type = TString
  def returnType: Type = pResultType.virtualType
  def returnPType(ctxType: PType, streamType: PStream): PType = pResultType

  if (stageLocally)
    throw new LowererUnsupportedOperation("stageLocally option not yet implemented")
  def ifIndexed[T >: Null](obj: => T): T = if (hasIndex) obj else null

  def consumeStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    stream: StreamProducer,
    context: EmitCode,
    region: Value[Region]): IEmitCode = {
    val keyType = ifIndexed { index.get._2 }
    val iAnnotationType = PCanonicalStruct(required = true, "entries_offset" -> PInt64Required)
    val mb = cb.emb
    val indexWriter = ifIndexed { StagedIndexWriter.withDefaults(keyType, mb.ecb, annotationType = iAnnotationType) }


    context.toI(cb).map(cb) { ctxCode: PCode =>
      val result = mb.newLocal[Long]("write_result")
      val filename1 = mb.newLocal[String]("filename1")
      val os1 = mb.newLocal[ByteTrackingOutputStream]("write_os1")
      val ob1 = mb.newLocal[OutputBuffer]("write_ob1")
      val filename2 = mb.newLocal[String]("filename2")
      val os2 = mb.newLocal[ByteTrackingOutputStream]("write_os2")
      val ob2 = mb.newLocal[OutputBuffer]("write_ob2")
      val n = mb.newLocal[Long]("partition_count")


      def writeFile(cb: EmitCodeBuilder, codeRow: EmitCode): Unit = {
        val pc = codeRow.toI(cb).get(cb, "row can't be missing").asBaseStruct
        val row = pc.memoize(cb, "row")
        if (hasIndex) {
          indexWriter.add(cb, {
            IEmitCode.present(cb, keyType.asInstanceOf[PCanonicalBaseStruct]
              .constructFromFields(cb, stream.elementRegion,
                keyType.fields.map(f => EmitCode.fromI(cb.emb)(cb => row.loadField(cb, f.name).typecast[PCode])),
                deepCopy = false))
          }, ob1.invoke[Long]("indexOffset"), {
            IEmitCode.present(cb,
              iAnnotationType.constructFromFields(cb, stream.elementRegion,
                FastIndexedSeq(EmitCode.present(cb.emb, primitive(ob2.invoke[Long]("indexOffset")))),
                deepCopy = false))
          })
        }
        cb += ob1.writeByte(1.asInstanceOf[Byte])

        spec1.encodedType.buildEncoder(row.st, cb.emb.ecb)
          .apply(cb, row, ob1)

        cb += ob2.writeByte(1.asInstanceOf[Byte])

        spec2.encodedType.buildEncoder(row.st, cb.emb.ecb)
          .apply(cb, row, ob2)
        cb.assign(n, n + 1L)
      }

      PCode(pResultType, EmitCodeBuilder.scopedCode(mb) { cb: EmitCodeBuilder =>
        val pctx = ctxCode.memoize(cb, "context")
        cb.assign(filename1, pctx.asString.loadString())
        if (hasIndex) {
          val indexFile = cb.newLocal[String]("indexFile")
          cb.assign(indexFile, const(index.get._1).concat(filename1))
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
        cb.assign(result, pResultType.allocate(region))
        if (hasIndex)
          indexWriter.close(cb)
        cb += ob1.flush()
        cb += ob2.flush()
        cb += os1.invoke[Unit]("close")
        cb += os2.invoke[Unit]("close")
        filenameType.storeAtAddress(cb, pResultType.fieldOffset(result, "filePath"), region, pctx, false)
        cb += Region.storeLong(pResultType.fieldOffset(result, "partitionCounts"), n)
        result.get
      })
    }
  }
}

class MatrixSpecHelper(path: String, rowRelPath: String, globalRelPath: String, colRelPath: String, entryRelPath: String, refRelPath: String, typ: MatrixType, log: Boolean) {
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
    val pc = writeAnnotations.get(cb, "write annotations can't be missing!").asInstanceOf[PBaseStructCode]
    val partCounts = cb.newLocal[Array[Long]]("partCounts")
    val c = pc.memoize(cb, "matrixPartCounts")
    val a = c.loadField(cb, "rows").get(cb).memoize(cb, "rowCounts").asInstanceOf[PIndexableValue]

    val n = cb.newLocal[Int]("n", a.loadLength())
    val i = cb.newLocal[Int]("i", 0)
    cb.assign(partCounts, Code.newArray[Long](n))
    cb.whileLoop(i < n, {
      val count = a.loadElement(cb, i).get(cb, "part count can't be missing!").asPCode
      cb += partCounts.update(i, count.asInt64.longCode(cb))
      cb.assign(i, i + 1)
    })
    cb += cb.emb.getObject(new MatrixSpecHelper(path, rowRelPath, globalRelPath, colRelPath, entryRelPath, refRelPath, typ, log))
      .invoke[FS, Long, Array[Long], Unit]("write", cb.emb.getFS, c.loadField(cb, "cols").get(cb).asInt64.longCode(cb), partCounts)
  }
}

case class MatrixVCFWriter(
  path: String,
  append: Option[String] = None,
  exportType: String = ExportType.CONCATENATED,
  metadata: Option[VCFMetadata] = None,
  tabix: Boolean = false
) extends MatrixWriter {
  def apply(ctx: ExecuteContext, mv: MatrixValue): Unit = ExportVCF(ctx, mv, path, append, exportType, metadata, tabix)
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
}

object MatrixNativeMultiWriter {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(List(classOf[MatrixNativeMultiWriter]), typeHintFieldName = "name")
  }
}

case class MatrixNativeMultiWriter(
  prefix: String,
  overwrite: Boolean = false,
  stageLocally: Boolean = false
) {
  def apply(ctx: ExecuteContext, mvs: IndexedSeq[MatrixValue]): Unit = MatrixValue.writeMultiple(ctx, mvs, prefix, overwrite, stageLocally)
}
