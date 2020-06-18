package is.hail.expr.ir

import java.io.OutputStream

import is.hail.GenericIndexedSeqSerializer
import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.EmitStream.SizedStream
import is.hail.expr.ir.lowering.{LowererUnsupportedOperation, TableStage}
import is.hail.io.fs.FS
import is.hail.io.index.StagedIndexWriter
import is.hail.types.virtual._

import is.hail.io.{AbstractTypedCodecSpec, BufferSpec, OutputBuffer, TypedCodecSpec}
import is.hail.rvd.{AbstractRVDSpec, IndexSpec, RVDPartitioner, RVDSpecMaker}
import is.hail.types.{RTable, TableType}
import is.hail.types.encoded.EType
import is.hail.types.physical.{PArray, PCanonicalString, PCanonicalStruct, PCode, PInt64, PStream, PString, PStruct, PType}
import is.hail.utils._
import is.hail.utils.richUtils.ByteTrackingOutputStream
import is.hail.variant.ReferenceGenome
import org.json4s.{DefaultFormats, Formats, ShortTypeHints}

object TableWriter {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(
      List(classOf[TableNativeWriter], classOf[TableTextWriter]))
    override val typeHintFieldName = "name"
  }
}

abstract class TableWriter {
  def path: String
  def apply(ctx: ExecuteContext, mv: TableValue): Unit
  def lower(ctx: ExecuteContext, ts: TableStage, t: TableIR, r: RTable): IR =
    throw new LowererUnsupportedOperation(s"${ this.getClass } does not have defined lowering!")
}

case class TableNativeWriter(
  path: String,
  overwrite: Boolean = true,
  stageLocally: Boolean = false,
  codecSpecJSONStr: String = null
) extends TableWriter {

  override def lower(ctx: ExecuteContext, ts: TableStage, t: TableIR, r: RTable): IR = {
    val bufferSpec: BufferSpec = BufferSpec.parseOrDefault(codecSpecJSONStr)
    val rowSpec = TypedCodecSpec(EType.fromTypeAndAnalysis(t.typ.rowType, r.rowType), t.typ.rowType, bufferSpec)
    val globalSpec = TypedCodecSpec(EType.fromTypeAndAnalysis(t.typ.globalType, r.globalType), t.typ.globalType, bufferSpec)

    // write out partitioner key, which may be stricter than table key
    val partitioner = ts.partitioner
    val pKey: PStruct = coerce[PStruct](rowSpec.decodedPType(partitioner.kType))
    val rowWriter = PartitionNativeWriter(rowSpec, s"$path/rows/parts/", Some(s"$path/index/parts/" -> pKey), if (stageLocally) Some(ctx.localTmpdir) else None)
    val globalWriter = PartitionNativeWriter(globalSpec, s"$path/globals/parts/", None, None)
    val metadataWriter = MetadataNativeWriter(path, overwrite,
      RVDSpecMaker(rowSpec, partitioner, IndexSpec.emptyAnnotation("../index", coerce[PStruct](pKey))),
      RVDSpecMaker(globalSpec, RVDPartitioner.unkeyed(1)),
      t.typ)

    ts.mapContexts { oldCtx =>
      val d = digitsNeeded(ts.numPartitions)
      val partFiles = Array.tabulate(ts.numPartitions)(i => Str(s"${ partFile(d, i) }-"))

      zip2(oldCtx, MakeStream(partFiles, TStream(TString)), ArrayZipBehavior.AssertSameLength) { (ctxElt, pf) =>
        MakeStruct(FastSeq(
          "oldCtx" -> ctxElt,
          "writeCtx" -> pf))
      }
    }(GetField(_, "oldCtx")).mapCollectWithContextsAndGlobals { (rows, ctxRef) =>
      val file = GetField(ctxRef, "writeCtx")
      WritePartition(rows, file + UUID4(), rowWriter)
    } { (parts, globals) =>
      val writeGlobals = WritePartition(
        MakeStream(FastSeq(globals), TStream(globals.typ)),
        Str(partFile(1, 0)), globalWriter)

      WriteMetadata(makestruct(
        "global" -> GetField(writeGlobals, "filePath"),
        "partitions" -> parts), metadataWriter)
    }
  }

  def apply(ctx: ExecuteContext, tv: TableValue): Unit = {
    val bufferSpec: BufferSpec = BufferSpec.parseOrDefault(codecSpecJSONStr)
    assert(tv.typ.isCanonical)
    val fs = ctx.fs

    if (overwrite)
      fs.delete(path, recursive = true)
    else if (fs.exists(path))
      fatal(s"file already exists: $path")

    fs.mkDir(path)

    val globalsPath = path + "/globals"
    fs.mkDir(globalsPath)
    AbstractRVDSpec.writeSingle(ctx, globalsPath, tv.globals.t, bufferSpec, Array(tv.globals.javaValue))

    val codecSpec = TypedCodecSpec(tv.rvd.rowPType, bufferSpec)
    val partitionCounts = tv.rvd.write(ctx, path + "/rows", "../index", stageLocally, codecSpec)

    val referencesPath = path + "/references"
    fs.mkDir(referencesPath)
    ReferenceGenome.exportReferences(fs, referencesPath, tv.typ.rowType)
    ReferenceGenome.exportReferences(fs, referencesPath, tv.typ.globalType)

    val spec = TableSpecParameters(
      FileFormat.version.rep,
      is.hail.HAIL_PRETTY_VERSION,
      "references",
      tv.typ,
      Map("globals" -> RVDComponentSpec("globals"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    spec.write(fs, path)

    writeNativeFileReadMe(fs, path)

    using(fs.create(path + "/_SUCCESS"))(_ => ())

    val nRows = partitionCounts.sum
    info(s"wrote table with $nRows ${ plural(nRows, "row") } " +
      s"in ${ partitionCounts.length } ${ plural(partitionCounts.length, "partition") } " +
      s"to $path")
  }
}

case class PartitionNativeWriter(spec: AbstractTypedCodecSpec, partPrefix: String, index: Option[(String, PStruct)], localDir: Option[String]) extends PartitionWriter {
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
    context: EmitCode,
    eltType: PStruct,
    mb: EmitMethodBuilder[_],
    region: Value[Region],
    stream: SizedStream): EmitCode = {
    val enc = spec.buildEmitEncoderF(eltType, mb.ecb, typeInfo[Long]) //(Value[Region], Value[T], Value[OutputBuffer]) => Code[Unit]

    val keyType = ifIndexed { index.get._2 }
    val indexWriter = ifIndexed { StagedIndexWriter.withDefaults(keyType, mb.ecb) }

    context.map { ctxCode: PCode =>
      val ctx = mb.newLocal[Long]("ctx")
      val result = mb.newLocal[Long]("write_result")

      val filename = mb.newLocal[String]("filename")
      val os = mb.newLocal[ByteTrackingOutputStream]("write_os")
      val ob = mb.newLocal[OutputBuffer]("write_ob")
      val n = mb.newLocal[Long]("partition_count")

      def writeFile(codeRow: EmitCode): Code[Unit] = {
        val rowType = coerce[PStruct](codeRow.pt)
        EmitCodeBuilder.scopedVoid(mb) { cb =>
          codeRow.toI(cb).consume(cb,
            cb._fatal("row can't be missing"),
            { pc =>
              val row = pc.memoize(cb, "row")
              if (hasIndex) {
                val keyRVB = new StagedRegionValueBuilder(mb, keyType)
                indexWriter.add(cb, {
                  cb += keyRVB.start()
                  keyType.fields.foreach { f =>
                    cb += keyRVB.addIRIntermediate(f.typ)(Region.loadIRIntermediate(f.typ)(rowType.fieldOffset(coerce[Long](row.value), f.name)))
                    cb += keyRVB.advance()
                  }
                  IEmitCode.present(cb, PCode(keyType, keyRVB.offset))
                },
                  ob.invoke[Long]("indexOffset"),
                  {
                    val pcs = PCanonicalStruct(true)
                    IEmitCode.present(cb, PCode(pcs, pcs.allocate(region)))
                  }
                )
              }
              cb += ob.writeByte(1.asInstanceOf[Byte])
              cb += enc(region, coerce[Long](row.value), ob)
              cb.assign(n, n + 1L)
            })
          }
      }

      PCode(pResultType, EmitCodeBuilder.scopedCode(mb) { cb: EmitCodeBuilder =>
        cb.assign(ctx, ctxCode.tcode[Long])
        cb.assign(filename, pContextType.loadString(ctx))
        if (hasIndex) {
          val indexFile = cb.newLocal[String]("indexFile")
          cb.assign(indexFile, const(index.get._1).concat(filename))
          indexWriter.init(cb, indexFile)
        }
        cb.assign(filename, const(partPrefix).concat(filename))
        cb.assign(os, Code.newInstance[ByteTrackingOutputStream, OutputStream](mb.create(filename)))
        cb.assign(ob, spec.buildCodeOutputBuffer(Code.checkcast[OutputStream](os)))
        cb.assign(n, 0L)
        cb += stream.getStream.forEach(mb, writeFile)
        cb += ob.writeByte(0.asInstanceOf[Byte])
        cb.assign(result, pResultType.allocate(region))
        if (hasIndex)
          indexWriter.close(cb)
        cb += ob.flush()
        cb += os.invoke[Unit]("close")
        cb += Region.storeIRIntermediate(filenameType)(
          pResultType.fieldOffset(result, "filePath"), ctx)
        cb += Region.storeLong(pResultType.fieldOffset(result, "partitionCounts"), n)
        result.get
      })
    }
  }
}

class NativeTableMetadata(path: String, typ: TableType, rowsSpec: RVDSpecMaker, globalSpec: RVDSpecMaker) {
  val globalsPath = s"$path/globals"
  val rowsPath = s"$path/rows"

  def write(fs: FS, globalPath: String, partFiles: Array[String], partitionCounts: Array[Long]): Unit = {
    // globalMetadata
    globalSpec(Array(globalPath)).write(fs, globalsPath)

    // rowMetadata
    rowsSpec(partFiles).write(fs, rowsPath)

    val referencesPath = path + "/references"
    fs.mkDir(referencesPath)
    ReferenceGenome.exportReferences(fs, referencesPath, typ.rowType)
    ReferenceGenome.exportReferences(fs, referencesPath, typ.globalType)

    val spec = TableSpecParameters(
      FileFormat.version.rep,
      is.hail.HAIL_PRETTY_VERSION,
      "references",
      typ,
      Map("globals" -> RVDComponentSpec("globals"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    spec.write(fs, path)

    writeNativeFileReadMe(fs, path)

    using(fs.create(path + "/_SUCCESS"))(_ => ())

    val nRows = partitionCounts.sum
    info(s"wrote table with $nRows ${ plural(nRows, "row") } " +
      s"in ${ partitionCounts.length } ${ plural(partitionCounts.length, "partition") } " +
      s"to $path")
  }
}

case class MetadataNativeWriter(
  path: String,
  overwrite: Boolean,
  rowsSpec: RVDSpecMaker,
  globalsSpec: RVDSpecMaker,
  typ: TableType) extends MetadataWriter {
  def annotationType: Type = TStruct(
    "global" -> TString,
    "partitions" -> TArray(TStruct(
      "filePath" -> TString,
      "partitionCounts" -> TInt64)))

  val metadata = new NativeTableMetadata(path, typ, rowsSpec, globalsSpec)
  def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region]): Unit = {
    if (overwrite)
      cb += cb.emb.getFS.invoke[String, Boolean, Unit]("delete", path, true)
    else
      cb.ifx(cb.emb.getFS.invoke[String, Boolean]("exists", path), cb._fatal(s"file already exists: $path"))

    cb += cb.emb.getFS.invoke[String, Unit]("mkDir", path)
    cb += cb.emb.getFS.invoke[String, Unit]("mkDir", s"$path/globals")
    cb += cb.emb.getFS.invoke[String, Unit]("mkDir", s"$path/rows")

    writeAnnotations.consume(cb,
      { cb._fatal("write annotations can't be missing!") },
      { pc =>
        val v = pc.memoize(cb, "write_annotations")
        val aType = coerce[PStruct](v.pt)
        val partType = coerce[PArray](aType.fieldType("partitions"))
        val eltType = coerce[PStruct](partType.elementType)

        // global: (filename, pcount)
        // row: Array[(filename, pcount)]

        val globalFile = cb.newLocal[String]("global_file")
        val partFiles = cb.newLocal[Array[String]]("partFiles")
        val partCounts = cb.newLocal[Array[Long]]("partCounts")

        val aoff = cb.newLocal[Long]("aoff")
        val eltOff = cb.newLocal[Long]("eltOff")
        val n = cb.newLocal[Int]("n")
        val i = cb.newLocal[Int]("i")

        cb.assign(globalFile, coerce[PString](aType.fieldType("global"))
          .loadString(aType.loadField(coerce[Long](v.value), "global")))

        cb.assign(aoff, aType.loadField(coerce[Long](v.value), "partitions"))
        cb.assign(i, 0)
        cb.assign(n, partType.loadLength(aoff))
        cb.assign(partFiles, Code.newArray[String](n))
        cb.assign(partCounts, Code.newArray[Long](n))
        cb.whileLoop(i < n, {
          cb.assign(eltOff, partType.loadElement(aoff, n, i))
          cb += partFiles.update(i, coerce[PString](eltType.fieldType("filePath")).loadString(eltType.loadField(eltOff, "filePath")))
          cb += partCounts.update(i, Region.loadLong(eltType.fieldOffset(eltOff, "partitionCounts")))
          cb.assign(i, i + 1)
        })
        cb += cb.emb.getObject(metadata).invoke[FS, String, Array[String], Array[Long], Unit]("write", cb.emb.getFS, globalFile, partFiles, partCounts)
      })
  }
}

case class TableTextWriter(
  path: String,
  typesFile: String = null,
  header: Boolean = true,
  exportType: String = ExportType.CONCATENATED,
  delimiter: String
) extends TableWriter {
  def apply(ctx: ExecuteContext, tv: TableValue): Unit = tv.export(ctx, path, typesFile, header, exportType, delimiter)
}

object WrappedMatrixNativeMultiWriter {
  implicit val formats: Formats = MatrixNativeMultiWriter.formats +
    ShortTypeHints(List(classOf[WrappedMatrixNativeMultiWriter])) +
    GenericIndexedSeqSerializer
}

case class WrappedMatrixNativeMultiWriter(
  writer: MatrixNativeMultiWriter,
  colKey: IndexedSeq[String]
) {
  def apply(ctx: ExecuteContext, mvs: IndexedSeq[TableValue]): Unit = writer.apply(
    ctx, mvs.map(_.toMatrixValue(colKey)))
}