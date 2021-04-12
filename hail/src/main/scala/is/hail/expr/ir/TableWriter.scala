package is.hail.expr.ir

import java.io.OutputStream
import is.hail.GenericIndexedSeqSerializer
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.lowering.{LowererUnsupportedOperation, TableStage}
import is.hail.expr.ir.streams.StreamProducer
import is.hail.io.fs.FS
import is.hail.io.index.StagedIndexWriter
import is.hail.io.{AbstractTypedCodecSpec, BufferSpec, OutputBuffer, TypedCodecSpec}
import is.hail.rvd.{AbstractRVDSpec, IndexSpec, RVDPartitioner, RVDSpecMaker}
import is.hail.types.encoded.EType
import is.hail.types.physical.{PCanonicalBaseStruct, PCanonicalString, PCanonicalStruct, PCode, PIndexableCode, PInt64, PStream, PStringCode, PStruct, PType}
import is.hail.types.virtual._
import is.hail.types.{RTable, TableType}
import is.hail.utils._
import is.hail.utils.richUtils.ByteTrackingOutputStream
import is.hail.variant.ReferenceGenome
import org.json4s.{DefaultFormats, Formats, ShortTypeHints}

object TableWriter {
  implicit val formats: Formats = new DefaultFormats()  {
    override val typeHints = ShortTypeHints(
      List(classOf[TableNativeWriter], classOf[TableTextWriter]), typeHintFieldName = "name")
  }
}

abstract class TableWriter {
  def path: String
  def apply(ctx: ExecuteContext, mv: TableValue): Unit
  def lower(ctx: ExecuteContext, ts: TableStage, t: TableIR, r: RTable, relationalLetsAbove: Map[String, IR]): IR =
    throw new LowererUnsupportedOperation(s"${ this.getClass } does not have defined lowering!")
}

case class TableNativeWriter(
  path: String,
  overwrite: Boolean = true,
  stageLocally: Boolean = false,
  codecSpecJSONStr: String = null
) extends TableWriter {

  override def lower(ctx: ExecuteContext, ts: TableStage, t: TableIR, r: RTable, relationalLetsAbove: Map[String, IR]): IR = {
    val bufferSpec: BufferSpec = BufferSpec.parseOrDefault(codecSpecJSONStr)
    val rowSpec = TypedCodecSpec(EType.fromTypeAndAnalysis(t.typ.rowType, r.rowType), t.typ.rowType, bufferSpec)
    val globalSpec = TypedCodecSpec(EType.fromTypeAndAnalysis(t.typ.globalType, r.globalType), t.typ.globalType, bufferSpec)

    // write out partitioner key, which may be stricter than table key
    val partitioner = ts.partitioner
    val pKey: PStruct = coerce[PStruct](rowSpec.decodedPType(partitioner.kType))
    val rowWriter = PartitionNativeWriter(rowSpec, s"$path/rows/parts/", Some(s"$path/index/" -> pKey), if (stageLocally) Some(ctx.localTmpdir) else None)
    val globalWriter = PartitionNativeWriter(globalSpec, s"$path/globals/parts/", None, None)

    ts.mapContexts { oldCtx =>
      val d = digitsNeeded(ts.numPartitions)
      val partFiles = Array.tabulate(ts.numPartitions)(i => Str(s"${ partFile(d, i) }-"))

      zip2(oldCtx, MakeStream(partFiles, TStream(TString)), ArrayZipBehavior.AssertSameLength) { (ctxElt, pf) =>
        MakeStruct(FastSeq(
          "oldCtx" -> ctxElt,
          "writeCtx" -> pf))
      }
    }(GetField(_, "oldCtx")).mapCollectWithContextsAndGlobals(relationalLetsAbove) { (rows, ctxRef) =>
      val file = GetField(ctxRef, "writeCtx")
      WritePartition(rows, file + UUID4(), rowWriter)
    } { (parts, globals) =>
      val writeGlobals = WritePartition(MakeStream(FastSeq(globals), TStream(globals.typ)),
        Str(partFile(1, 0)), globalWriter)

      RelationalWriter.scoped(path, overwrite, Some(t.typ))(
        bindIR(parts) { fileAndCount =>
          Begin(FastIndexedSeq(
            WriteMetadata(MakeArray(GetField(writeGlobals, "filePath")),
              RVDSpecWriter(s"$path/globals", RVDSpecMaker(globalSpec, RVDPartitioner.unkeyed(1)))),
            WriteMetadata(ToArray(mapIR(ToStream(fileAndCount)) { fc => GetField(fc, "filePath") }),
              RVDSpecWriter(s"$path/rows", RVDSpecMaker(rowSpec, partitioner, IndexSpec.emptyAnnotation("../index", coerce[PStruct](pKey))))),
            WriteMetadata(ToArray(mapIR(ToStream(fileAndCount)) { fc => GetField(fc, "partitionCounts") }),
              TableSpecWriter(path, t.typ, "rows", "globals", "references", log = true))))
        })
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
    val Array(globalFileData) = AbstractRVDSpec.writeSingle(ctx, globalsPath, tv.globals.t, bufferSpec, Array(tv.globals.javaValue))

    val codecSpec = TypedCodecSpec(tv.rvd.rowPType, bufferSpec)
    val fileData = tv.rvd.write(ctx, path + "/rows", "../index", stageLocally, codecSpec)
    val partitionCounts = fileData.map(_.rowsWritten)

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

    val partitionBytesWritten = fileData.map(_.bytesWritten)
    val totalRowsBytes = partitionBytesWritten.sum
    val globalBytesWritten = globalFileData.bytesWritten
    val totalBytesWritten: Long = totalRowsBytes + globalBytesWritten
    val (smallestStr, largestStr) = if (fileData.isEmpty)
      ("N/A", "N/A")
    else {
      val smallestPartition = fileData.minBy(_.bytesWritten)
      val largestPartition = fileData.maxBy(_.bytesWritten)
      val smallestStr = s"${ smallestPartition.rowsWritten } rows (${ formatSpace(smallestPartition.bytesWritten) })"
      val largestStr = s"${ largestPartition.rowsWritten } rows (${ formatSpace(largestPartition.bytesWritten) })"
      (smallestStr, largestStr)
    }

    val nRows = partitionCounts.sum
    info(s"wrote table with $nRows ${ plural(nRows, "row") } " +
      s"in ${ partitionCounts.length } ${ plural(partitionCounts.length, "partition") } " +
      s"to $path" +
      s"\n    Total size: ${ formatSpace(totalBytesWritten) }" +
      s"\n    * Rows: ${ formatSpace(totalRowsBytes) }" +
      s"\n    * Globals: ${ formatSpace(globalBytesWritten) }" +
      s"\n    * Smallest partition: $smallestStr" +
      s"\n    * Largest partition:  $largestStr")
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
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    stream: StreamProducer,
    context: EmitCode,
    region: Value[Region]): IEmitCode = {

    val mb = cb.emb

    val keyType = ifIndexed { index.get._2 }
    val indexWriter = ifIndexed { StagedIndexWriter.withDefaults(keyType, mb.ecb) }

    context.toI(cb).map(cb) { ctxCode: PCode =>
      val result = mb.newLocal[Long]("write_result")

      val filename = mb.newLocal[String]("filename")
      val os = mb.newLocal[ByteTrackingOutputStream]("write_os")
      val ob = mb.newLocal[OutputBuffer]("write_ob")
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
            },
              ob.invoke[Long]("indexOffset"),
              IEmitCode.present(cb, PCode(+PCanonicalStruct(), 0L)))
          }
          cb += ob.writeByte(1.asInstanceOf[Byte])

          spec.encodedType.buildEncoder(row.st, cb.emb.ecb)
            .apply(cb, row, ob)

          cb.assign(n, n + 1L)
      }

      PCode(pResultType, EmitCodeBuilder.scopedCode(mb) { cb: EmitCodeBuilder =>
        val pctx = ctxCode.memoize(cb, "context")
        cb.assign(filename, pctx.asString.loadString())
        if (hasIndex) {
          val indexFile = cb.newLocal[String]("indexFile")
          cb.assign(indexFile, const(index.get._1).concat(filename).concat(".idx"))
          indexWriter.init(cb, indexFile)
        }
        cb.assign(filename, const(partPrefix).concat(filename))
        cb.assign(os, Code.newInstance[ByteTrackingOutputStream, OutputStream](mb.create(filename)))
        cb.assign(ob, spec.buildCodeOutputBuffer(Code.checkcast[OutputStream](os)))
        cb.assign(n, 0L)

        stream.memoryManagedConsume(region, cb) { cb =>
          writeFile(cb, stream.element)
        }

        cb += ob.writeByte(0.asInstanceOf[Byte])
        cb.assign(result, pResultType.allocate(region))
        if (hasIndex)
          indexWriter.close(cb)
        cb += ob.flush()
        cb += os.invoke[Unit]("close")
        filenameType.storeAtAddress(cb, pResultType.fieldOffset(result, "filePath"), region, pctx, false)
        cb += Region.storeLong(pResultType.fieldOffset(result, "partitionCounts"), n)
        result.get
      })
    }
  }
}

case class RVDSpecWriter(path: String, spec: RVDSpecMaker) extends MetadataWriter {
  def annotationType: Type = TArray(TString)
  def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region]): Unit = {
    cb += cb.emb.getFS.invoke[String, Unit]("mkDir", path)
    val pc = writeAnnotations.get(cb, "write annotations can't be missing!").asInstanceOf[PIndexableCode]
    val a = pc.memoize(cb, "filePaths")
    val partFiles = cb.newLocal[Array[String]]("partFiles")
    val n = cb.newLocal[Int]("n", a.loadLength())
    val i = cb.newLocal[Int]("i", 0)
    cb.assign(partFiles, Code.newArray[String](n))
    cb.whileLoop(i < n, {
      val s = a.loadElement(cb, i).get(cb, "file name can't be missing!").asInstanceOf[PStringCode]
      cb += partFiles.update(i, s.loadString())
      cb.assign(i, i + 1)
    })
    cb += cb.emb.getObject(spec)
      .invoke[Array[String], AbstractRVDSpec]("apply", partFiles)
      .invoke[FS, String, Unit]("write", cb.emb.getFS, path)
  }
}

class TableSpecHelper(path: String, rowRelPath: String, globalRelPath: String, refRelPath: String, typ: TableType, log: Boolean) {
  def write(fs: FS, partCounts: Array[Long]): Unit = {
    val spec = TableSpecParameters(
      FileFormat.version.rep,
      is.hail.HAIL_PRETTY_VERSION,
      refRelPath,
      typ,
      Map("globals" -> RVDComponentSpec(globalRelPath),
        "rows" -> RVDComponentSpec(rowRelPath),
        "partition_counts" -> PartitionCountsComponentSpec(partCounts)))

    spec.write(fs, path)

    val nRows = partCounts.sum
    if (log) info(s"wrote table with $nRows ${ plural(nRows, "row") } " +
      s"in ${ partCounts.length } ${ plural(partCounts.length, "partition") } " +
      s"to $path")
  }
}

case class TableSpecWriter(path: String, typ: TableType, rowRelPath: String, globalRelPath: String, refRelPath: String, log: Boolean) extends MetadataWriter {
  def annotationType: Type = TArray(TInt64)

  def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region]): Unit = {
    cb += cb.emb.getFS.invoke[String, Unit]("mkDir", path)
    val pc = writeAnnotations.get(cb, "write annotations can't be missing!").asInstanceOf[PIndexableCode]
    val partCounts = cb.newLocal[Array[Long]]("partCounts")
    val a = pc.memoize(cb, "writePartCounts")

    val n = cb.newLocal[Int]("n", a.loadLength())
    val i = cb.newLocal[Int]("i", 0)
    cb.assign(partCounts, Code.newArray[Long](n))
    cb.whileLoop(i < n, {
      val count = a.loadElement(cb, i).get(cb, "part count can't be missing!")
      cb += partCounts.update(i, count.asLong.longCode(cb))
      cb.assign(i, i + 1)
    })
    cb += cb.emb.getObject(new TableSpecHelper(path, rowRelPath, globalRelPath, refRelPath, typ, log))
      .invoke[FS, Array[Long], Unit]("write", cb.emb.getFS, partCounts)
  }
}

object RelationalWriter {
  def scoped(path: String, overwrite: Boolean, refs: Option[TableType])(write: IR): IR = WriteMetadata(
    write, RelationalWriter(path, overwrite, refs.map(typ => "references" -> (ReferenceGenome.getReferences(typ.rowType) ++ ReferenceGenome.getReferences(typ.globalType)))))
}

case class RelationalWriter(path: String, overwrite: Boolean, maybeRefs: Option[(String, Set[ReferenceGenome])]) extends MetadataWriter {
  def annotationType: Type = TVoid

  def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region]): Unit = {
    if (overwrite)
      cb += cb.emb.getFS.invoke[String, Boolean, Unit]("delete", path, true)
    else
      cb.ifx(cb.emb.getFS.invoke[String, Boolean]("exists", path), cb._fatal(s"file already exists: $path"))
    cb += cb.emb.getFS.invoke[String, Unit]("mkDir", path)

    maybeRefs.foreach { case (refRelPath, refs) =>
      cb += cb.emb.getFS.invoke[String, Unit]("mkDir", s"$path/$refRelPath")
      refs.foreach { rg =>
        cb += Code.invokeScalaObject3[FS, String, ReferenceGenome, Unit](ReferenceGenome.getClass, "writeReference", cb.emb.getFS, path, cb.emb.getReferenceGenome(rg))
      }
    }

    writeAnnotations.consume(cb, {}, { pc => cb += pc.tcode[Unit] })

    cb += Code.invokeScalaObject2[FS, String, Unit](Class.forName("is.hail.utils.package$"), "writeNativeFileReadMe", cb.emb.getFS, path)
    cb += cb.emb.create(s"$path/_SUCCESS").invoke[Unit]("close")
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