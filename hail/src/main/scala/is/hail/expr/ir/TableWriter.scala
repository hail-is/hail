package is.hail.expr.ir

import java.io.OutputStream
import scala.language.existentials
import is.hail.GenericIndexedSeqSerializer
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.TableAnnotationImpex
import is.hail.expr.ir.functions.StringFunctions
import is.hail.expr.ir.lowering.{LowererUnsupportedOperation, TableStage}
import is.hail.expr.ir.streams.StreamProducer
import is.hail.io.fs.FS
import is.hail.io.index.StagedIndexWriter
import is.hail.io.{AbstractTypedCodecSpec, BufferSpec, OutputBuffer, TypedCodecSpec}
import is.hail.rvd.{AbstractRVDSpec, IndexSpec, RVDPartitioner, RVDSpecMaker}
import is.hail.types.encoded.EType
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SContainer, SStringValue, SVoidValue}
import is.hail.types.physical._
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.concrete.{SStackStruct, SJavaArrayString, SJavaArrayStringValue}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SBooleanValue, SInt64Value}
import is.hail.types.virtual._
import is.hail.types.{RIterable, RStruct, RTable, TableType, TypeWithRequiredness}
import is.hail.utils._
import is.hail.utils.richUtils.ByteTrackingOutputStream
import is.hail.variant.ReferenceGenome
import org.json4s.JsonAST.JString
import org.json4s.{DefaultFormats, Formats, JBool, JObject, ShortTypeHints}

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

  def canLowerEfficiently: Boolean = false
}

object TableNativeWriter {
  def lower(ctx: ExecuteContext, ts: TableStage, tt: TableType, path: String, overwrite: Boolean, stageLocally: Boolean,
    rowSpec: TypedCodecSpec, globalSpec: TypedCodecSpec, relationalLetsAbove: Map[String, IR]): IR = {
    // write out partitioner key, which may be stricter than table key
    val partitioner = ts.partitioner
    val pKey: PStruct = coerce[PStruct](rowSpec.decodedPType(partitioner.kType))
    val rowWriter = PartitionNativeWriter(rowSpec, pKey.fieldNames, s"$path/rows/parts/", Some(s"$path/index/" -> pKey), if (stageLocally) Some(ctx.localTmpdir) else None)
    val globalWriter = PartitionNativeWriter(globalSpec, IndexedSeq(), s"$path/globals/parts/", None, None)

    RelationalWriter.scoped(path, overwrite, Some(tt))(
      ts.mapContexts { oldCtx =>
        val d = digitsNeeded(ts.numPartitions)
        val partFiles = Literal(TArray(TString), Array.tabulate(ts.numPartitions)(i => s"${ partFile(d, i) }-").toFastIndexedSeq)

        zip2(oldCtx, ToStream(partFiles), ArrayZipBehavior.AssertSameLength) { (ctxElt, pf) =>
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

        bindIR(parts) { fileCountAndDistinct =>
          Begin(FastIndexedSeq(
            WriteMetadata(MakeArray(GetField(writeGlobals, "filePath")),
              RVDSpecWriter(s"$path/globals", RVDSpecMaker(globalSpec, RVDPartitioner.unkeyed(1)))),
            WriteMetadata(ToArray(mapIR(ToStream(fileCountAndDistinct)) { fc => GetField(fc, "filePath") }),
              RVDSpecWriter(s"$path/rows", RVDSpecMaker(rowSpec, partitioner, IndexSpec.emptyAnnotation("../index", coerce[PStruct](pKey))))),
            WriteMetadata(ToArray(mapIR(ToStream(fileCountAndDistinct)) { fc =>
              SelectFields(fc, Seq("partitionCounts", "distinctlyKeyed", "firstKey", "lastKey"))
            }),
              TableSpecWriter(path, tt, "rows", "globals", "references", log = true))))
        }
      }
    )
  }
}

case class TableNativeWriter(
  path: String,
  overwrite: Boolean = true,
  stageLocally: Boolean = false,
  codecSpecJSONStr: String = null
) extends TableWriter {

  override def canLowerEfficiently: Boolean = !stageLocally

  override def lower(ctx: ExecuteContext, ts: TableStage, t: TableIR, r: RTable, relationalLetsAbove: Map[String, IR]): IR = {
    val bufferSpec: BufferSpec = BufferSpec.parseOrDefault(codecSpecJSONStr)
    val rowSpec = TypedCodecSpec(EType.fromTypeAndAnalysis(t.typ.rowType, r.rowType), t.typ.rowType, bufferSpec)
    val globalSpec = TypedCodecSpec(EType.fromTypeAndAnalysis(t.typ.globalType, r.globalType), t.typ.globalType, bufferSpec)

    TableNativeWriter.lower(ctx, ts, t.typ, path, overwrite, stageLocally, rowSpec, globalSpec, relationalLetsAbove)
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

case class PartitionNativeWriter(spec: AbstractTypedCodecSpec, keyFields: IndexedSeq[String], partPrefix: String, index: Option[(String, PStruct)], localDir: Option[String]) extends PartitionWriter {
  def stageLocally: Boolean = localDir.isDefined
  def hasIndex: Boolean = index.isDefined
  val filenameType = PCanonicalString(required = true)
  def pContextType = PCanonicalString()

  val keyType = spec.encodedVirtualType.asInstanceOf[TStruct].select(keyFields)._1

  def ctxType: Type = TString
  def returnType: Type = TStruct("filePath" -> TString, "partitionCounts" -> TInt64, "distinctlyKeyed" -> TBoolean, "firstKey" -> keyType, "lastKey" -> keyType)
  def unionTypeRequiredness(r: TypeWithRequiredness, ctxType: TypeWithRequiredness, streamType: RIterable): Unit = {
    val rs = r.asInstanceOf[RStruct]
    val rKeyType = streamType.elementType.asInstanceOf[RStruct].select(keyFields.toArray)
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

    val mb = cb.emb

    val indexKeyType = ifIndexed { index.get._2 }
    val indexWriter = ifIndexed { StagedIndexWriter.withDefaults(indexKeyType, mb.ecb) }

    context.toI(cb).map(cb) { case ctx: SStringValue =>
      val filename = mb.newLocal[String]("filename")
      val os = mb.newLocal[ByteTrackingOutputStream]("write_os")
      val ob = mb.newLocal[OutputBuffer]("write_ob")
      val n = mb.newLocal[Long]("partition_count")
      val distinctlyKeyed = mb.newLocal[Boolean]("distinctlyKeyed")
      cb.assign(distinctlyKeyed, !keyFields.isEmpty) // True until proven otherwise, if there's a key to care about at all.

      val keyEmitType = EmitType(spec.decodedPType(keyType).sType, false)

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
                indexKeyPType.fields.map{ f =>
                  EmitCode.fromI(cb.emb)(cb => row.loadField(cb, f.name))
                },
                deepCopy = true))
          },
            ob.invoke[Long]("indexOffset"),
            IEmitCode.present(cb, PCanonicalStruct().loadCheapSCode(cb, 0L)))
        }

        val key = SStackStruct.constructFromArgs(cb, stream.elementRegion, keyType, keyType.fields.map { f =>
          EmitCode.fromI(cb.emb)(cb => row.loadField(cb, f.name))
        }:_*)

        if (!keyFields.isEmpty) {
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

        cb += ob.writeByte(1.asInstanceOf[Byte])

        spec.encodedType.buildEncoder(row.st, cb.emb.ecb)
          .apply(cb, row, ob)

        cb.assign(n, n + 1L)
      }

      cb.assign(filename, ctx.loadString(cb))
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
      if (hasIndex)
        indexWriter.close(cb)
      cb += ob.flush()
      cb += os.invoke[Unit]("close")

      SStackStruct.constructFromArgs(cb, region, returnType.asInstanceOf[TBaseStruct],
        EmitCode.present(mb, ctx),
        EmitCode.present(mb, new SInt64Value(n)),
        EmitCode.present(mb, new SBooleanValue(distinctlyKeyed)),
        firstSeenSettable,
        lastSeenSettable
      )
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
    val a = writeAnnotations.get(cb, "write annotations can't be missing!").asIndexable
    val partFiles = cb.newLocal[Array[String]]("partFiles")
    val n = cb.newLocal[Int]("n", a.loadLength())
    val i = cb.newLocal[Int]("i", 0)
    cb.assign(partFiles, Code.newArray[String](n))
    cb.whileLoop(i < n, {
      val s = a.loadElement(cb, i).get(cb, "file name can't be missing!").asString
      cb += partFiles.update(i, s.loadString(cb))
      cb.assign(i, i + 1)
    })
    cb += cb.emb.getObject(spec)
      .invoke[Array[String], AbstractRVDSpec]("apply", partFiles)
      .invoke[FS, String, Unit]("write", cb.emb.getFS, path)
  }
}

class TableSpecHelper(path: String, rowRelPath: String, globalRelPath: String, refRelPath: String, typ: TableType, log: Boolean) extends Serializable {
  def write(fs: FS, partCounts: Array[Long], distinctlyKeyed: Boolean): Unit = {
    val spec = TableSpecParameters(
      FileFormat.version.rep,
      is.hail.HAIL_PRETTY_VERSION,
      refRelPath,
      typ,
      Map("globals" -> RVDComponentSpec(globalRelPath),
        "rows" -> RVDComponentSpec(rowRelPath),
        "partition_counts" -> PartitionCountsComponentSpec(partCounts),
        "properties" -> PropertiesSpec(JObject(
          "distinctlyKeyed" -> JBool(distinctlyKeyed)
        ))
      ))

    spec.write(fs, path)

    val nRows = partCounts.sum
    if (log) info(s"wrote table with $nRows ${ plural(nRows, "row") } " +
      s"in ${ partCounts.length } ${ plural(partCounts.length, "partition") } " +
      s"to $path")
  }
}

case class TableSpecWriter(path: String, typ: TableType, rowRelPath: String, globalRelPath: String, refRelPath: String, log: Boolean) extends MetadataWriter {
  def annotationType: Type = TArray(TStruct("partitionCounts" -> TInt64, "distinctlyKeyed" -> TBoolean, "firstKey" -> typ.keyType, "lastKey" -> typ.keyType))

  def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region]): Unit = {
    cb += cb.emb.getFS.invoke[String, Unit]("mkDir", path)

    val hasKey = !this.typ.keyType.fields.isEmpty

    val a = writeAnnotations.get(cb, "write annotations can't be missing!").asIndexable
    val partCounts = cb.newLocal[Array[Long]]("partCounts")

    val idxOfFirstKeyField = annotationType.asInstanceOf[TArray].elementType.asInstanceOf[TStruct].fieldIdx("firstKey")
    val keySType = a.st.elementType.asInstanceOf[SBaseStruct].fieldTypes(idxOfFirstKeyField)

    val lastSeenSettable = cb.emb.newEmitLocal(EmitType(keySType, false))
    cb.assign(lastSeenSettable, EmitCode.missing(cb.emb, keySType))
    val distinctlyKeyed = cb.newLocal[Boolean]("tsw_write_metadata_distinctlyKeyed", hasKey)

    val n = cb.newLocal[Int]("n", a.loadLength())
    val i = cb.newLocal[Int]("i", 0)
    cb.assign(partCounts, Code.newArray[Long](n))
    cb.whileLoop(i < n, {
      val curElement =  a.loadElement(cb, i).get(cb, "writeMetadata annotation can't be missing").asBaseStruct
      val count = curElement.asBaseStruct.loadField(cb, "partitionCounts").get(cb, "part count can't be missing!").asLong.value

      if (hasKey) {
        // Only nonempty partitions affect first, last, and distinctlyKeyed.
        cb.ifx(count cne 0L, {
          val curFirst = curElement.loadField(cb, "firstKey").get(cb, const("firstKey of curElement can't be missing, part size was ") concat count.toS)

          val comparator = NEQ(lastSeenSettable.emitType.virtualType).codeOrdering(cb.emb.ecb, lastSeenSettable.st, curFirst.st)
          val notEqualToLast = comparator(cb, lastSeenSettable, EmitValue.present(curFirst)).asInstanceOf[Value[Boolean]]

          val partWasDistinctlyKeyed = curElement.loadField(cb, "distinctlyKeyed").get(cb).asBoolean.value
          cb.assign(distinctlyKeyed, distinctlyKeyed && partWasDistinctlyKeyed && notEqualToLast)
          cb.assign(lastSeenSettable, curElement.loadField(cb, "lastKey"))
        })
      }

      cb += partCounts.update(i, count)
      cb.assign(i, i + 1)
    })
    cb += cb.emb.getObject(new TableSpecHelper(path, rowRelPath, globalRelPath, refRelPath, typ, log))
      .invoke[FS, Array[Long], Boolean, Unit]("write", cb.emb.getFS, partCounts, distinctlyKeyed)
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

    writeAnnotations.consume(cb, {}, { pc => assert(pc == SVoidValue) }) // PVoidCode.code is Code._empty

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

  override def canLowerEfficiently: Boolean = exportType != ExportType.PARALLEL_COMPOSABLE
  override def lower(ctx: ExecuteContext, ts: TableStage, t: TableIR, r: RTable, relationalLetsAbove: Map[String, IR]): IR = {
    require(exportType != ExportType.PARALLEL_COMPOSABLE)

    val ext = ctx.fs.getCodecExtension(path)

    val folder = if (exportType == ExportType.CONCATENATED)
      ctx.createTmpPath("write-table-concatenated")
    else
      path
    val lineWriter = TableTextPartitionWriter(ts.rowType, delimiter, writeHeader = exportType == ExportType.PARALLEL_HEADER_IN_SHARD)

    ts.mapContexts { oldCtx =>
      val d = digitsNeeded(ts.numPartitions)
      val partFiles = Literal(TArray(TString), Array.tabulate(ts.numPartitions)(i => s"$folder/${ partFile(d, i) }$ext").toFastIndexedSeq)

      zip2(oldCtx, ToStream(partFiles), ArrayZipBehavior.AssertSameLength) { (ctxElt, pf) =>
        MakeStruct(FastSeq(
          "oldCtx" -> ctxElt,
          "partFile" -> pf))
      }
    }(GetField(_, "oldCtx")).mapCollectWithContextsAndGlobals(relationalLetsAbove) { (rows, ctxRef) =>
      val file = GetField(ctxRef, "partFile")
      WritePartition(rows, file, lineWriter)
    } { (parts, _) =>
      val commit = TableTextFinalizer(path, ts.rowType, delimiter, header, exportType)
      Begin(FastIndexedSeq(WriteMetadata(parts, commit)))
    }
  }
}

case class TableTextPartitionWriter(rowType: TStruct, delimiter: String, writeHeader: Boolean) extends SimplePartitionWriter {
  lazy val headerStr = rowType.fields.map(_.name).mkString(delimiter)

  override def preConsume(cb: EmitCodeBuilder, os: Value[OutputStream]): Unit = if (writeHeader) {
    cb += os.invoke[Array[Byte], Unit]("write", const(headerStr).invoke[Array[Byte]]("getBytes"))
    cb += os.invoke[Int, Unit]("write", '\n')
  }

  def consumeElement(cb: EmitCodeBuilder, element: EmitCode, os: Value[OutputStream], region: Value[Region]): Unit = {
    require(element.st.virtualType == rowType)
    val delimBytes: Value[Array[Byte]] = cb.memoize(cb.emb.getObject(delimiter.getBytes))

    element.toI(cb).consume(cb, { cb._fatal("stream element can not be missing!") }, { case sv: SBaseStructValue =>
      // I hope we're buffering our writes correctly!
      (0 until sv.st.size).foreachBetween { i =>
        val f = sv.loadField(cb, i)
        val annotation = f.consumeCode[AnyRef](cb, Code._null[AnyRef],
          { sv => StringFunctions.svalueToJavaValue(cb, region, sv) })
        val str = Code.invokeScalaObject2[Any, Type, String](TableAnnotationImpex.getClass, "exportAnnotation",
          annotation, cb.emb.getType(f.st.virtualType))
        cb += os.invoke[Array[Byte], Unit]("write", str.invoke[Array[Byte]]("getBytes"))
      }(cb += os.invoke[Array[Byte], Unit]("write", delimBytes))
      cb += os.invoke[Int, Unit]("write", '\n')
    })
  }
}

case class TableTextFinalizer(outputPath: String, rowType: TStruct, delimiter: String,
    header: Boolean = true, exportType: String = ExportType.CONCATENATED) extends MetadataWriter {
  def annotationType: Type = TArray(TString)
  def writeMetadata(writeAnnotations: => IEmitCode, cb: EmitCodeBuilder, region: Value[Region]): Unit = {
    val ctx: ExecuteContext = cb.emb.ctx
    val ext = ctx.fs.getCodecExtension(outputPath)
    val partPaths = writeAnnotations.get(cb, "write annotations cannot be missing!")
    exportType match {
      case ExportType.CONCATENATED =>
        val files = partPaths.castTo(cb, region, SJavaArrayString(true), false)
        val jFiles = if (header) {
          val headerFilePath = ctx.createTmpPath("header", ext)
          val headerStr = rowType.fields.map(_.name).mkString(delimiter)
          val os = cb.memoize(cb.emb.create(const(headerFilePath)))
          cb += os.invoke[Array[Byte], Unit]("write", const(headerStr).invoke[Array[Byte]]("getBytes"))
          cb += os.invoke[Int, Unit]("write", '\n')
          cb += os.invoke[Unit]("close")

          val jFiles = files.asInstanceOf[SJavaArrayStringValue].array
          val allFiles = cb.memoize(Code.newArray[String](jFiles.length + 1))
          cb += (allFiles(0) = const(headerFilePath))
          cb += Code.invokeStatic5[System, Any, Int, Any, Int, Int, Unit](
            "arraycopy", jFiles /*src*/, 0 /*srcPos*/, allFiles /*dest*/, 1 /*destPos*/, jFiles.length /*len*/)
          allFiles
        } else {
          files.asInstanceOf[SJavaArrayStringValue].array
        }

        cb += cb.emb.getFS.invoke[Array[String], String, Unit]("concatenateFiles", jFiles, const(outputPath))

        val i = cb.newLocal[Int]("i")
        cb.forLoop(cb.assign(i, 0), i < jFiles.length, cb.assign(i, i + 1), {
          cb += cb.emb.getFS.invoke[String, Boolean, Unit]("delete", jFiles(i), const(false))
        })

      case ExportType.PARALLEL_HEADER_IN_SHARD =>
        cb += cb.emb.getFS.invoke[String, Unit]("touch", const(outputPath).concat("/_SUCCESS"))

      case ExportType.PARALLEL_SEPARATE_HEADER =>
        if (header) {
          val headerFilePath = s"$outputPath/header$ext"
          val headerStr = rowType.fields.map(_.name).mkString(delimiter)
          val os = cb.memoize(cb.emb.create(const(headerFilePath)))
          cb += os.invoke[Array[Byte], Unit]("write", const(headerStr).invoke[Array[Byte]]("getBytes"))
          cb += os.invoke[Int, Unit]("write", '\n')
          cb += os.invoke[Unit]("close")
        }

        cb += cb.emb.getFS.invoke[String, Unit]("touch", const(outputPath).concat("/_SUCCESS"))
    }
  }
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
