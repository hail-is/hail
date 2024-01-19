package is.hail.expr.ir

import is.hail.GenericIndexedSeqSerializer
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.TableAnnotationImpex
import is.hail.expr.ir.functions.StringFunctions
import is.hail.expr.ir.lowering.{LowererUnsupportedOperation, TableStage}
import is.hail.expr.ir.streams.StreamProducer
import is.hail.io.{AbstractTypedCodecSpec, BufferSpec, OutputBuffer, TypedCodecSpec}
import is.hail.io.fs.FS
import is.hail.io.index.StagedIndexWriter
import is.hail.rvd.{AbstractRVDSpec, IndexSpec, RVDPartitioner, RVDSpecMaker}
import is.hail.types._
import is.hail.types.encoded.EType
import is.hail.types.physical._
import is.hail.types.physical.stypes.{EmitType, SCode, SValue}
import is.hail.types.physical.stypes.concrete.{
  SJavaArrayString, SJavaArrayStringValue, SStackStruct,
}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.{SBooleanValue, SInt64, SInt64Value}
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.utils.richUtils.ByteTrackingOutputStream
import is.hail.variant.ReferenceGenome

import scala.language.existentials

import java.io.{BufferedOutputStream, OutputStream}
import java.nio.file.{FileSystems, Path}
import java.util.UUID

import org.json4s.{DefaultFormats, Formats, JBool, JObject, ShortTypeHints}

object TableWriter {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(
      List(classOf[TableNativeFanoutWriter], classOf[TableNativeWriter], classOf[TableTextWriter]),
      typeHintFieldName = "name",
    )
  }
}

abstract class TableWriter {
  def path: String

  def apply(ctx: ExecuteContext, tv: TableValue): Unit = {
    val tableStage = TableValueIntermediate(tv).asTableStage(ctx)
    CompileAndEvaluate(ctx, lower(ctx, tableStage, RTable.fromTableStage(ctx, tableStage)))
  }

  def lower(ctx: ExecuteContext, ts: TableStage, r: RTable): IR =
    throw new LowererUnsupportedOperation(s"${this.getClass} does not have defined lowering!")

  def canLowerEfficiently: Boolean =
    true
}

object TableNativeWriter {
  def lower(
    ctx: ExecuteContext,
    ts: TableStage,
    path: String,
    overwrite: Boolean,
    stageLocally: Boolean,
    rowSpec: TypedCodecSpec,
    globalSpec: TypedCodecSpec,
  ): IR = {
    // write out partitioner key, which may be stricter than table key
    val partitioner = ts.partitioner
    val pKey: PStruct = tcoerce[PStruct](rowSpec.decodedPType(partitioner.kType))
    val rowWriter = PartitionNativeWriter(
      rowSpec,
      pKey.fieldNames,
      s"$path/rows/parts/",
      Some(s"$path/index/" -> pKey),
      if (stageLocally) Some(FileSystems.getDefault.getPath(
        ctx.localTmpdir,
        s"hail_staging_tmp_${UUID.randomUUID()}",
        "rows",
        "parts",
      ))
      else None,
    )

    val globalWriter =
      PartitionNativeWriter(globalSpec, IndexedSeq(), s"$path/globals/parts/", None, None)
    RelationalWriter.scoped(path, overwrite, Some(ts.tableType))(
      ts.mapContexts { oldCtx =>
        val d = digitsNeeded(ts.numPartitions)
        val partFiles = Literal(
          TArray(TString),
          Array.tabulate(ts.numPartitions)(i => s"${partFile(d, i)}-").toFastSeq,
        )

        zip2(oldCtx, ToStream(partFiles), ArrayZipBehavior.AssertSameLength) { (ctxElt, pf) =>
          MakeStruct(FastSeq(
            "oldCtx" -> ctxElt,
            "writeCtx" -> pf,
          ))
        }
      }(GetField(_, "oldCtx")).mapCollectWithContextsAndGlobals("table_native_writer") {
        (rows, ctxRef) =>
          val file = GetField(ctxRef, "writeCtx")
          WritePartition(rows, file + UUID4(), rowWriter)
      } { (parts, globals) =>
        val writeGlobals = WritePartition(
          MakeStream(FastSeq(globals), TStream(globals.typ)),
          Str(partFile(1, 0)),
          globalWriter,
        )

        bindIR(parts) { fileCountAndDistinct =>
          Begin(FastSeq(
            WriteMetadata(
              MakeArray(GetField(writeGlobals, "filePath")),
              RVDSpecWriter(
                s"$path/globals",
                RVDSpecMaker(globalSpec, RVDPartitioner.unkeyed(ctx.stateManager, 1)),
              ),
            ),
            WriteMetadata(
              ToArray(mapIR(ToStream(fileCountAndDistinct))(fc => GetField(fc, "filePath"))),
              RVDSpecWriter(
                s"$path/rows",
                RVDSpecMaker(
                  rowSpec,
                  partitioner,
                  IndexSpec.emptyAnnotation("../index", tcoerce[PStruct](pKey)),
                ),
              ),
            ),
            WriteMetadata(
              ToArray(mapIR(ToStream(fileCountAndDistinct)) { fc =>
                SelectFields(
                  fc,
                  FastSeq("partitionCounts", "distinctlyKeyed", "firstKey", "lastKey"),
                )
              }),
              TableSpecWriter(path, ts.tableType, "rows", "globals", "references", log = true),
            ),
          ))
        }
      }
    )
  }
}

case class TableNativeWriter(
  path: String,
  overwrite: Boolean = true,
  stageLocally: Boolean = false,
  codecSpecJSONStr: String = null,
) extends TableWriter {

  override def lower(ctx: ExecuteContext, ts: TableStage, r: RTable): IR = {
    val bufferSpec: BufferSpec = BufferSpec.parseOrDefault(codecSpecJSONStr)
    val rowSpec =
      TypedCodecSpec(EType.fromTypeAndAnalysis(ts.rowType, r.rowType), ts.rowType, bufferSpec)
    val globalSpec = TypedCodecSpec(
      EType.fromTypeAndAnalysis(ts.globalType, r.globalType),
      ts.globalType,
      bufferSpec,
    )

    TableNativeWriter.lower(ctx, ts, path, overwrite, stageLocally, rowSpec, globalSpec)
  }
}

object PartitionNativeWriter {
  val ctxType = TString

  def fullReturnType(keyType: TStruct): TStruct = TStruct(
    "filePath" -> TString,
    "partitionCounts" -> TInt64,
    "distinctlyKeyed" -> TBoolean,
    "firstKey" -> keyType,
    "lastKey" -> keyType,
    "partitionByteSize" -> TInt64,
  )

  def returnType(keyType: TStruct, trackTotalBytes: Boolean): TStruct = {
    val t = PartitionNativeWriter.fullReturnType(keyType)
    if (trackTotalBytes) t else t.filterSet(Set("partitionByteSize"), include = false)._1
  }
}

case class PartitionNativeWriter(
  spec: AbstractTypedCodecSpec,
  keyFields: IndexedSeq[String],
  partPrefix: String,
  index: Option[(String, PStruct)] = None,
  stagingFolder: Option[Path] = None,
  trackTotalBytes: Boolean = false,
) extends PartitionWriter {
  val keyType = spec.encodedVirtualType.asInstanceOf[TStruct].select(keyFields)._1

  def ctxType = PartitionNativeWriter.ctxType
  val returnType = PartitionNativeWriter.returnType(keyType, trackTotalBytes)

  def unionTypeRequiredness(
    r: TypeWithRequiredness,
    ctxType: TypeWithRequiredness,
    streamType: RIterable,
  ): Unit = {
    val rs = r.asInstanceOf[RStruct]
    val rKeyType = streamType.elementType.asInstanceOf[RStruct].select(keyFields.toArray)
    rs.field("firstKey").union(false)
    rs.field("firstKey").unionFrom(rKeyType)
    rs.field("lastKey").union(false)
    rs.field("lastKey").unionFrom(rKeyType)
    r.union(ctxType.required)
    r.union(streamType.required)
  }

  class StreamConsumer(
    ctx: SValue,
    private val cb: EmitCodeBuilder,
    private val region: Value[Region],
  ) {
    private val mb = cb.emb

    private val writeIndexInfo = index.map { case (name, ktype) =>
      val branchingFactor =
        Option(mb.ctx.getFlag("index_branching_factor")).map(_.toInt).getOrElse(4096)
      (
        name,
        ktype,
        StagedIndexWriter.withDefaults(ktype, mb.ecb, branchingFactor = branchingFactor),
      )
    }

    private val filename = mb.newLocal[String]("filename")
    private val stagingInfo = stagingFolder.map(folder => (folder, mb.newLocal[String]("stage")))

    private val ob = mb.newLocal[OutputBuffer]("write_ob")
    private val n = mb.newLocal[Long]("partition_count")

    private val byteCount =
      if (trackTotalBytes) Some(mb.newPLocal("partition_byte_count", SInt64)) else None

    private val distinctlyKeyed = mb.newLocal[Boolean]("distinctlyKeyed")
    private val keyEmitType = EmitType(spec.decodedPType(keyType).sType, false)
    private val firstSeenSettable = mb.newEmitLocal("pnw_firstSeen", keyEmitType)
    private val lastSeenSettable = mb.newEmitLocal("pnw_lastSeen", keyEmitType)
    private val lastSeenRegion = mb.newLocal[Region]("last_key_region")

    def setup(): Unit = {
      cb.assign(
        distinctlyKeyed,
        !keyFields.isEmpty,
      ) // True until proven otherwise, if there's a key to care about at all.
      // Start off missing, we will use this to determine if we haven't processed any rows yet.
      cb.assign(firstSeenSettable, EmitCode.missing(cb.emb, keyEmitType.st))
      cb.assign(lastSeenSettable, EmitCode.missing(cb.emb, keyEmitType.st))
      cb.assign(lastSeenRegion, Region.stagedCreate(Region.TINY, region.getPool()))

      val ctxValue = ctx.asString.loadString(cb)

      cb.assign(filename, const(partPrefix).concat(ctxValue))
      writeIndexInfo.foreach { case (indexName, _, writer) =>
        val indexFile = cb.newLocal[String]("indexFile")
        cb.assign(indexFile, const(indexName).concat(ctxValue).concat(".idx"))
        writer.init(cb, indexFile, cb.memoize(mb.getObject[Map[String, Any]](Map.empty)))
      }

      val stagingFile = stagingInfo.map { case (folder, fileRef) =>
        cb.assign(fileRef, const(s"$folder/").concat(ctxValue))
        fileRef
      }

      val os = mb.newLocal[ByteTrackingOutputStream]("write_os")
      cb.assign(
        os,
        Code.newInstance[ByteTrackingOutputStream, OutputStream](
          mb.create(stagingFile.getOrElse(filename).get)
        ),
      )
      cb.assign(ob, spec.buildCodeOutputBuffer(Code.checkcast[OutputStream](os)))
      cb.assign(n, 0L)
    }

    def consumeElement(cb: EmitCodeBuilder, codeRow: SValue, elementRegion: Settable[Region])
      : Unit = {
      val row = codeRow.asBaseStruct

      writeIndexInfo.foreach { case (_, indexKeyType, writer) =>
        writer.add(
          cb, {
            IEmitCode.present(
              cb,
              indexKeyType.asInstanceOf[PCanonicalBaseStruct]
                .constructFromFields(
                  cb,
                  elementRegion,
                  indexKeyType.fields.map { f =>
                    EmitCode.fromI(cb.emb)(cb => row.loadField(cb, f.name))
                  },
                  deepCopy = true,
                ),
            )
          },
          ob.invoke[Long]("indexOffset"),
          IEmitCode.present(cb, PCanonicalStruct().loadCheapSCode(cb, 0L)),
        )
      }

      val key = SStackStruct.constructFromArgs(
        cb,
        elementRegion,
        keyType,
        keyType.fields.map(f => EmitCode.fromI(cb.emb)(cb => row.loadField(cb, f.name))): _*
      )

      if (!keyFields.isEmpty) {
        cb.if_(
          distinctlyKeyed, {
            lastSeenSettable.loadI(cb).consume(
              cb,
              // If there's no last seen, we are in the first row.
              cb.assign(
                firstSeenSettable,
                EmitValue.present(key.copyToRegion(cb, region, firstSeenSettable.st)),
              ),
              { lastSeen =>
                val comparator = EQ(lastSeenSettable.emitType.virtualType).codeOrdering(
                  cb.emb.ecb,
                  lastSeenSettable.st,
                  key.st,
                )
                val equalToLast = comparator(cb, lastSeenSettable, EmitValue.present(key))
                cb.if_(equalToLast.asInstanceOf[Value[Boolean]], cb.assign(distinctlyKeyed, false))
              },
            )
          },
        )
        cb += lastSeenRegion.clearRegion()
        cb.assign(
          lastSeenSettable,
          IEmitCode.present(cb, key.copyToRegion(cb, lastSeenRegion, lastSeenSettable.st)),
        )
      }

      cb += ob.writeByte(1.asInstanceOf[Byte])

      spec.encodedType.buildEncoder(row.st, cb.emb.ecb)
        .apply(cb, row, ob)

      cb.assign(n, n + 1L)
      byteCount.foreach(bc => cb.assign(bc, SCode.add(cb, bc, row.sizeToStoreInBytes(cb), true)))
    }

    def result(): SValue = {
      cb += ob.writeByte(0.asInstanceOf[Byte])
      writeIndexInfo.foreach(_._3.close(cb))
      cb += ob.flush()
      cb += ob.close()

      stagingInfo.foreach { case (_, source) =>
        cb += mb.getFS.invoke[String, String, Boolean, Unit]("copy", source, filename, const(true))
      }

      lastSeenSettable.loadI(cb).consume(
        cb,
        { /* do nothing */ },
        lastSeen =>
          cb.assign(
            lastSeenSettable,
            IEmitCode.present(cb, lastSeen.copyToRegion(cb, region, lastSeenSettable.st)),
          ),
      )
      cb += lastSeenRegion.invalidate()
      val values = Seq[EmitCode](
        EmitCode.present(mb, ctx),
        EmitCode.present(mb, new SInt64Value(n)),
        EmitCode.present(mb, new SBooleanValue(distinctlyKeyed)),
        firstSeenSettable,
        lastSeenSettable,
      ) ++ byteCount.map(EmitCode.present(mb, _))

      SStackStruct.constructFromArgs(cb, region, returnType.asInstanceOf[TBaseStruct], values: _*)
    }
  }

  def consumeStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    stream: StreamProducer,
    context: EmitCode,
    region: Value[Region],
  ): IEmitCode = {
    val ctx = context.toI(cb).get(cb)
    val consumer = new StreamConsumer(ctx, cb, region)
    consumer.setup()
    stream.memoryManagedConsume(region, cb) { cb =>
      val element = stream.element.toI(cb).get(cb, "row can't be missing")
      consumer.consumeElement(cb, element, stream.elementRegion)
    }
    IEmitCode.present(cb, consumer.result())
  }
}

case class RVDSpecWriter(path: String, spec: RVDSpecMaker) extends MetadataWriter {
  def annotationType: Type = TArray(TString)

  def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region],
  ): Unit = {
    cb += cb.emb.getFS.invoke[String, Unit]("mkDir", path)
    val a = writeAnnotations.get(cb, "write annotations can't be missing!").asIndexable
    val partFiles = cb.newLocal[Array[String]]("partFiles")
    val n = cb.newLocal[Int]("n", a.loadLength())
    val i = cb.newLocal[Int]("i", 0)
    cb.assign(partFiles, Code.newArray[String](n))
    cb.while_(
      i < n, {
        val s = a.loadElement(cb, i).get(cb, "file name can't be missing!").asString
        cb += partFiles.update(i, s.loadString(cb))
        cb.assign(i, i + 1)
      },
    )
    cb += cb.emb.getObject(spec)
      .invoke[Array[String], AbstractRVDSpec]("apply", partFiles)
      .invoke[FS, String, Unit]("write", cb.emb.getFS, path)
  }
}

class TableSpecHelper(
  path: String,
  rowRelPath: String,
  globalRelPath: String,
  refRelPath: String,
  typ: TableType,
  log: Boolean,
) extends Serializable {
  def write(fs: FS, partCounts: Array[Long], distinctlyKeyed: Boolean): Unit = {
    val spec = TableSpecParameters(
      FileFormat.version.rep,
      is.hail.HAIL_PRETTY_VERSION,
      refRelPath,
      typ,
      Map(
        "globals" -> RVDComponentSpec(globalRelPath),
        "rows" -> RVDComponentSpec(rowRelPath),
        "partition_counts" -> PartitionCountsComponentSpec(partCounts),
        "properties" -> PropertiesSpec(JObject(
          "distinctlyKeyed" -> JBool(distinctlyKeyed)
        )),
      ),
    )

    spec.write(fs, path)

    val nRows = partCounts.sum
    if (log) info(s"wrote table with $nRows ${plural(nRows, "row")} " +
      s"in ${partCounts.length} ${plural(partCounts.length, "partition")} " +
      s"to $path")
  }
}

case class TableSpecWriter(
  path: String,
  typ: TableType,
  rowRelPath: String,
  globalRelPath: String,
  refRelPath: String,
  log: Boolean,
) extends MetadataWriter {
  def annotationType: Type = TArray(TStruct(
    "partitionCounts" -> TInt64,
    "distinctlyKeyed" -> TBoolean,
    "firstKey" -> typ.keyType,
    "lastKey" -> typ.keyType,
  ))

  def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region],
  ): Unit = {
    cb += cb.emb.getFS.invoke[String, Unit]("mkDir", path)

    val hasKey = !this.typ.keyType.fields.isEmpty

    val a = writeAnnotations.get(cb, "write annotations can't be missing!").asIndexable
    val partCounts = cb.newLocal[Array[Long]]("partCounts")

    val idxOfFirstKeyField =
      annotationType.asInstanceOf[TArray].elementType.asInstanceOf[TStruct].fieldIdx("firstKey")
    val keySType = a.st.elementType.asInstanceOf[SBaseStruct].fieldTypes(idxOfFirstKeyField)

    val lastSeenSettable = cb.emb.newEmitLocal(EmitType(keySType, false))
    cb.assign(lastSeenSettable, EmitCode.missing(cb.emb, keySType))
    val distinctlyKeyed = cb.newLocal[Boolean]("tsw_write_metadata_distinctlyKeyed", hasKey)

    val n = cb.newLocal[Int]("n", a.loadLength())
    val i = cb.newLocal[Int]("i", 0)
    cb.assign(partCounts, Code.newArray[Long](n))
    cb.while_(
      i < n, {
        val curElement =
          a.loadElement(cb, i).get(cb, "writeMetadata annotation can't be missing").asBaseStruct
        val count = curElement.asBaseStruct.loadField(cb, "partitionCounts").get(
          cb,
          "part count can't be missing!",
        ).asLong.value

        if (hasKey) {
          // Only nonempty partitions affect first, last, and distinctlyKeyed.
          cb.if_(
            count cne 0L, {
              val curFirst = curElement.loadField(cb, "firstKey").get(
                cb,
                const("firstKey of curElement can't be missing, part size was ") concat count.toS,
              )

              val comparator = NEQ(lastSeenSettable.emitType.virtualType).codeOrdering(
                cb.emb.ecb,
                lastSeenSettable.st,
                curFirst.st,
              )
              val notEqualToLast = comparator(
                cb,
                lastSeenSettable,
                EmitValue.present(curFirst),
              ).asInstanceOf[Value[Boolean]]

              val partWasDistinctlyKeyed =
                curElement.loadField(cb, "distinctlyKeyed").get(cb).asBoolean.value
              cb.assign(
                distinctlyKeyed,
                distinctlyKeyed && partWasDistinctlyKeyed && notEqualToLast,
              )
              cb.assign(lastSeenSettable, curElement.loadField(cb, "lastKey"))
            },
          )
        }

        cb += partCounts.update(i, count)
        cb.assign(i, i + 1)
      },
    )
    cb += cb.emb.getObject(new TableSpecHelper(path, rowRelPath, globalRelPath, refRelPath, typ,
      log))
      .invoke[FS, Array[Long], Boolean, Unit]("write", cb.emb.getFS, partCounts, distinctlyKeyed)
  }
}

object RelationalWriter {
  def scoped(path: String, overwrite: Boolean, refs: Option[TableType])(write: IR): IR =
    WriteMetadata(
      write,
      RelationalWriter(
        path,
        overwrite,
        refs.map(typ =>
          "references" -> (ReferenceGenome.getReferences(
            typ.rowType
          ) ++ ReferenceGenome.getReferences(typ.globalType))
        ),
      ),
    )
}

case class RelationalSetup(path: String, overwrite: Boolean, refs: Option[TableType])
    extends MetadataWriter {
  lazy val maybeRefs = refs.map(typ =>
    "references" -> (ReferenceGenome.getReferences(typ.rowType) ++ ReferenceGenome.getReferences(
      typ.globalType
    ))
  )

  def annotationType: Type = TVoid

  def writeMetadata(ignored: => IEmitCode, cb: EmitCodeBuilder, region: Value[Region]): Unit = {
    if (overwrite)
      cb += cb.emb.getFS.invoke[String, Boolean, Unit]("delete", path, true)
    else
      cb.if_(
        cb.emb.getFS.invoke[String, Boolean]("exists", path),
        cb._fatal(s"file already exists: $path"),
      )
    cb += cb.emb.getFS.invoke[String, Unit]("mkDir", path)

    maybeRefs.foreach { case (refRelPath, refs) =>
      val referencesFQPath = s"$path/$refRelPath"
      cb += cb.emb.getFS.invoke[String, Unit]("mkDir", referencesFQPath)
      refs.foreach { rg =>
        cb += Code.invokeScalaObject3[FS, String, ReferenceGenome, Unit](
          ReferenceGenome.getClass,
          "writeReference",
          cb.emb.getFS,
          referencesFQPath,
          cb.emb.getReferenceGenome(rg),
        )
      }
    }
  }
}

case class RelationalCommit(path: String) extends MetadataWriter {
  def annotationType: Type = TStruct()

  def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region],
  ): Unit = {
    cb += Code.invokeScalaObject2[FS, String, Unit](
      Class.forName("is.hail.utils.package$"),
      "writeNativeFileReadMe",
      cb.emb.getFS,
      path,
    )
    cb += cb.emb.create(s"$path/_SUCCESS").invoke[Unit]("close")
  }
}

case class RelationalWriter(
  path: String,
  overwrite: Boolean,
  maybeRefs: Option[(String, Set[String])],
) extends MetadataWriter {
  def annotationType: Type = TVoid

  def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region],
  ): Unit = {
    if (overwrite)
      cb += cb.emb.getFS.invoke[String, Boolean, Unit]("delete", path, true)
    else
      cb.if_(
        cb.emb.getFS.invoke[String, Boolean]("exists", path),
        cb._fatal(s"file already exists: $path"),
      )
    cb += cb.emb.getFS.invoke[String, Unit]("mkDir", path)

    maybeRefs.foreach { case (refRelPath, refs) =>
      val referencesFQPath = s"$path/$refRelPath"
      cb += cb.emb.getFS.invoke[String, Unit]("mkDir", referencesFQPath)
      refs.foreach { rg =>
        cb += Code.invokeScalaObject3[FS, String, ReferenceGenome, Unit](
          ReferenceGenome.getClass,
          "writeReference",
          cb.emb.getFS,
          referencesFQPath,
          cb.emb.getReferenceGenome(rg),
        )
      }
    }

    writeAnnotations.consume(
      cb,
      {},
      pc => assert(pc == SVoidValue),
    ) // PVoidCode.code is Code._empty

    cb += Code.invokeScalaObject2[FS, String, Unit](
      Class.forName("is.hail.utils.package$"),
      "writeNativeFileReadMe",
      cb.emb.getFS,
      path,
    )
    cb += cb.emb.create(s"$path/_SUCCESS").invoke[Unit]("close")
  }
}

case class TableTextWriter(
  path: String,
  typesFile: String = null,
  header: Boolean = true,
  exportType: String = ExportType.CONCATENATED,
  delimiter: String,
) extends TableWriter {

  override def canLowerEfficiently: Boolean = exportType != ExportType.PARALLEL_COMPOSABLE

  override def apply(ctx: ExecuteContext, tv: TableValue): Unit =
    tv.export(ctx, path, typesFile, header, exportType, delimiter)

  override def lower(ctx: ExecuteContext, ts: TableStage, r: RTable): IR = {
    require(exportType != ExportType.PARALLEL_COMPOSABLE)

    val ext = ctx.fs.getCodecExtension(path)

    val folder = if (exportType == ExportType.CONCATENATED)
      ctx.createTmpPath("write-table-concatenated")
    else
      path
    val lineWriter = TableTextPartitionWriter(
      ts.rowType,
      delimiter,
      writeHeader = exportType == ExportType.PARALLEL_HEADER_IN_SHARD,
    )

    ts.mapContexts { oldCtx =>
      val d = digitsNeeded(ts.numPartitions)
      val partFiles = Literal(
        TArray(TString),
        Array.tabulate(ts.numPartitions)(i => s"$folder/${partFile(d, i)}-").toFastSeq,
      )

      zip2(oldCtx, ToStream(partFiles), ArrayZipBehavior.AssertSameLength) { (ctxElt, pf) =>
        MakeStruct(FastSeq(
          "oldCtx" -> ctxElt,
          "partFile" -> pf,
        ))
      }
    }(GetField(_, "oldCtx")).mapCollectWithContextsAndGlobals("table_text_writer") {
      (rows, ctxRef) =>
        val file = GetField(ctxRef, "partFile") + UUID4() + Str(ext)
        WritePartition(rows, file, lineWriter)
    } { (parts, _) =>
      val commit = TableTextFinalizer(path, ts.rowType, delimiter, header, exportType)
      Begin(FastSeq(WriteMetadata(parts, commit)))
    }
  }
}

case class TableTextPartitionWriter(rowType: TStruct, delimiter: String, writeHeader: Boolean)
    extends SimplePartitionWriter {
  lazy val headerStr = rowType.fields.map(_.name).mkString(delimiter)

  override def preConsume(cb: EmitCodeBuilder, os: Value[OutputStream]): Unit = if (writeHeader) {
    cb += os.invoke[Array[Byte], Unit]("write", const(headerStr).invoke[Array[Byte]]("getBytes"))
    cb += os.invoke[Int, Unit]("write", '\n')
  }

  def consumeElement(
    cb: EmitCodeBuilder,
    element: EmitCode,
    os: Value[OutputStream],
    region: Value[Region],
  ): Unit = {
    require(element.st.virtualType == rowType)
    val delimBytes: Value[Array[Byte]] = cb.memoize(cb.emb.getObject(delimiter.getBytes))

    element.toI(cb).consume(
      cb,
      cb._fatal("stream element can not be missing!"),
      { case sv: SBaseStructValue =>
        // I hope we're buffering our writes correctly!
        (0 until sv.st.size).foreachBetween { i =>
          val f = sv.loadField(cb, i)
          val annotation = f.consumeCode[AnyRef](
            cb,
            Code._null[AnyRef],
            sv => StringFunctions.svalueToJavaValue(cb, region, sv),
          )
          val str = Code.invokeScalaObject2[Any, Type, String](
            TableAnnotationImpex.getClass,
            "exportAnnotation",
            annotation,
            cb.emb.getType(f.st.virtualType),
          )
          cb += os.invoke[Array[Byte], Unit]("write", str.invoke[Array[Byte]]("getBytes"))
        }(cb += os.invoke[Array[Byte], Unit]("write", delimBytes))
        cb += os.invoke[Int, Unit]("write", '\n')
      },
    )
  }
}

object TableTextFinalizer {
  def writeManifest(
    fs: FS,
    outputPath: String,
    files: Array[String],
    optionalAdditionalFirstPath: String,
  ): Unit = {

    def basename(f: String): String = (new java.io.File(f)).getName

    using(fs.createNoCompression(fs.makeQualified(outputPath + "/shard-manifest.txt"))) { os =>
      val bos = new BufferedOutputStream(os)

      if (optionalAdditionalFirstPath != null) {
        bos.write(basename(optionalAdditionalFirstPath).getBytes())
        bos.write('\n')
      }
      files.foreach { f =>
        bos.write(basename(f).getBytes())
        bos.write('\n')
      }
      bos.flush()
    }
  }

  def cleanup(fs: FS, outputPath: String, files: Array[String]): Unit = {
    val outputFiles = fs.listDirectory(fs.makeQualified(outputPath)).map(_.getPath).toSet
    val fileSet = files.map(fs.makeQualified(_)).toSet
    outputFiles.diff(fileSet).foreach(fs.delete(_, false))
  }
}

case class TableTextFinalizer(
  outputPath: String,
  rowType: TStruct,
  delimiter: String,
  header: Boolean = true,
  exportType: String = ExportType.CONCATENATED,
) extends MetadataWriter {
  def annotationType: Type = TArray(TString)

  def writeMetadata(writeAnnotations: => IEmitCode, cb: EmitCodeBuilder, region: Value[Region])
    : Unit = {
    val ctx: ExecuteContext = cb.emb.ctx
    val ext = ctx.fs.getCodecExtension(outputPath)
    val partPaths = writeAnnotations.get(cb, "write annotations cannot be missing!")
    val files = partPaths.castTo(cb, region, SJavaArrayString(true), false).asInstanceOf[
      SJavaArrayStringValue
    ].array
    exportType match {
      case ExportType.CONCATENATED =>
        val jFiles = if (header) {
          val headerFilePath = ctx.createTmpPath("header", ext)
          val headerStr = rowType.fields.map(_.name).mkString(delimiter)
          val os = cb.memoize(cb.emb.create(const(headerFilePath)))
          cb += os.invoke[Array[Byte], Unit](
            "write",
            const(headerStr).invoke[Array[Byte]]("getBytes"),
          )
          cb += os.invoke[Int, Unit]("write", '\n')
          cb += os.invoke[Unit]("close")

          val allFiles = cb.memoize(Code.newArray[String](files.length + 1))
          cb += (allFiles(0) = const(headerFilePath))
          cb += Code.invokeStatic5[System, Any, Int, Any, Int, Int, Unit](
            "arraycopy",
            files /*src*/,
            0 /*srcPos*/,
            allFiles /*dest*/,
            1 /*destPos*/,
            files.length, /*len*/
          )
          allFiles
        } else {
          files
        }

        cb += cb.emb.getFS.invoke[Array[String], String, Unit](
          "concatenateFiles",
          jFiles,
          const(outputPath),
        )

        val i = cb.newLocal[Int]("i")
        cb.for_(
          cb.assign(i, 0),
          i < jFiles.length,
          cb.assign(i, i + 1),
          cb += cb.emb.getFS.invoke[String, Boolean, Unit]("delete", jFiles(i), const(false)),
        )

      case ExportType.PARALLEL_HEADER_IN_SHARD =>
        cb += Code.invokeScalaObject3[FS, String, Array[String], Unit](
          TableTextFinalizer.getClass,
          "cleanup",
          cb.emb.getFS,
          outputPath,
          files,
        )
        cb += Code.invokeScalaObject4[FS, String, Array[String], String, Unit](
          TableTextFinalizer.getClass,
          "writeManifest",
          cb.emb.getFS,
          outputPath,
          files,
          Code._null[String],
        )

        cb += cb.emb.getFS.invoke[String, Unit]("touch", const(outputPath).concat("/_SUCCESS"))

      case ExportType.PARALLEL_SEPARATE_HEADER =>
        cb += Code.invokeScalaObject3[FS, String, Array[String], Unit](
          TableTextFinalizer.getClass,
          "cleanup",
          cb.emb.getFS,
          outputPath,
          files,
        )
        val headerPath = if (header) {
          val headerFilePath = const(s"$outputPath/header$ext")
          val headerStr = rowType.fields.map(_.name).mkString(delimiter)
          val os = cb.memoize(cb.emb.create(headerFilePath))
          cb += os.invoke[Array[Byte], Unit](
            "write",
            const(headerStr).invoke[Array[Byte]]("getBytes"),
          )
          cb += os.invoke[Int, Unit]("write", '\n')
          cb += os.invoke[Unit]("close")
          headerFilePath
        } else Code._null[String]

        cb += Code.invokeScalaObject4[FS, String, Array[String], String, Unit](
          TableTextFinalizer.getClass,
          "writeManifest",
          cb.emb.getFS,
          outputPath,
          files,
          headerPath,
        )
        cb += cb.emb.getFS.invoke[String, Unit]("touch", const(outputPath).concat("/_SUCCESS"))
    }
  }
}

class FanoutWriterTarget(
  val field: String,
  val path: String,
  val rowSpec: TypedCodecSpec,
  val keyPType: PStruct,
  val tableType: TableType,
  val rowWriter: PartitionNativeWriter,
  val globalWriter: PartitionNativeWriter,
)

case class TableNativeFanoutWriter(
  val path: String,
  val fields: IndexedSeq[String],
  overwrite: Boolean = true,
  stageLocally: Boolean = false,
  codecSpecJSONStr: String = null,
) extends TableWriter {

  override def lower(ctx: ExecuteContext, ts: TableStage, r: RTable): IR = {
    val partitioner = ts.partitioner
    val bufferSpec = BufferSpec.parseOrDefault(codecSpecJSONStr)
    val globalSpec = TypedCodecSpec(
      EType.fromTypeAndAnalysis(ts.globalType, r.globalType),
      ts.globalType,
      bufferSpec,
    )
    val targets = {
      val rowType = ts.rowType
      val rowRType = r.rowType
      val keyType = partitioner.kType
      val keyFields = keyType.fieldNames

      fields.map { field =>
        val targetPath = path + "/" + field
        val fieldAndKey = (field +: keyFields)
        val targetRowType = rowType.typeAfterSelectNames(fieldAndKey)
        val targetRowRType = rowRType.select(fieldAndKey)
        val rowSpec = TypedCodecSpec(
          EType.fromTypeAndAnalysis(targetRowType, targetRowRType),
          targetRowType,
          bufferSpec,
        )
        val keyPType = tcoerce[PStruct](rowSpec.decodedPType(keyType))
        val tableType = TableType(targetRowType, keyFields, ts.globalType)
        val rowWriter = PartitionNativeWriter(
          rowSpec,
          keyFields,
          s"$targetPath/rows/parts/",
          Some(s"$targetPath/index/" -> keyPType),
          if (stageLocally) Some(FileSystems.getDefault.getPath(
            ctx.localTmpdir,
            s"hail_staging_tmp_${UUID.randomUUID()}",
            "rows",
            "parts",
          ))
          else None,
        )
        val globalWriter =
          PartitionNativeWriter(globalSpec, IndexedSeq(), s"$targetPath/globals/parts/", None, None)
        new FanoutWriterTarget(field, targetPath, rowSpec, keyPType, tableType, rowWriter,
          globalWriter)
      }.toFastSeq
    }

    val writeTables = ts.mapContexts { oldCtx =>
      val d = digitsNeeded(ts.numPartitions)
      val partFiles = Literal(
        TArray(TString),
        Array.tabulate(ts.numPartitions)(i => s"${partFile(d, i)}-").toFastSeq,
      )

      zip2(oldCtx, ToStream(partFiles), ArrayZipBehavior.AssertSameLength) { (ctxElt, pf) =>
        MakeStruct(FastSeq(
          "oldCtx" -> ctxElt,
          "writeCtx" -> pf,
        ))
      }
    }(
      GetField(_, "oldCtx")
    ).mapCollectWithContextsAndGlobals("table_native_fanout_writer") { (rows, ctxRef) =>
      val file = GetField(ctxRef, "writeCtx")
      WritePartition(rows, file + UUID4(), new PartitionNativeFanoutWriter(targets))
    } { (parts, globals) =>
      bindIR(parts) { fileCountAndDistinct =>
        Begin(targets.zipWithIndex.map { case (target, index) =>
          Begin(FastSeq(
            WriteMetadata(
              MakeArray(
                GetField(
                  WritePartition(
                    MakeStream(FastSeq(globals), TStream(globals.typ)),
                    Str(partFile(1, 0)),
                    target.globalWriter,
                  ),
                  "filePath",
                )
              ),
              RVDSpecWriter(
                s"${target.path}/globals",
                RVDSpecMaker(globalSpec, RVDPartitioner.unkeyed(ctx.stateManager, 1)),
              ),
            ),
            WriteMetadata(
              ToArray(mapIR(ToStream(fileCountAndDistinct)) { fc =>
                GetField(GetTupleElement(fc, index), "filePath")
              }),
              RVDSpecWriter(
                s"${target.path}/rows",
                RVDSpecMaker(
                  target.rowSpec,
                  partitioner,
                  IndexSpec.emptyAnnotation("../index", tcoerce[PStruct](target.keyPType)),
                ),
              ),
            ),
            WriteMetadata(
              ToArray(mapIR(ToStream(fileCountAndDistinct)) { fc =>
                SelectFields(
                  GetTupleElement(fc, index),
                  FastSeq("partitionCounts", "distinctlyKeyed", "firstKey", "lastKey"),
                )
              }),
              TableSpecWriter(
                target.path,
                target.tableType,
                "rows",
                "globals",
                "references",
                log = true,
              ),
            ),
          ))
        }.toFastSeq)
      }
    }

    targets.foldLeft(writeTables) { (rest: IR, target: FanoutWriterTarget) =>
      RelationalWriter.scoped(
        target.path,
        overwrite,
        Some(target.tableType),
      )(
        rest
      )
    }
  }
}

class PartitionNativeFanoutWriter(
  targets: IndexedSeq[FanoutWriterTarget]
) extends PartitionWriter {
  def consumeStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    stream: StreamProducer,
    context: EmitCode,
    region: Value[Region],
  ): IEmitCode = {
    val ctx = context.toI(cb).get(cb)
    val consumers = targets.map(target => new target.rowWriter.StreamConsumer(ctx, cb, region))

    consumers.foreach(_.setup())
    stream.memoryManagedConsume(region, cb) { cb =>
      val row = stream.element.toI(cb).get(cb, "row can't be missing")

      (consumers zip targets).foreach { case (consumer, target) =>
        consumer.consumeElement(
          cb,
          row.asBaseStruct.subset((target.keyPType.fieldNames :+ target.field): _*),
          stream.elementRegion,
        )
      }
    }
    IEmitCode.present(
      cb,
      SStackStruct.constructFromArgs(
        cb,
        region,
        returnType,
        consumers.map(consumer => EmitCode.present(cb.emb, consumer.result())): _*
      ),
    )
  }

  def ctxType = TString

  def returnType: TTuple =
    TTuple(targets.map(target => target.rowWriter.returnType): _*)

  def unionTypeRequiredness(
    returnType: TypeWithRequiredness,
    ctxType: TypeWithRequiredness,
    streamType: RIterable,
  ): Unit = {
    val targetReturnTypes = returnType.asInstanceOf[RTuple].fields.map(_.typ)

    ((targetReturnTypes) zip targets).foreach { case (returnType, target) =>
      target.rowWriter.unionTypeRequiredness(returnType, ctxType, streamType)
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
  colKey: IndexedSeq[String],
) {
  def lower(ctx: ExecuteContext, ts: IndexedSeq[(TableStage, RTable)]): IR =
    writer.lower(
      ctx,
      ts.map { case (ts, rt) =>
        (LowerMatrixIR.colsFieldName, LowerMatrixIR.entriesFieldName, colKey, ts, rt)
      },
    )

  def apply(ctx: ExecuteContext, mvs: IndexedSeq[TableValue]): Unit = writer.apply(
    ctx,
    mvs.map(_.toMatrixValue(colKey)),
  )
}
