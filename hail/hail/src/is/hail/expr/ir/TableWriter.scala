package is.hail.expr.ir

import is.hail.PrettyVersion
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits.toRichIterable
import is.hail.expr.TableAnnotationImpex
import is.hail.expr.ir.defs._
import is.hail.expr.ir.functions.StringFunctions
import is.hail.expr.ir.lowering.{LowerMatrixIR, TableStage}
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.io.fs.FS
import is.hail.rvd.{AbstractRVDSpec, IndexSpec, RVDPartitioner, RVDSpecMaker}
import is.hail.types._
import is.hail.types.encoded.EType
import is.hail.types.physical._
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.concrete.{
  SJavaArrayString, SJavaArrayStringValue,
}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import java.io.{BufferedOutputStream, OutputStream, OutputStreamWriter}
import java.text.SimpleDateFormat
import java.util.Date

import org.json4s.{DefaultFormats, Formats, JBool, JObject, ShortTypeHints}

object TableWriter {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(
      List(classOf[TableNativeFanoutWriter], classOf[TableNativeWriter], classOf[TableTextWriter]),
      typeHintFieldName = "name",
    )
  }

  def writerHelper(
    rowSpec: TypedCodecSpec,
    rows: IR,
    partFile: IR,
    partRoot: IR,
    indexInfo: Option[(PStruct, IR)] = None,
    trackTotalBytes: Boolean = false
  ): IR = {
    val partResult = streamAggIR(rows) { row =>
      assert(row.typ.isInstanceOf[TStruct])
      TableWriter.rowWriterHelper(rowSpec, row, partFile, partRoot, indexInfo, trackTotalBytes)
    }
    bindIR(partResult)(TableWriter.resultHelper(_))
  }

  def rowWriterHelper(
    rowSpec: TypedCodecSpec,
    row: Atom,
    partFile: IR,
    partRoot: IR,
    indexInfo: Option[(PStruct, IR)] = None,
    trackTotalBytes: Boolean = false,
  ): IR = {
    val pKey = indexInfo.map(_._1).getOrElse(PCanonicalStruct())
    val initOpArgs = Seq(partFile, partRoot) ++ indexInfo.map(_._2)
    val zero = makestruct(
      "distinct" -> !pKey.fieldNames.isEmpty,
      "firstKey" -> NA(pKey.virtualType),
      "lastKey" -> NA(pKey.virtualType),
    )
    val args = Seq(
      "partpath" -> ApplyAggOp(
        WriteRows(rowSpec, indexInfo.map(_._1)),
        initOpArgs: _*
      )(row),
      "partitionCounts" -> ApplyAggOp(Count())(),
      "keyMeta" -> aggFoldIR(zero) { accum =>
        bindIRs(SelectFields(row, pKey.fieldNames), GetField(accum, "lastKey")) {
          case Seq(key, prev) =>
            makestruct(
              "distinct" -> (GetField(accum, "distinct") && Coalesce(FastSeq(
                prev.cne(key),
                True(),
              ))),
              "firstKey" -> Coalesce(FastSeq(GetField(accum, "firstKey"), key)),
              "lastKey" -> Coalesce(FastSeq(key, prev)),
            )
        }
      } { (accum1, accum2) =>
        Die("unreachable: calling combop on writer fold makes no sense", zero.typ)
      }) ++ Some("partitionByteSize" -> ApplyAggOp(Sum())(invoke("sizeofValue", TInt64, row))).filter(_ => trackTotalBytes)
    makestruct(args: _*)
  }

  def resultHelper(result: Atom): IR = {
    bindIR(GetField(result, "keyMeta")) { keymeta =>
      val args = Seq(
        "filePath" -> GetField(result, "partpath"),
        "partitionCounts" -> GetField(result, "partitionCounts"),
        "distinctlyKeyed" -> GetField(keymeta, "distinct"),
        "firstKey" -> GetField(keymeta, "firstKey"),
        "lastKey" -> GetField(keymeta, "lastKey"),
      ) ++ {
        if (tcoerce[TStruct](result.typ).hasField("partitionByteSize"))
          Some("partitionByteSize" -> GetField(result, "partitionByteSize"))
        else
          None
      }
      makestruct(args: _*)
    }
  }
}

abstract class TableWriter {
  def path: String

  def apply(ctx: ExecuteContext, tv: TableValue): Unit = {
    val tableStage = TableValueIntermediate(tv).asTableStage(ctx)
    CompileAndEvaluate(ctx, lower(ctx, tableStage, RTable.fromTableStage(ctx, tableStage)))
  }

  def lower(ctx: ExecuteContext, ts: TableStage, r: RTable): IR

  def canLowerEfficiently: Boolean =
    true
}

object TableNativeWriter {
  def lower(
    ctx: ExecuteContext,
    ts: TableStage,
    path: String,
    overwrite: Boolean,
    rowSpec: TypedCodecSpec,
    globalSpec: TypedCodecSpec,
  ): IR = {
    // write out partitioner key, which may be stricter than table key
    val partitioner = ts.partitioner
    val pKey: PStruct = tcoerce[PStruct](rowSpec.decodedPType(partitioner.kType))

    RelationalWriter.scoped(path, overwrite, Some(ts.tableType))(
      ts.mapContexts { oldCtx =>
        val d = digitsNeeded(ts.numPartitions)
        val partFiles = Literal(
          TArray(TString),
          ArraySeq.tabulate(ts.numPartitions)(i => s"${partFile(d, i)}-"),
        )

        zip2(oldCtx, ToStream(partFiles), ArrayZipBehavior.AssertSameLength) { (ctxElt, pf) =>
          MakeStruct(FastSeq(
            "oldCtx" -> ctxElt,
            "writeCtx" -> pf,
          ))
        }
      }(GetField(_, "oldCtx")).mapCollectWithContextsAndGlobals("table_native_writer") {
        (rows, ctxRef) =>
          bindIR(GetField(ctxRef, "writeCtx") + UUID4()) { partFile =>
            val partRoot = Str(s"$path/rows/parts/")
            val indexRoot = Str(s"$path/index/")
            TableWriter.writerHelper(rowSpec, rows, partFile, partRoot, Some(pKey -> indexRoot))
          }
      } { (parts, globals) =>
        val writeGlobals = TableWriter.writerHelper(
          globalSpec,
          MakeStream(globals),
          Str(partFile(1, 0)),
          Str(s"$path/globals/parts/"),
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
                  IndexSpec.emptyAnnotation(ctx, "../index", tcoerce[PStruct](pKey)),
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

  def writeFileReadMe(fs: FS, path: String): Unit = {
    val dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss")
    using(new OutputStreamWriter(fs.create(path + "/README.txt"))) { out =>
      out.write(
        s"""This folder comprises a Hail (www.hail.is) native Table or MatrixTable.
           |  Written with version $PrettyVersion
           |  Created at ${dateFormat.format(new Date())}""".stripMargin
      )
    }
  }
}

case class TableNativeWriter(
  path: String,
  overwrite: Boolean = true,
  codecSpecJSONStr: String = null,
) extends TableWriter {

  override def lower(ctx: ExecuteContext, ts: TableStage, r: RTable): IR = {
    val bufferSpec: BufferSpec = BufferSpec.parseOrDefault(codecSpecJSONStr)
    val rowSpec =
      TypedCodecSpec(EType.fromTypeAndAnalysis(ctx, ts.rowType, r.rowType), ts.rowType, bufferSpec)
    val globalSpec = TypedCodecSpec(
      EType.fromTypeAndAnalysis(ctx, ts.globalType, r.globalType),
      ts.globalType,
      bufferSpec,
    )

    TableNativeWriter.lower(ctx, ts, path, overwrite, rowSpec, globalSpec)
  }
}

case class RVDSpecWriter(path: String, spec: RVDSpecMaker) extends MetadataWriter {
  override def annotationType: Type = TArray(TString)

  override def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region],
  ): Unit = {
    cb += cb.emb.getFS.invoke[String, Unit]("mkDir", path)
    val a = writeAnnotations.getOrFatal(cb, "write annotations can't be missing!").asIndexable
    val partFiles = cb.newLocal[Array[String]]("partFiles")
    val n = cb.newLocal[Int]("n", a.loadLength)
    val i = cb.newLocal[Int]("i", 0)
    cb.assign(partFiles, Code.newArray[String](n))
    cb.while_(
      i < n, {
        val s = a.loadElement(cb, i).getOrFatal(cb, "file name can't be missing!").asString
        cb += partFiles.update(i, s.loadString(cb))
        cb.assign(i, i + 1)
      },
    )
    cb += cb.emb.getObject(spec)
      .invoke[Array[String], AbstractRVDSpec]("applyFromCodegen", partFiles)
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
) extends Serializable with Logging {
  def write(fs: FS, partCounts: Array[Long], distinctlyKeyed: Boolean): Unit = {
    val spec = TableSpecParameters(
      FileFormat.version.rep,
      is.hail.PrettyVersion,
      refRelPath,
      typ,
      Map(
        "globals" -> RVDComponentSpec(globalRelPath),
        "rows" -> RVDComponentSpec(rowRelPath),
        "partition_counts" -> PartitionCountsComponentSpec(ArraySeq.unsafeWrapArray(partCounts)),
        "properties" -> PropertiesSpec(JObject(
          "distinctlyKeyed" -> JBool(distinctlyKeyed)
        )),
      ),
    )

    spec.write(fs, path)

    val nRows = partCounts.sum
    if (log) logger.info(s"wrote table with $nRows ${plural(nRows, "row")} " +
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
  override def annotationType: Type = TArray(TStruct(
    "partitionCounts" -> TInt64,
    "distinctlyKeyed" -> TBoolean,
    "firstKey" -> typ.keyType,
    "lastKey" -> typ.keyType,
  ))

  override def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region],
  ): Unit = {
    cb += cb.emb.getFS.invoke[String, Unit]("mkDir", path)

    val hasKey = !this.typ.keyType.fields.isEmpty

    val a = writeAnnotations.getOrFatal(cb, "write annotations can't be missing!").asIndexable
    val partCounts = cb.newLocal[Array[Long]]("partCounts")

    val idxOfFirstKeyField =
      annotationType.asInstanceOf[TArray].elementType.asInstanceOf[TStruct].fieldIdx("firstKey")
    val keySType = a.st.elementType.asInstanceOf[SBaseStruct].fieldTypes(idxOfFirstKeyField)

    val lastSeenSettable = cb.emb.newEmitLocal(EmitType(keySType, false))
    cb.assign(lastSeenSettable, EmitCode.missing(cb.emb, keySType))
    val distinctlyKeyed = cb.newLocal[Boolean]("tsw_write_metadata_distinctlyKeyed", hasKey)

    val n = cb.newLocal[Int]("n", a.loadLength)
    val i = cb.newLocal[Int]("i", 0)
    cb.assign(partCounts, Code.newArray[Long](n))
    cb.while_(
      i < n, {
        val curElement =
          a.loadElement(cb, i).getOrFatal(
            cb,
            "writeMetadata annotation can't be missing",
          ).asBaseStruct
        val count = curElement.asBaseStruct.loadField(cb, "partitionCounts").getOrFatal(
          cb,
          "part count can't be missing!",
        ).asLong.value

        if (hasKey) {
          // Only nonempty partitions affect first, last, and distinctlyKeyed.
          cb.if_(
            count cne 0L, {
              val curFirst = curElement.loadField(cb, "firstKey").getOrFatal(
                cb,
                const("firstKey of curElement can't be missing, part size was ") concat count.toS,
              )

              val comparator = NEQ.codeOrdering(
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
                curElement.loadField(cb, "distinctlyKeyed").getOrAssert(cb).asBoolean.value
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

  override def annotationType: Type = TVoid

  override def writeMetadata(ignored: => IEmitCode, cb: EmitCodeBuilder, region: Value[Region])
    : Unit = {
    if (overwrite)
      cb += cb.emb.getFS.invoke[String, Boolean, Unit]("delete", path, true)
    else
      cb.if_(
        cb.emb.getFS.invoke[String, Boolean]("exists", path),
        cb._fatal(s"RelationalSetup.writeMetadata: file already exists: $path"),
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
  override def annotationType: Type = TStruct()

  override def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region],
  ): Unit = {
    cb += Code.invokeScalaObject2[FS, String, Unit](
      Class.forName("is.hail.expr.ir.TableNativeWriter$"),
      "writeFileReadMe",
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
  override def annotationType: Type = TVoid

  override def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region],
  ): Unit = {
    if (overwrite)
      cb += cb.emb.getFS.invoke[String, Boolean, Unit]("delete", path, true)
    else
      cb.if_(
        cb.emb.getFS.invoke[String, Boolean]("exists", path),
        cb._fatal(s"RelationalWriter.writeMetadata: file already exists: $path"),
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
      Class.forName("is.hail.expr.ir.TableNativeWriter$"),
      "writeFileReadMe",
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
        ArraySeq.tabulate(ts.numPartitions)(i => s"$folder/${partFile(d, i)}-"),
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

  override def consumeElement(
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
  override def annotationType: Type = TArray(TString)

  override def writeMetadata(
    writeAnnotations: => IEmitCode,
    cb: EmitCodeBuilder,
    region: Value[Region],
  ): Unit = {
    val ctx: ExecuteContext = cb.emb.ctx
    val ext = ctx.fs.getCodecExtension(outputPath)
    val partPaths = writeAnnotations.getOrFatal(cb, "write annotations cannot be missing!")
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

          val allFiles = cb.memoize(Code.newArray[String](files.length() + 1))
          cb += (allFiles(0) = const(headerFilePath))
          cb += Code.invokeStatic5[System, Any, Int, Any, Int, Int, Unit](
            "arraycopy",
            files /*src*/,
            0 /*srcPos*/,
            allFiles /*dest*/,
            1 /*destPos*/,
            files.length(), /*len*/
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
          i < jFiles.length(),
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
)

case class TableNativeFanoutWriter(
  val path: String,
  val fields: IndexedSeq[String],
  overwrite: Boolean = true,
  codecSpecJSONStr: String = null,
) extends TableWriter {

  override def lower(ctx: ExecuteContext, ts: TableStage, r: RTable): IR = {
    val partitioner = ts.partitioner
    val bufferSpec = BufferSpec.parseOrDefault(codecSpecJSONStr)
    val globalSpec = TypedCodecSpec(
      EType.fromTypeAndAnalysis(ctx, ts.globalType, r.globalType),
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
          EType.fromTypeAndAnalysis(ctx, targetRowType, targetRowRType),
          targetRowType,
          bufferSpec,
        )
        val keyPType = tcoerce[PStruct](rowSpec.decodedPType(keyType))
        val tableType = TableType(targetRowType, keyFields, ts.globalType)
        new FanoutWriterTarget(field, targetPath, rowSpec, keyPType, tableType)
      }
    }

    val writeTables = ts.mapContexts { oldCtx =>
      val d = digitsNeeded(ts.numPartitions)
      val partFiles = Literal(
        TArray(TString),
        ArraySeq.tabulate(ts.numPartitions)(i => s"${partFile(d, i)}-"),
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
      bindIR(GetField(ctxRef, "writeCtx") + UUID4()) { partFile =>
        val partResult = streamAggIR(rows) { row =>
          maketuple(targets.map { target =>
            aggBindIR(SelectFields(row, target.tableType.rowType.fieldNames)) { targetRow =>
              val partRoot = Str(s"${target.path}/rows/parts/")
              val indexRoot = Str(s"${target.path}/index/")
              TableWriter.rowWriterHelper(
                target.rowSpec,
                targetRow,
                partFile,
                partRoot,
                Some(target.keyPType -> indexRoot),
              )
            }
          }: _*)
        }
        bindIR(partResult) { partResult =>
          maketuple((0 until targets.length).map { i =>
            bindIR(GetTupleElement(partResult, i))(TableWriter.resultHelper(_))
          }: _*)
        }
      }
    } { (parts, globals) =>
      bindIR(parts) { fileCountAndDistinct =>
        Begin(targets.zipWithIndex.map { case (target, index) =>
          val writeGlobals = TableWriter.writerHelper(
            globalSpec,
            MakeStream(globals),
            Str(partFile(1, 0)),
            Str(s"${target.path}/globals/parts/"),
          )
          Begin(FastSeq(
            WriteMetadata(
              MakeArray(GetField(writeGlobals, "filePath")),
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
                  IndexSpec.emptyAnnotation(ctx, "../index", tcoerce[PStruct](target.keyPType)),
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
        })
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

object WrappedMatrixNativeMultiWriter {
  implicit val formats: Formats = MatrixNativeMultiWriter.formats +
    ShortTypeHints(List(classOf[WrappedMatrixNativeMultiWriter]))
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
