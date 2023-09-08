package is.hail.io.bgen

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.lowering.{TableStage, TableStageDependency}
import is.hail.expr.ir.streams.StreamProducer
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitMethodBuilder, EmitSettable, EmitValue, IEmitCode, IR, IRParserEnvironment, Literal, LowerMatrixIR, MakeStruct, MatrixHybridReader, MatrixReader, PartitionNativeIntervalReader, PartitionReader, ReadPartition, Ref, StreamTake, TableExecuteIntermediate, TableNativeReader, TableReader, TableValue, ToStream}
import is.hail.io._
import is.hail.io.fs.{FS, FileListEntry, SeekableDataInputStream}
import is.hail.io.index.{IndexReader, StagedIndexReader}
import is.hail.io.vcf.LoadVCF
import is.hail.rvd.RVDPartitioner
import is.hail.types._
import is.hail.types.physical.PType
import is.hail.types.physical.stypes.concrete.{SJavaArrayString, SStackStruct}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.sql.Row
import org.json4s.JsonAST.{JArray, JInt, JNull, JString}
import org.json4s.{DefaultFormats, Extraction, Formats, JObject, JValue}

import scala.collection.mutable
import scala.io.Source

case class BgenHeader(
  compression: Int, // 0 uncompressed, 1 zlib, 2 zstd
  nSamples: Int,
  nVariants: Int,
  headerLength: Int,
  dataStart: Long,
  hasIds: Boolean,
  version: Int,
  fileByteSize: Long,
  path: String
)

case class BgenFileMetadata(
  indexPath: String,
  indexVersion: SemanticVersion,
  header: BgenHeader,
  rg: Option[String],
  contigRecoding: Map[String, String],
  skipInvalidLoci: Boolean,
  nVariants: Long,
  @transient indexKeyType: Type,
  @transient indexAnnotationType: Type,
  @transient rangeBounds: Interval) {
  def nSamples: Int = header.nSamples
  def compression: Int = header.compression
  def path: String = header.path
}

object LoadBgen {
  def readSamples(fs: FS, file: String): Array[String] = {
    val bState = readState(fs, file)
    if (bState.hasIds) {
      using(new HadoopFSDataBinaryReader(fs.openNoCompression(file))) { is =>
        is.seek(bState.headerLength + 4)
        val sampleIdSize = is.readInt()
        val nSamples = is.readInt()

        if (nSamples != bState.nSamples)
          fatal("BGEN file is malformed -- number of sample IDs in header does not equal number in file")

        if (sampleIdSize + bState.headerLength > bState.dataStart - 4)
          fatal("BGEN file is malformed -- offset is smaller than length of header")

        (0 until nSamples).map { i =>
          is.readLengthAndString(2)
        }.toArray
      }
    } else {
      warn(s"BGEN file '$file' contains no sample ID block and no sample ID file given.\n" +
        s"  Using _0, _1, ..., _N as sample IDs.")
      (0 until bState.nSamples).map(i => s"_$i").toArray
    }
  }

  def readSampleFile(fs: FS, file: String): Array[String] = {
    using(fs.open(file)) { s =>
      Source.fromInputStream(s)
        .getLines()
        .drop(2)
        .filter(line => !line.isEmpty)
        .map { line =>
          val arr = line.split("\\s+")
          arr(0)
        }
        .toArray
    }
  }

  def readState(fs: FS, file: String): BgenHeader = {
    using(new HadoopFSDataBinaryReader(fs.openNoCompression(file))) { is =>
      readState(is, file, fs.getFileSize(file))
    }
  }

  def readState(is: HadoopFSDataBinaryReader, path: String, byteSize: Long): BgenHeader = {
    is.seek(0)
    val allInfoLength = is.readInt()
    val headerLength = is.readInt()
    val dataStart = allInfoLength.toLong + 4

    assert(headerLength <= allInfoLength)
    val nVariants = is.readInt()
    val nSamples = is.readInt()

    val magicNumber = is.readBytes(4).map(_.toInt).toFastIndexedSeq

    if (magicNumber != FastSeq(0, 0, 0, 0) && magicNumber != FastSeq(98, 103, 101, 110))
      fatal(s"expected magic number [0000] or [bgen], got [${ magicNumber.mkString }]")

    if (headerLength > 20)
      is.skipBytes(headerLength - 20)

    val flags = is.readInt()
    val compressType = flags & 3

    if (compressType != 0 && compressType != 1 && compressType != 2)
      fatal(s"Hail only supports zlib or zstd compression.")

    val version = (flags >>> 2) & 0xf
    if (version != 2)
      fatal(s"Hail supports BGEN version 1.2, got version 1.$version")

    val hasIds = (flags >> 31 & 1) != 0
    BgenHeader(
      compressType,
      nSamples,
      nVariants,
      headerLength,
      dataStart,
      hasIds,
      version,
      byteSize,
      path
    )
  }

  def checkVersionTwo(headers: Array[BgenHeader]) {
    val notVersionTwo = headers.filter(_.version != 2).map(x => x.path -> x.version)
    if (notVersionTwo.length > 0)
      fatal(
        s"""The following BGEN files are not BGENv2:
           |  ${ notVersionTwo.mkString("\n  ") }""".stripMargin)
  }

  def getAllFileListEntries(fs: FS, files: Array[String]): Array[FileListEntry] = {
    val badFiles = new BoxedArrayBuilder[String]()

    val fileListEntries = files.flatMap { file =>
      val matches = fs.glob(file)
      if (matches.isEmpty)
        badFiles += file

      matches.flatMap { fileListEntry =>
        val file = fileListEntry.getPath.toString
        if (!file.endsWith(".bgen"))
          warn(s"input file does not have .bgen extension: $file")

        if (fs.isDir(file))
          fs.listFileListEntry(file)
            .filter(fileListEntry => ".*part-[0-9]+(-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})?".r.matches(fileListEntry.getPath.toString))
        else
          Array(fileListEntry)
      }
    }

    if (!badFiles.isEmpty)
      fatal(
        s"""The following paths refer to no files:
           |  ${ badFiles.result().mkString("\n  ") }""".stripMargin)

    fileListEntries
  }

  def getAllFilePaths(fs: FS, files: Array[String]): Array[String] =
    getAllFileListEntries(fs, files).map(_.getPath.toString)


  def getBgenFileMetadata(ctx: ExecuteContext, files: Array[String], indexFiles: Array[String]): Array[BgenFileMetadata] = {
    val fs = ctx.fs
    require(files.length == indexFiles.length)
    val headers = getFileHeaders(fs, files)

    val cacheByRG: mutable.Map[Option[String], (String, Array[Long]) => Array[AnyRef]] = mutable.Map.empty

    headers.zip(indexFiles).map { case (h, indexFile) =>
      val (keyType, annotationType) = IndexReader.readTypes(fs, indexFile)
      val rg = keyType.asInstanceOf[TStruct].field("locus").typ match {
        case TLocus(rg) => Some(rg)
        case _ => None
      }
      val metadata = IndexReader.readMetadata(fs, indexFile, keyType, annotationType)
      val indexVersion = SemanticVersion(metadata.fileVersion)
      val (leafSpec, internalSpec) = BgenSettings.indexCodecSpecs(indexVersion, rg)

      val getKeys = cacheByRG.getOrElseUpdate(rg, StagedBGENReader.queryIndexByPosition(ctx, leafSpec, internalSpec))

      val attributes = metadata.attributes
      val skipInvalidLoci = attributes("skip_invalid_loci").asInstanceOf[Boolean]
      val contigRecoding = Option(attributes("contig_recoding")).map(_.asInstanceOf[Map[String, String]]).getOrElse(Map.empty[String, String])
      val nVariants = metadata.nKeys

      val rangeBounds = if (nVariants > 0) {
        val Array(start, end) = getKeys(indexFile, Array[Long](0L, nVariants - 1))
        Interval(start, end, includesStart = true, includesEnd = true)
      } else null

      BgenFileMetadata(
        indexFile,
        indexVersion,
        h,
        rg,
        contigRecoding,
        skipInvalidLoci,
        metadata.nKeys,
        keyType,
        annotationType,
        rangeBounds)
    }
  }

  def getIndexFileNames(fs: FS, files: Array[String], indexFileMap: Map[String, String]): Array[String] = {
    def absolutePath(rel: String): String = fs.fileListEntry(rel).getPath.toString

    val fileMapping = Option(indexFileMap)
      .getOrElse(Map.empty[String, String])
      .map { case (f, index) => (absolutePath(f), index) }

    val badExtensions = fileMapping.filterNot { case (_, f) => f.endsWith("idx2") }.values
    if (badExtensions.nonEmpty)
      fatal(
        s"""The following index file paths defined by 'index_file_map' are missing a .idx2 file extension:
           |  ${ badExtensions.mkString("\n  ") })""".stripMargin)

    files.map(absolutePath).map(f => fileMapping.getOrElse(f, f + ".idx2"))
  }

  def getIndexFiles(fs: FS, files: Array[String], indexFileMap: Map[String, String]): Array[String] = {
    val indexFiles = getIndexFileNames(fs, files, indexFileMap)
    val missingIdxFiles = files.zip(indexFiles).filterNot { case (f, index) => fs.exists(index) && index.endsWith("idx2") }.map(_._1)
    if (missingIdxFiles.nonEmpty)
      fatal(
        s"""The following BGEN files have no .idx2 index file. Use 'index_bgen' to create the index file once before calling 'import_bgen':
           |  ${ missingIdxFiles.mkString("\n  ") })""".stripMargin)
    indexFiles
  }

  def getFileHeaders(fs: FS, files: Seq[String]): Array[BgenHeader] =
    files.map(LoadBgen.readState(fs, _)).toArray

  def getReferenceGenome(fileMetadata: Array[BgenFileMetadata]): Option[String] =
    getReferenceGenome(fileMetadata.map(_.rg))

  def getReferenceGenome(rgs: Array[Option[String]]): Option[String] = {
    if (rgs.distinct.length != 1)
      fatal(
        s"""Found multiple reference genomes were specified in the BGEN index files:
           |  ${ rgs.distinct.map(_.getOrElse("None")).mkString("\n  ") }""".stripMargin)
    rgs.head
  }

  def getIndexTypes(fileMetadata: Array[BgenFileMetadata]): (Type, Type) = {
    val indexKeyTypes = fileMetadata.map(_.indexKeyType).distinct
    val indexAnnotationTypes = fileMetadata.map(_.indexAnnotationType).distinct

    if (indexKeyTypes.length != 1)
      fatal(
        s"""Found more than one BGEN index key type:
           |  ${ indexKeyTypes.mkString("\n  ") })""".stripMargin)

    if (indexAnnotationTypes.length != 1)
      fatal(
        s"""Found more than one BGEN index annotation type:
           |  ${ indexAnnotationTypes.mkString("\n  ") })""".stripMargin)

    (indexKeyTypes.head, indexAnnotationTypes.head)
  }
}

object MatrixBGENReader {
  def fullMatrixTypeWithoutUIDs(rg: Option[String]): MatrixType = {
    MatrixType(
      globalType = TStruct.empty,
      colType = TStruct(
        "s" -> TString),
      colKey = Array("s"),
      rowType = TStruct(
        "locus" -> TLocus.schemaFromRG(rg),
        "alleles" -> TArray(TString),
        "rsid" -> TString,
        "varid" -> TString,
        "offset" -> TInt64,
        "file_idx" -> TInt32),
      rowKey = Array("locus", "alleles"),
      entryType = TStruct(
        "GT" -> TCall,
        "GP" -> TArray(TFloat64),
        "dosage" -> TFloat64))
  }

  def fullMatrixType(rg: Option[String]): MatrixType = {
    val mt = fullMatrixTypeWithoutUIDs(rg)
    val newRowType = mt.rowType.appendKey(MatrixReader.rowUIDFieldName, TTuple(TInt64, TInt64))
    val newColType = mt.colType.appendKey(MatrixReader.colUIDFieldName, TInt64)

    mt.copy(rowType = newRowType, colType = newColType)
  }

  def fullTableType(rg: Option[String]): TableType = {
    val mt = fullMatrixTypeWithoutUIDs(rg)
    val ttNoUID = mt.copy(mt.colType.appendKey(MatrixReader.colUIDFieldName, TInt64))
      .toTableType(LowerMatrixIR.entriesFieldName, LowerMatrixIR.colsFieldName)

    ttNoUID.copy(rowType = ttNoUID.rowType.appendKey(MatrixReader.rowUIDFieldName, TTuple(TInt64, TInt64)))
  }


  def fromJValue(env: IRParserEnvironment, jv: JValue): MatrixBGENReader = {
    MatrixBGENReader(env.ctx, MatrixBGENReaderParameters.fromJValue(jv))
  }

  def apply(ctx: ExecuteContext,
    files: Seq[String],
    sampleFile: Option[String],
    indexFileMap: Map[String, String],
    nPartitions: Option[Int],
    blockSizeInMB: Option[Int],
    includedVariants: Option[String]): MatrixBGENReader = {
    MatrixBGENReader(ctx,
      MatrixBGENReaderParameters(files, sampleFile, indexFileMap, nPartitions, blockSizeInMB, includedVariants))
  }

  def apply(ctx: ExecuteContext, params: MatrixBGENReaderParameters): MatrixBGENReader = {
    val fs = ctx.fs

    val allFiles = LoadBgen.getAllFilePaths(fs, params.files.toArray)
    val indexFiles = LoadBgen.getIndexFiles(fs, allFiles, params.indexFileMap)
    val fileMetadata = LoadBgen.getBgenFileMetadata(ctx, allFiles, indexFiles)
    assert(fileMetadata.nonEmpty)
    if (fileMetadata.exists(md => md.indexVersion != fileMetadata.head.indexVersion)) {
      fatal("BGEN index version mismatch. The index versions of all files must be the same, use 'index_bgen' to reindex all files to ensure that all index versions match before calling 'import_bgen' again")
    }

    val sampleIds = params.sampleFile.map(file => LoadBgen.readSampleFile(fs, file))
      .getOrElse(LoadBgen.readSamples(fs, fileMetadata.head.path))

    LoadVCF.warnDuplicates(sampleIds)

    val nSamples = sampleIds.length

    val unequalSamples = fileMetadata.filter(_.header.nSamples != nSamples).map(x => (x.path, x.header.nSamples))
    if (unequalSamples.length > 0) {
      val unequalSamplesString =
        unequalSamples.map(x => s"""(${ x._2 } ${ x._1 }""").mkString("\n  ")
      fatal(
        s"""The following BGEN files did not contain the expected number of samples $nSamples:
           |  $unequalSamplesString""".stripMargin)
    }

    val noVariants = fileMetadata.filter(_.nVariants == 0).map(_.path)
    if (noVariants.length > 0)
      fatal(
        s"""The following BGEN files did not contain at least 1 variant:
           |  ${ noVariants.mkString("\n  ") })""".stripMargin)

    LoadBgen.checkVersionTwo(fileMetadata.map(_.header))

    val nVariants = fileMetadata.map(_.nVariants).sum

    info(s"Number of BGEN files parsed: ${ fileMetadata.length }")
    info(s"Number of samples in BGEN files: $nSamples")
    info(s"Number of variants across all BGEN files: $nVariants")

    val referenceGenome = LoadBgen.getReferenceGenome(fileMetadata)

    val fullMatrixType: MatrixType = MatrixBGENReader.fullMatrixTypeWithoutUIDs(referenceGenome)

    val (indexKeyType, indexAnnotationType) = LoadBgen.getIndexTypes(fileMetadata)

    val filePartInfo = BgenRDDPartitions(ctx, referenceGenome, fileMetadata,
      if (params.nPartitions.isEmpty && params.blockSizeInMB.isEmpty)
        Some(128)
      else
        params.blockSizeInMB, params.nPartitions, indexKeyType)

    new MatrixBGENReader(
      params, referenceGenome, fullMatrixType, indexKeyType, indexAnnotationType, sampleIds, filePartInfo, params.includedVariants)
  }
}

object MatrixBGENReaderParameters {
  def fromJValue(jv: JValue): MatrixBGENReaderParameters = {
    implicit val foramts: Formats = DefaultFormats
    val files = (jv \ "files").extract[Array[String]]
    val sampleFile = (jv \ "sampleFile").extractOpt[String]
    val indexFileMap = (jv \ "indexFileMap").extract[Map[String, String]]
    val nPartitions = (jv \ "nPartitions").extractOpt[Int]
    val blockSizeInMB = (jv \ "blockSizeInMB").extractOpt[Int]
    val includedVariants = jv \ "includedVariants" match {
      case JNull => None
      case JString(s) => Some(s)
    }
    new MatrixBGENReaderParameters(files, sampleFile, indexFileMap, nPartitions, blockSizeInMB, includedVariants)
  }
}

case class MatrixBGENReaderParameters(
  files: Seq[String],
  sampleFile: Option[String],
  indexFileMap: Map[String, String],
  nPartitions: Option[Int],
  blockSizeInMB: Option[Int],
  includedVariants: Option[String]) {

  def toJValue: JValue = {
    JObject(List(
      "name" -> JString("MatrixBGENReader"),
      "files" -> JArray(files.map(JString).toList),
      "sampleFile" -> sampleFile.map(JString).getOrElse(JNull),
      "indexFileMap" -> JObject(indexFileMap.map { case (k, v) =>
        k -> JString(v)
      }.toList),
      "nPartitions" -> nPartitions.map(JInt(_)).getOrElse(JNull),
      "blockSizeInMB" -> blockSizeInMB.map(JInt(_)).getOrElse(JNull),
      "includedVariants" -> includedVariants.map(t => JString(t)).getOrElse(JNull)))
  }
}

class MatrixBGENReader(
  val params: MatrixBGENReaderParameters,
  referenceGenome: Option[String],
  val fullMatrixTypeWithoutUIDs: MatrixType,
  indexKeyType: Type,
  indexAnnotationType: Type,
  sampleIds: Array[String],
  filePartitionInfo: IndexedSeq[FilePartitionInfo],
  variants: Option[String]) extends MatrixHybridReader {
  def pathsUsed: Seq[String] = filePartitionInfo.map(_.metadata.path)

  lazy val nVariants: Long = filePartitionInfo.map(_.metadata.nVariants).sum

  def rowUIDType = TTuple(TInt64, TInt64)

  def colUIDType = TInt64

  private val nSamples = sampleIds.length

  def columnCount: Option[Int] = Some(nSamples)

  def partitionCounts: Option[IndexedSeq[Long]] = None

  private var _settings: BgenSettings = _

  def getSettings(requestedType: TableType): BgenSettings = {
    if (_settings == null || _settings.requestedType != requestedType) {
      _settings = BgenSettings(
        nSamples,
        requestedType,
        referenceGenome,
        indexAnnotationType)
    }
    _settings
  }

  override def concreteRowRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq = {
    val settings = getSettings(requestedType)
    VirtualTypeWithReq(settings.rowPType)
  }

  override def uidRequiredness: VirtualTypeWithReq =
    VirtualTypeWithReq.fullyRequired(TTuple(TInt64, TInt64))

  override def globalRequiredness(ctx: ExecuteContext, requestedType: TableType): VirtualTypeWithReq =
    VirtualTypeWithReq(PType.canonical(requestedType.globalType, required = true))

  override def lowerGlobals(ctx: ExecuteContext, requestedGlobalType: TStruct): IR = {
    requestedGlobalType.fieldOption(LowerMatrixIR.colsFieldName) match {
      case Some(f) =>
        val ta = f.typ.asInstanceOf[TArray]
        MakeStruct(FastSeq((LowerMatrixIR.colsFieldName, {
          val arraysToZip = new BoxedArrayBuilder[IndexedSeq[Any]]()
          val colType = ta.elementType.asInstanceOf[TStruct]
          if (colType.hasField("s"))
            arraysToZip += sampleIds
          if (colType.hasField(colUIDFieldName))
            arraysToZip += sampleIds.indices.map(_.toLong)

          val fields = arraysToZip.result()
          Literal(ta, sampleIds.indices.map(i => Row.fromSeq(fields.map(_.apply(i)))))
        })))
      case None => MakeStruct(FastSeq())
    }
  }

  override def toJValue: JValue = params.toJValue

  def renderShort(): String = defaultRender()

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: MatrixBGENReader => params == that.params
    case _ => false
  }

  override def lower(ctx: ExecuteContext, requestedType: TableType): TableStage = {

    val globals = lowerGlobals(ctx, requestedType.globalType)
    variants match {
      case Some(v) =>
        val t0 = TableNativeReader.read(ctx.fs, v, None)

        val contexts = new BoxedArrayBuilder[Row]()
        val rangeBounds = new BoxedArrayBuilder[Interval]()
        filePartitionInfo.zipWithIndex.foreach { case (file, fileIdx) =>
          val filePartitioner = new RVDPartitioner(ctx.stateManager, tcoerce[TStruct](indexKeyType), file.intervals)

          val filterKeyLen = t0.spec.table_type.key.length
          val strictShortKey = filePartitioner.coarsen(filterKeyLen).strictify()
          val strictBgenKey = strictShortKey.extendKey(filePartitioner.kType)
          rangeBounds ++= strictBgenKey.rangeBounds

          strictShortKey.partitionBoundsIRRepresentation.value.asInstanceOf[IndexedSeq[_]]
            .foreach { interval =>
              contexts += Row(fileIdx, interval)
            }
        }

        val partitioner = new RVDPartitioner(ctx.stateManager, tcoerce[TStruct](indexKeyType), rangeBounds.result())

        val reader = BgenPartitionReaderWithVariantFilter(
          filePartitionInfo.map(_.metadata).toArray,
          referenceGenome,
          PartitionNativeIntervalReader(ctx.stateManager, v, t0.spec, "__dummy"))

        TableStage(
          globals = globals,
          partitioner = partitioner,
          dependency = TableStageDependency.none,
          contexts = ToStream(Literal(TArray(reader.contextType), contexts.result().toFastIndexedSeq)),
          (ref: Ref) => ReadPartition(ref, requestedType.rowType, reader)
        )

      case None =>
        val partitioner = new RVDPartitioner(ctx.stateManager, tcoerce[TStruct](indexKeyType), filePartitionInfo.flatMap(_.intervals))
        val reader = BgenPartitionReader(fileMetadata = filePartitionInfo.map(_.metadata).toArray, referenceGenome)

        val contexts = new BoxedArrayBuilder[Row]()

        var partIdx = 0
        var fileIdx = 0
        filePartitionInfo.foreach { file =>
          assert(file.intervals.length == file.partN.length && file.intervals.length == file.partStarts.length)
          file.intervals.indices.foreach { idxInFile =>
            contexts += Row(fileIdx, file.partStarts(idxInFile), file.partN(idxInFile), partIdx)
            partIdx += 1
          }
          fileIdx += 1
        }

        TableStage(
          globals = globals,
          partitioner = partitioner,
          dependency = TableStageDependency.none,
          contexts = ToStream(Literal(TArray(reader.contextType), contexts.result().toFastIndexedSeq)),
          (ref: Ref) => ReadPartition(ref, requestedType.rowType, reader)
        )
    }
  }
}

case class BgenPartitionReaderWithVariantFilter(fileMetadata: Array[BgenFileMetadata], rg: Option[String], child: PartitionNativeIntervalReader) extends PartitionReader {
  lazy val contextType: TStruct = TStruct(
    "file_index" -> TInt32,
    "interval" -> RVDPartitioner.intervalIRRepresentation(child.tableSpec.table_type.keyType))

  lazy val uidType = TTuple(TInt64, TInt64)
  lazy val fullRowType: TStruct = MatrixBGENReader.fullTableType(rg).rowType

  def rowRequiredness(requestedType: TStruct): RStruct = StagedBGENReader.rowRequiredness(requestedType)

  def uidFieldName: String = TableReader.uidFieldName

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct): IEmitCode = {

    val cbfis = mb.genFieldThisRef[HadoopFSDataBinaryReader]("bgen_cbfis")
    val nSamples = mb.genFieldThisRef[Int]("bgen_nsamples")
    val fileIdx = mb.genFieldThisRef[Int]("bgen_fileIdx")
    val compression = mb.genFieldThisRef[Int]("bgen_compression")
    val skipInvalidLoci = mb.genFieldThisRef[Boolean]("bgen_skip_invalid_loci")
    val contigRecoding = mb.genFieldThisRef[Map[String, String]]("bgen_contig_recoding")

    val indexNKeys = mb.genFieldThisRef[Long]("index_nkeys")
    val (leafCodec, intCodec) = BgenSettings.indexCodecSpecs(fileMetadata.head.indexVersion, rg)
    val index = new StagedIndexReader(mb, leafCodec, intCodec)

    val currVariantIndex = mb.genFieldThisRef[Long]("currVariantIndex")
    val stopVariantIndex = mb.genFieldThisRef[Long]("stopVariantIndex")

    var out: EmitSettable = null // filled in later

    context.toI(cb).flatMap(cb) { case context: SBaseStructValue =>

      val rangeBound = EmitCode.fromI(mb)(cb => context.loadField(cb, "interval"))

      child.emitStream(ctx, cb, mb, rangeBound, child.fullRowType.deleteKey(child.uidFieldName))
        .map(cb) { case variantsStream: SStreamValue =>
          val vs = variantsStream.getProducer(mb)

          SStreamValue(new StreamProducer {
            override def method: EmitMethodBuilder[_] = mb

            override val length: Option[EmitCodeBuilder => Code[Int]] = None

            override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {
              vs.initialize(cb, outerRegion)

              cb.assign(fileIdx, context.loadField(cb, "file_index").get(cb).asInt.value)
              val metadata = cb.memoize(mb.getObject[IndexedSeq[BgenFileMetadata]](fileMetadata.toFastIndexedSeq)
                .invoke[Int, BgenFileMetadata]("apply", fileIdx))
              val fileName = cb.memoize(metadata.invoke[String]("path"))
              val indexName = cb.memoize(metadata.invoke[String]("indexPath"))
              cb.assign(nSamples, metadata.invoke[Int]("nSamples"))
              cb.assign(contigRecoding, metadata.invoke[Map[String, String]]("contigRecoding"))
              cb.assign(compression, metadata.invoke[Int]("compression"))
              cb.assign(skipInvalidLoci, metadata.invoke[Boolean]("skipInvalidLoci"))

              cb.assign(cbfis, Code.newInstance[HadoopFSDataBinaryReader, SeekableDataInputStream](
                mb.getFS.invoke[String, SeekableDataInputStream]("openNoCompression", fileName)))
              index.initialize(cb, indexName)
              cb.assign(indexNKeys, index.nKeys(cb))
            }

            override val elementRegion: Settable[Region] = vs.elementRegion
            override val requiresMemoryManagementPerElement: Boolean = vs.requiresMemoryManagementPerElement
            override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
              val Lstart = CodeLabel()
              cb.define(Lstart)
              cb.ifx(currVariantIndex < stopVariantIndex, {
                val addr = index.queryIndex(cb, vs.elementRegion, currVariantIndex)
                  .loadField(cb, "offset")
                  .get(cb).asLong.value
                cb += cbfis.invoke[Long, Unit]("seek", addr)

                val reqTypeNoUID = if (requestedType.hasField(uidFieldName)) requestedType.deleteKey(uidFieldName) else requestedType
                val sc = StagedBGENReader.decodeRow(cb, elementRegion, cbfis, nSamples, fileIdx, compression, skipInvalidLoci, contigRecoding, reqTypeNoUID, rg)
                  .toI(cb).get(cb)
                val scUID = if (requestedType.hasField(uidFieldName))
                  sc.asBaseStruct.insert(cb, elementRegion, requestedType,
                    (uidFieldName, EmitValue.present(SStackStruct.constructFromArgs(cb, elementRegion, uidType,
                      EmitValue.present(primitive(cb.memoize(fileIdx.toL))),
                      EmitValue.present(primitive(currVariantIndex))))))
                else
                  sc
                out = mb.newEmitField(scUID.st, true)
                cb.assign(out, EmitCode.present(mb, scUID))

                cb.assign(currVariantIndex, currVariantIndex + 1)
                cb.goto(LproduceElementDone)
              })


              cb.goto(vs.LproduceElement)
              cb.define(vs.LproduceElementDone)

              val nextVariant = vs.element.toI(cb).get(cb).asBaseStruct
              val bound = SStackStruct.constructFromArgs(cb, vs.elementRegion, TTuple(nextVariant.st.virtualType, TInt32),
                EmitValue.present(if (nextVariant.st.size == 1)
                  nextVariant.insert(cb, elementRegion,
                    nextVariant.st.virtualType.insert(TArray(TString), "alleles")._1.asInstanceOf[TStruct], ("alleles", EmitValue.missing(SJavaArrayString(true))))
                else
                  nextVariant),
                EmitValue.present(primitive(const(nextVariant.st.size)))
              )

              cb.assign(currVariantIndex, index.queryBound(cb, bound, false).loadField(cb, 0).get(cb).asLong.value)
              cb.assign(stopVariantIndex, index.queryBound(cb, bound, true).loadField(cb, 0).get(cb).asLong.value)
              cb.goto(Lstart)

              cb.define(vs.LendOfStream)
              cb.goto(LendOfStream)
            }
            override val element: EmitCode = out

            override def close(cb: EmitCodeBuilder): Unit = {
              cb += cbfis.invoke[Unit]("close")
              index.close(cb)
              vs.close(cb)
            }

          })
        }
    }
  }

  def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)
}


case class BgenPartitionReader(fileMetadata: Array[BgenFileMetadata], rg: Option[String]) extends PartitionReader {
  lazy val contextType: TStruct = TStruct(
    "file_index" -> TInt32,
    "first_variant_index" -> TInt64,
    "n_variants" -> TInt64,
    "partition_index" -> TInt32)

  lazy val uidType = TTuple(TInt64, TInt64)

  lazy val fullRowType: TStruct = MatrixBGENReader.fullTableType(rg).rowType

  def rowRequiredness(requestedType: TStruct): RStruct = StagedBGENReader.rowRequiredness(requestedType)

  def uidFieldName: String = TableReader.uidFieldName

  def emitStream(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    mb: EmitMethodBuilder[_],
    context: EmitCode,
    requestedType: TStruct): IEmitCode = {

    val eltRegion = mb.genFieldThisRef[Region]("bgen_region")
    val cbfis = mb.genFieldThisRef[HadoopFSDataBinaryReader]("bgen_cbfis")
    val nSamples = mb.genFieldThisRef[Int]("bgen_nsamples")
    val fileIdx = mb.genFieldThisRef[Int]("bgen_fileIdx")
    val compression = mb.genFieldThisRef[Int]("bgen_compression")
    val skipInvalidLoci = mb.genFieldThisRef[Boolean]("bgen_skip_invalid_loci")
    val contigRecoding = mb.genFieldThisRef[Map[String, String]]("bgen_contig_recoding")

    val currVariantIndex = mb.genFieldThisRef[Long]("bgen_currIdx")
    val endVariantIndex = mb.genFieldThisRef[Long]("bgen_endIdx")
    val (leafCodec, intCodec) = BgenSettings.indexCodecSpecs(fileMetadata.head.indexVersion, rg)
    val index = new StagedIndexReader(mb, leafCodec, intCodec)

    var out: EmitSettable = null // filled in later

    context.toI(cb).map(cb) { case context: SBaseStructValue =>

      val ctxField = cb.memoizeField(context, "ctxField")
      SStreamValue(new StreamProducer {
        override def method: EmitMethodBuilder[_] = mb

        override val length: Option[EmitCodeBuilder => Code[Int]] = Some(cb => ctxField.asBaseStruct.loadField(cb, "n_variants").get(cb).asLong.value.toI)

        override def initialize(cb: EmitCodeBuilder, outerRegion: Value[Region]): Unit = {

          cb.assign(fileIdx, context.loadField(cb, "file_index").get(cb).asInt.value)
          val metadata = cb.memoize(mb.getObject[IndexedSeq[BgenFileMetadata]](fileMetadata.toFastIndexedSeq)
            .invoke[Int, BgenFileMetadata]("apply", fileIdx))
          val fileName = cb.memoize(metadata.invoke[String]("path"))
          val indexName = cb.memoize(metadata.invoke[String]("indexPath"))
          cb.assign(nSamples, metadata.invoke[Int]("nSamples"))
          cb.assign(contigRecoding, metadata.invoke[Map[String, String]]("contigRecoding"))
          cb.assign(compression, metadata.invoke[Int]("compression"))
          cb.assign(skipInvalidLoci, metadata.invoke[Boolean]("skipInvalidLoci"))

          cb.assign(cbfis, Code.newInstance[HadoopFSDataBinaryReader, SeekableDataInputStream](
            mb.getFS.invoke[String, SeekableDataInputStream]("openNoCompression", fileName)))
          index.initialize(cb, indexName)

          cb.assign(currVariantIndex, context.loadField(cb, "first_variant_index").get(cb).asLong.value)
          cb.assign(endVariantIndex, currVariantIndex + context.loadField(cb, "n_variants").get(cb).asLong.value)
        }

        override val elementRegion: Settable[Region] = eltRegion
        override val requiresMemoryManagementPerElement: Boolean = true
        override val LproduceElement: CodeLabel = mb.defineAndImplementLabel { cb =>
          val Lstart = CodeLabel()
          cb.define(Lstart)
          cb.ifx(currVariantIndex ceq endVariantIndex, cb.goto(LendOfStream))

          val addr = index.queryIndex(cb, eltRegion, currVariantIndex)
            .loadField(cb, "offset")
            .get(cb).asLong.value
          cb += cbfis.invoke[Long, Unit]("seek", addr)

          val reqTypeNoUID = if (requestedType.hasField(uidFieldName)) requestedType.deleteKey(uidFieldName) else requestedType
          val e = StagedBGENReader.decodeRow(cb, elementRegion, cbfis, nSamples, fileIdx, compression, skipInvalidLoci, contigRecoding, reqTypeNoUID, rg)
          e.toI(cb).consume(cb, {
            cb += elementRegion.clearRegion()
            cb.goto(Lstart)
          }, { sc =>
            val scUID = if (requestedType.hasField(uidFieldName))
              sc.asBaseStruct.insert(cb, eltRegion, requestedType,
                (uidFieldName, EmitValue.present(SStackStruct.constructFromArgs(cb, eltRegion, uidType,
                  EmitValue.present(primitive(cb.memoize(fileIdx.toL))),
                  EmitValue.present(primitive(currVariantIndex))))))
            else
              sc
            out = mb.newEmitField(scUID.st, true)
            cb.assign(out, EmitCode.present(mb, scUID))
          })

          cb.assign(currVariantIndex, currVariantIndex + 1)
          cb.goto(LproduceElementDone)
        }
        override val element: EmitCode = out

        override def close(cb: EmitCodeBuilder): Unit = {
          cb += cbfis.invoke[Unit]("close")
          index.close(cb)
        }
      })
    }
  }

  def toJValue: JValue = Extraction.decompose(this)(PartitionReader.formats)
}
