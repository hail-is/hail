package is.hail.io.bgen

import is.hail.HailContext
import is.hail.backend.BroadcastValue
import is.hail.backend.spark.SparkBackend
import is.hail.expr.ir
import is.hail.expr.ir.{ExecuteContext, IRParser, IRParserEnvironment, Interpret, MatrixHybridReader, Pretty, TableIR, TableRead, TableValue}
import is.hail.types._
import is.hail.types.physical.{PCanonicalStruct, PStruct, PType}
import is.hail.types.virtual._
import is.hail.io._
import is.hail.io.fs.{FS, FileStatus}
import is.hail.io.index.{IndexReader, IndexReaderBuilder}
import is.hail.io.vcf.LoadVCF
import is.hail.rvd.{RVD, RVDPartitioner, RVDType}
import is.hail.sparkextras.RepartitionedOrderedRDD2
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.Partition
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.json4s.JsonAST.{JArray, JInt, JNull, JString}
import org.json4s.{CustomSerializer, DefaultFormats, Extraction, Formats, JObject, JValue}

import scala.io.Source

case class BgenHeader(
  compressed: Boolean,
  nSamples: Int,
  nVariants: Int,
  headerLength: Int,
  dataStart: Int,
  hasIds: Boolean,
  version: Int,
  fileByteSize: Long,
  path: String
)

case class BgenFileMetadata(
  path: String,
  indexPath: String,
  header: BgenHeader,
  rg: Option[ReferenceGenome],
  contigRecoding: Map[String, String],
  skipInvalidLoci: Boolean,
  nVariants: Long,
  indexKeyType: Type,
  indexAnnotationType: Type,
  rangeBounds: Interval)

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
    val dataStart = allInfoLength + 4

    assert(headerLength <= allInfoLength)
    val nVariants = is.readInt()
    val nSamples = is.readInt()

    val magicNumber = is.readBytes(4).map(_.toInt).toFastIndexedSeq

    if (magicNumber != FastSeq(0, 0, 0, 0) && magicNumber != FastSeq(98, 103, 101, 110))
      fatal(s"expected magic number [0000] or [bgen], got [${ magicNumber.mkString }]")

    if (headerLength > 20)
      is.skipBytes(headerLength.toInt - 20)

    val flags = is.readInt()
    val compressType = flags & 3

    if (compressType != 0 && compressType != 1)
      fatal(s"Hail only supports zlib compression.")

    val isCompressed = compressType != 0

    val version = (flags >>> 2) & 0xf
    if (version != 2)
      fatal(s"Hail supports BGEN version 1.2, got version 1.$version")

    val hasIds = (flags >> 31 & 1) != 0
    BgenHeader(
      isCompressed,
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

  def getAllFileStatuses(fs: FS, files: Array[String]): Array[FileStatus] = {
    val badFiles = new BoxedArrayBuilder[String]()

    val statuses = files.flatMap { file =>
      val matches = fs.glob(file)
      if (matches.isEmpty)
        badFiles += file

      matches.flatMap { status =>
        val file = status.getPath.toString
        if (!file.endsWith(".bgen"))
          warn(s"input file does not have .bgen extension: $file")

        if (fs.isDir(file))
          fs.listStatus(file)
            .filter(status => ".*part-[0-9]+".r.matches(status.getPath.toString))
        else
          Array(status)
      }
    }

    if (!badFiles.isEmpty)
      fatal(
        s"""The following paths refer to no files:
            |  ${ badFiles.result().mkString("\n  ") }""".stripMargin)

    statuses
  }

  def getAllFilePaths(fs: FS, files: Array[String]): Array[String] =
    getAllFileStatuses(fs, files).map(_.getPath.toString)

  def getBgenFileMetadata(ctx: ExecuteContext, files: Array[String], indexFiles: Array[String]): Array[BgenFileMetadata] = {
    val fs = ctx.fs
    require(files.length == indexFiles.length)
    val headers = getFileHeaders(fs, files)
    headers.zip(indexFiles).map { case (h, indexFile) =>
      val (keyType, _) = IndexReader.readTypes(fs, indexFile)
      val rg = keyType.asInstanceOf[TStruct].field("locus").typ match {
        case TLocus(rg) => Some(rg.value)
        case _ => None
      }
      val indexReaderBuilder = {
        val (leafCodec, internalNodeCodec) = BgenSettings.indexCodecSpecs(rg)
        val (leafPType: PStruct, leafDec) = leafCodec.buildDecoder(ctx, leafCodec.encodedVirtualType)
        val (intPType: PStruct, intDec) = internalNodeCodec.buildDecoder(ctx, internalNodeCodec.encodedVirtualType)
        IndexReaderBuilder.withDecoders(leafDec, intDec, BgenSettings.indexKeyType(rg), BgenSettings.indexAnnotationType, leafPType, intPType)
      }
      using(indexReaderBuilder(fs, indexFile, 8, ctx.r.pool)) { index =>
        val attributes = index.attributes
        val rg = Option(attributes("reference_genome")).map(name => ReferenceGenome.getReference(name.asInstanceOf[String]))
        val skipInvalidLoci = attributes("skip_invalid_loci").asInstanceOf[Boolean]
        val contigRecoding = Option(attributes("contig_recoding")).map(_.asInstanceOf[Map[String, String]]).getOrElse(Map.empty[String, String])
        val nVariants = index.nKeys

        val rangeBounds = if (nVariants > 0) {
          val start = index.queryByIndex(0).key
          val end = index.queryByIndex(nVariants - 1).key
          Interval(start, end, includesStart = true, includesEnd = true)
        } else null

        BgenFileMetadata(
          h.path,
          indexFile,
          h,
          rg,
          contigRecoding,
          skipInvalidLoci,
          nVariants,
          index.keyType,
          index.annotationType,
          rangeBounds)
      }
    }
  }

  def getIndexFileNames(fs: FS, files: Array[String], indexFileMap: Map[String, String]): Array[String] = {
    def absolutePath(rel: String): String = fs.fileStatus(rel).getPath.toString

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

  def getReferenceGenome(fileMetadata: Array[BgenFileMetadata]): Option[ReferenceGenome] =
    getReferenceGenome(fileMetadata.map(_.rg))

  def getReferenceGenome(rgs: Array[Option[ReferenceGenome]]): Option[ReferenceGenome] = {
    if (rgs.distinct.length != 1)
      fatal(s"""Found multiple reference genomes were specified in the BGEN index files:
              |  ${ rgs.distinct.map(_.map(_.name).getOrElse("None")).mkString("\n  ") }""".stripMargin)
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
  def fullMatrixType(rg: Option[ReferenceGenome]): MatrixType = {
    MatrixType(
      globalType = TStruct.empty,
      colType = TStruct("s" -> TString),
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

  def fromJValue(env: IRParserEnvironment, jv: JValue): MatrixBGENReader = {
    MatrixBGENReader(env.ctx, MatrixBGENReaderParameters.fromJValue(env, jv))
  }

  def apply(ctx: ExecuteContext,
    files: Seq[String],
    sampleFile: Option[String],
    indexFileMap: Map[String, String],
    nPartitions: Option[Int],
    blockSizeInMB: Option[Int],
    includedVariants: Option[TableIR]): MatrixBGENReader = {
    MatrixBGENReader(ctx,
      MatrixBGENReaderParameters(files, sampleFile, indexFileMap, nPartitions, blockSizeInMB, includedVariants))
  }

  def apply(ctx: ExecuteContext, params: MatrixBGENReaderParameters): MatrixBGENReader = {
    val fs = ctx.fs

    val allFiles = LoadBgen.getAllFilePaths(fs, params.files.toArray)
    val indexFiles = LoadBgen.getIndexFiles(fs, allFiles, params.indexFileMap)
    val fileMetadata = LoadBgen.getBgenFileMetadata(ctx, allFiles, indexFiles)
    assert(fileMetadata.nonEmpty)

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

    val fullMatrixType: MatrixType = MatrixBGENReader.fullMatrixType(referenceGenome)

    val (indexKeyType, indexAnnotationType) = LoadBgen.getIndexTypes(fileMetadata)

    val (maybePartitions, partitionRangeBounds) = BgenRDDPartitions(ctx, referenceGenome, fileMetadata,
      if (params.nPartitions.isEmpty && params.blockSizeInMB.isEmpty)
        Some(128)
      else
        params.blockSizeInMB, params.nPartitions, indexKeyType)
    val partitioner = new RVDPartitioner(indexKeyType.asInstanceOf[TStruct], partitionRangeBounds)

    val (partitions, variants) = params.includedVariants match {
      case Some(variantsTableIR) =>
        val rowType = variantsTableIR.typ.rowType
        assert(rowType.isPrefixOf(fullMatrixType.rowKeyStruct))
        assert(rowType.types.nonEmpty)

        val rvd = Interpret(ir.TableDistinct(variantsTableIR), ctx).rvd

        val repartitioned = RepartitionedOrderedRDD2(rvd, partitionRangeBounds.map(_.coarsen(rowType.types.length)))
          .toRows(rvd.rowPType)
        assert(repartitioned.getNumPartitions == maybePartitions.length)

        (maybePartitions.zipWithIndex.map { case (p, i) =>
          p.asInstanceOf[LoadBgenPartition].copy(filterPartition = repartitioned.partitions(i)).asInstanceOf[Partition]
        }, repartitioned)
      case _ =>
        (maybePartitions, null)
    }

    new MatrixBGENReader(
      params,
      allFiles, referenceGenome, fullMatrixType, indexKeyType, indexAnnotationType, sampleIds, nVariants, partitions, partitioner, variants)
  }
}

object MatrixBGENReaderParameters {
  def fromJValue(env: IRParserEnvironment, jv: JValue): MatrixBGENReaderParameters = {
    implicit val foramts: Formats = DefaultFormats
    val files = (jv \ "files").extract[Array[String]]
    val sampleFile = (jv \ "sampleFile").extractOpt[String]
    val indexFileMap = (jv \ "indexFileMap").extract[Map[String, String]]
    val nPartitions = (jv \ "nPartitions").extractOpt[Int]
    val blockSizeInMB = (jv \ "blockSizeInMB").extractOpt[Int]
    val includedVariantsIR = (jv \ "includedVariants").extractOpt[String].map(IRParser.parse_table_ir(_, env))
    new MatrixBGENReaderParameters(files, sampleFile, indexFileMap, nPartitions, blockSizeInMB, includedVariantsIR)
  }
}

case class MatrixBGENReaderParameters(
  files: Seq[String],
  sampleFile: Option[String],
  indexFileMap: Map[String, String],
  nPartitions: Option[Int],
  blockSizeInMB: Option[Int],
  includedVariants: Option[TableIR]) {

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
      "includedVariants" -> includedVariants.map(t => JString(Pretty(t))).getOrElse(JNull)))
  }
}

class MatrixBGENReader(
  val params: MatrixBGENReaderParameters,
  allFiles: Array[String],
  referenceGenome: Option[ReferenceGenome],
  val fullMatrixType: MatrixType,
  indexKeyType: Type,
  indexAnnotationType: Type,
  sampleIds: Array[String],
  val nVariants: Long,
  partitions: Array[Partition],
  partitioner: RVDPartitioner,
  variants: RDD[Row]) extends MatrixHybridReader {
  def pathsUsed: Seq[String] = allFiles

  private val nSamples = sampleIds.length

  def columnCount: Option[Int] = Some(nSamples)

  def partitionCounts: Option[IndexedSeq[Long]] = None

  private var _settings: BgenSettings = _

  def getSettings(requestedType: TableType): BgenSettings = {
    if (_settings == null || _settings.requestedType != requestedType) {
      _settings = BgenSettings(
        nSamples,
        requestedType,
        referenceGenome.map(_.broadcast),
        indexAnnotationType)
    }
    _settings
  }

  def rowAndGlobalPTypes(context: ExecuteContext, requestedType: TableType): (PStruct, PStruct) = {
    val settings = getSettings(requestedType)
    settings.rowPType -> PType.canonical(requestedType.globalType, required = true).asInstanceOf[PStruct]
  }

  def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    val requestedType = tr.typ

    assert(requestedType.keyType == indexKeyType)

    val settings = getSettings(requestedType)

    val rvdType = RVDType(coerce[PStruct](settings.rowPType.subsetTo(requestedType.rowType)),
      fullType.key.take(requestedType.key.length))

    val rvd = if (tr.dropRows)
      RVD.empty(rvdType)
    else
      new RVD(
        rvdType,
        partitioner,
        BgenRDD(ctx, partitions, settings, variants).toCRDDPtr)

    val globalValue = makeGlobalValue(ctx, requestedType.globalType, sampleIds.map(Row(_)))

    TableValue(ctx, tr.typ, globalValue, rvd)
  }

  override def toJValue: JValue = params.toJValue

  override def hashCode(): Int = params.hashCode()

  override def equals(that: Any): Boolean = that match {
    case that: MatrixBGENReader => params == that.params
    case _ => false
  }
}
