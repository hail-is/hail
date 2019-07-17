package is.hail.io.bgen

import is.hail.HailContext
import is.hail.expr.ir
import is.hail.expr.ir.{ExecuteContext, IRParser, IRParserEnvironment, Interpret, MatrixHybridReader, Pretty, TableIR, TableRead, TableValue}
import is.hail.expr.types._
import is.hail.expr.types.virtual._
import is.hail.io._
import is.hail.io.fs.{FS, FileStatus}
import is.hail.io.index.IndexReader
import is.hail.io.vcf.LoadVCF
import is.hail.rvd.{RVD, RVDPartitioner}
import is.hail.sparkextras.RepartitionedOrderedRDD2
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.Partition
import org.apache.spark.sql.Row
import org.json4s.JsonAST.{JArray, JInt, JNull, JString}
import org.json4s.{CustomSerializer, JObject}

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
  rangeBounds: Interval
)

object LoadBgen {
  def readSamples(fs: is.hail.io.fs.FS, file: String): Array[String] = {
    val bState = readState(fs, file)
    if (bState.hasIds) {
      fs.readFile(file) { is =>
        val reader = new HadoopFSDataBinaryReader(is)

        reader.seek(bState.headerLength + 4)
        val sampleIdSize = reader.readInt()
        val nSamples = reader.readInt()

        if (nSamples != bState.nSamples)
          fatal("BGEN file is malformed -- number of sample IDs in header does not equal number in file")

        if (sampleIdSize + bState.headerLength > bState.dataStart - 4)
          fatal("BGEN file is malformed -- offset is smaller than length of header")

        (0 until nSamples).map { i =>
          reader.readLengthAndString(2)
        }.toArray
      }
    } else {
      warn(s"BGEN file '$file' contains no sample ID block and no sample ID file given.\n" +
        s"  Using _0, _1, ..., _N as sample IDs.")
      (0 until bState.nSamples).map(i => s"_$i").toArray
    }
  }

  def readSampleFile(fs: is.hail.io.fs.FS, file: String): Array[String] = {
    fs.readFile(file) { s =>
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

  def readState(fs: is.hail.io.fs.FS, file: String): BgenHeader = {
    fs.readFile(file) { is =>
      val reader = new HadoopFSDataBinaryReader(is)
      readState(reader, file, fs.getFileSize(file))
    }
  }

  def readState(reader: HadoopFSDataBinaryReader, path: String, byteSize: Long): BgenHeader = {
    reader.seek(0)
    val allInfoLength = reader.readInt()
    val headerLength = reader.readInt()
    val dataStart = allInfoLength + 4

    assert(headerLength <= allInfoLength)
    val nVariants = reader.readInt()
    val nSamples = reader.readInt()

    val magicNumber = reader.readBytes(4)
      .map(_.toInt)
      .toSeq

    if (magicNumber != FastSeq(0, 0, 0, 0) && magicNumber != FastSeq(98, 103, 101, 110))
      fatal(s"expected magic number [0000] or [bgen], got [${ magicNumber.mkString }]")

    if (headerLength > 20)
      reader.skipBytes(headerLength.toInt - 20)

    val flags = reader.readInt()
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
    val badFiles = new ArrayBuilder[String]()

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

  def getBgenFileMetadata(fs: FS, files: Array[String], indexFiles: Array[String]): Array[BgenFileMetadata] = {
    require(files.length == indexFiles.length)
    val headers = getFileHeaders(fs, files)
    headers.zip(indexFiles).map { case (h, indexFile) =>
      using(IndexReader(fs, indexFile)) { index =>
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
          rangeBounds
        )
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

class MatrixBGENReaderSerializer(env: IRParserEnvironment) extends CustomSerializer[MatrixBGENReader](
  format =>
  ({ case jObj: JObject =>
    implicit val fmt = format
    val files = (jObj \ "files").extract[Array[String]]
    val sampleFile = (jObj \ "sampleFile").extractOpt[String]
    val indexFileMap = (jObj \ "indexFileMap").extract[Map[String, String]]
    val nPartitions = (jObj \ "nPartitions").extractOpt[Int]
    val blockSizeInMB = (jObj \ "blockSizeInMB").extractOpt[Int]
    val includedVariantsIR = (jObj \ "includedVariants").extractOpt[String].map(IRParser.parse_table_ir(_, env))
    MatrixBGENReader(files, sampleFile, indexFileMap, nPartitions, blockSizeInMB, includedVariantsIR)
  }, { case reader: MatrixBGENReader =>
    JObject(List(
      "files" -> JArray(reader.files.map(JString).toList),
      "sampleFile" -> reader.sampleFile.map(JString).getOrElse(JNull),
      "indexFileMap" -> JArray(reader.indexFileMap.map { case (k, v) => JObject(
        "key" -> JString(k), "value" -> JString(v)
      )}.toList),
      "nPartitions" -> reader.nPartitions.map(JInt(_)).getOrElse(JNull),
      "blockSizeInMB" -> reader.blockSizeInMB.map(JInt(_)).getOrElse(JNull),
      "includedVariants" -> reader.includedVariants.map(t => JString(Pretty(t))).getOrElse(JNull)
    ))
  })
)

object MatrixBGENReader {
  def fullMatrixType(rg: Option[ReferenceGenome]): MatrixType = {
    MatrixType(
      globalType = TStruct.empty(),
      colType = TStruct("s" -> TString()),
      colKey = Array("s"),
      rowType = TStruct(
        "locus" -> TLocus.schemaFromRG(rg),
        "alleles" -> TArray(TString()),
        "rsid" -> TString(),
        "varid" -> TString(),
        "offset" -> TInt64(),
        "file_idx" -> TInt32()),
      rowKey = Array("locus", "alleles"),
      entryType = TStruct(
        "GT" -> TCall(),
        "GP" -> TArray(TFloat64Required, required = true),
        "dosage" -> TFloat64Required
      )
    )
  }
}

case class MatrixBGENReader(
  files: Seq[String],
  sampleFile: Option[String],
  indexFileMap: Map[String, String],
  nPartitions: Option[Int],
  blockSizeInMB: Option[Int],
  includedVariants: Option[TableIR]) extends MatrixHybridReader {
  private val hc = HailContext.get
  private val sc = hc.sc
  private val fs = hc.sFS

  val allFiles = LoadBgen.getAllFilePaths(fs, files.toArray)
  val indexFiles = LoadBgen.getIndexFiles(fs, allFiles, indexFileMap)
  val fileMetadata = LoadBgen.getBgenFileMetadata(fs, allFiles, indexFiles)
  assert(fileMetadata.nonEmpty)

  private val sampleIds = sampleFile.map(file => LoadBgen.readSampleFile(fs, file))
    .getOrElse(LoadBgen.readSamples(fs, fileMetadata.head.path))

  LoadVCF.warnDuplicates(sampleIds)

  private val nSamples = sampleIds.length

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

  private val referenceGenome = LoadBgen.getReferenceGenome(fileMetadata)

  val fullMatrixType: MatrixType = MatrixBGENReader.fullMatrixType(referenceGenome)

  val (indexKeyType, indexAnnotationType) = LoadBgen.getIndexTypes(fileMetadata)

  val (maybePartitions, partitionRangeBounds) = BgenRDDPartitions(sc, fileMetadata,
    if (nPartitions.isEmpty && blockSizeInMB.isEmpty)
    Some(128)
  else
    blockSizeInMB, nPartitions, indexKeyType)
  val partitioner = new RVDPartitioner(indexKeyType.asInstanceOf[TStruct], partitionRangeBounds)

  val (partitions, variants) = includedVariants match {
    case Some(variantsTableIR) =>
      val rowType = variantsTableIR.typ.rowType
      assert(rowType.isPrefixOf(fullMatrixType.rowKeyStruct))
      assert(rowType.types.nonEmpty)

      val rvd = ExecuteContext.scoped { ctx =>
        Interpret(ir.TableDistinct(variantsTableIR), ctx).rvd
      }

      val repartitioned = RepartitionedOrderedRDD2(rvd, partitionRangeBounds.map(_.coarsen(rowType.types.length)))
        .toRows(rowType.physicalType)
      assert(repartitioned.getNumPartitions == maybePartitions.length)

      (maybePartitions.zipWithIndex.map { case (p, i) =>
        p.asInstanceOf[LoadBgenPartition].copy(filterPartition = repartitioned.partitions(i)).asInstanceOf[Partition]
      }, repartitioned)
    case _ => (maybePartitions, null)
  }

  def columnCount: Option[Int] = Some(nSamples)

  def partitionCounts: Option[IndexedSeq[Long]] = None


  def apply(tr: TableRead, ctx: ExecuteContext): TableValue = {
    require(files.nonEmpty)

    val requestedType = tr.typ

    assert(requestedType.keyType == indexKeyType)

    val settings = BgenSettings(
      nSamples,
      requestedType,
      referenceGenome.map(_.broadcast),
      indexAnnotationType)

    val rvd = if (tr.dropRows)
      RVD.empty(sc, requestedType.canonicalRVDType)
    else
      new RVD(requestedType.canonicalRVDType,
        partitioner,
        BgenRDD(sc, partitions, settings, variants))

    val globalValue = makeGlobalValue(ctx, requestedType, sampleIds.map(Row(_)))

    TableValue(tr.typ, globalValue, rvd)
  }
}
