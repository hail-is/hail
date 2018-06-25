package is.hail.io.bgen

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.types._
import is.hail.io.vcf.LoadVCF
import is.hail.io._
import is.hail.rvd.{OrderedRVD, RVDContext}
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.variant._
import org.apache.commons.codec.binary.Base64
import org.apache.hadoop.io.LongWritable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import scala.io.Source

case class BgenHeader(compressed: Boolean, nSamples: Int, nVariants: Int,
  headerLength: Int, dataStart: Int, hasIds: Boolean, version: Int)

case class BgenResult(
  file: String,
  nSamples: Int,
  nVariants: Int,
  rdd: RDD[(LongWritable, BgenRecordV12)]
)

object LoadBgen {
  private[bgen] val includedVariantsPositionsHadoopPrefix = "__includedVariantsPositions__"
  private[bgen] val includedVariantsIndicesHadoopPrefix = "__includedVariantsIndices__"

  def load(hc: HailContext,
    files: Array[String],
    sampleFile: Option[String] = None,
    includeGT: Boolean,
    includeGP: Boolean,
    includeDosage: Boolean,
    includeLid: Boolean,
    includeRsid: Boolean,
    includeFileRowIdx: Boolean,
    nPartitions: Option[Int] = None,
    rg: Option[ReferenceGenome] = Some(ReferenceGenome.defaultReference),
    contigRecoding: Map[String, String] = Map.empty[String, String],
    skipInvalidLoci: Boolean = false,
    includedVariantsPerFile: Map[String, Seq[Int]] = Map.empty[String, Seq[Int]]
  ): MatrixTable = {

    require(files.nonEmpty)
    val hadoop = hc.hadoopConf

    val sampleIds = sampleFile.map(file => LoadBgen.readSampleFile(hadoop, file))
      .getOrElse(LoadBgen.readSamples(hadoop, files.head))

    LoadVCF.warnDuplicates(sampleIds)

    val nSamples = sampleIds.length

    hadoop.setBoolean("includeGT", includeGT)
    hadoop.setBoolean("includeGP", includeGP)
    hadoop.setBoolean("includeDosage", includeDosage)
    hadoop.setBoolean("includeLid", includeLid)
    hadoop.setBoolean("includeRsid", includeRsid)

    val sc = hc.sc
    val results = files.map { file =>
      val bState = readState(sc.hadoopConfiguration, file)

      includedVariantsPerFile.get(file) match {
        case Some(indices) =>
          val variantPositions =
            using(new OnDiskBTreeIndexToValue(file + ".idx", hadoop)) { index =>
              index.positionOfVariants(indices.toArray)
            }
          hadoop.set(includedVariantsPositionsHadoopPrefix + file, encodeLongs(variantPositions))
          hadoop.set(includedVariantsIndicesHadoopPrefix + file, encodeInts(indices.toArray))
        case None =>
          // if import_bgen was previously called, we must clear the old
          // configuration
          hadoop.unset(includedVariantsPositionsHadoopPrefix + file)
          hadoop.unset(includedVariantsIndicesHadoopPrefix + file)
      }

      bState.version match {
        case 2 =>
          BgenResult(file, bState.nSamples, bState.nVariants,
            sc.hadoopFile(file, classOf[BgenInputFormatV12], classOf[LongWritable], classOf[BgenRecordV12], nPartitions.getOrElse(sc.defaultMinPartitions)))
        case x => fatal(s"Hail does not support BGEN v1.$x.")
      }
    }

    val unequalSamples = results.filter(_.nSamples != nSamples).map(x => (x.file, x.nSamples))
    if (unequalSamples.length > 0)
      fatal(
        s"""The following BGEN files did not contain the expected number of samples $nSamples:
            |  ${ unequalSamples.map(x => s"""(${ x._2 } ${ x._1 }""").mkString("\n  ") }""".stripMargin)

    val noVariants = results.filter(_.nVariants == 0).map(_.file)
    if (noVariants.length > 0)
      fatal(
        s"""The following BGEN files did not contain at least 1 variant:
            |  ${ noVariants.mkString("\n  ") })""".stripMargin)

    val nVariants = results.map(_.nVariants).sum

    info(s"Number of BGEN files parsed: ${ results.length }")
    info(s"Number of samples in BGEN files: $nSamples")
    info(s"Number of variants across all BGEN files: $nVariants")

    val rowFields = Array(
      (true, "locus" -> TLocus.schemaFromRG(rg)),
      (true, "alleles" -> TArray(TString())),
      (includeRsid, "rsid" -> TString()),
      (includeLid, "varid" -> TString()),
      (includeFileRowIdx, "file_row_idx" -> TInt64()))
      .withFilter(_._1).map(_._2)

    val signature = TStruct(rowFields: _*)

    val entryFields = Array(
      (includeGT, "GT" -> TCall()),
      (includeGP, "GP" -> +TArray(+TFloat64())),
      (includeDosage, "dosage" -> +TFloat64()))
      .withFilter(_._1).map(_._2)

    val matrixType: MatrixType = MatrixType.fromParts(
      globalType = TStruct.empty(),
      colKey = Array("s"),
      colType = TStruct("s" -> TString()),
      rowType = signature,
      rowKey = Array("locus", "alleles"),
      rowPartitionKey = Array("locus"),
      entryType = TStruct(entryFields: _*))

    val kType = matrixType.orvdType.kType
    val rowType = matrixType.rvRowType

    val crdds = results.map(x => ContextRDD.weaken[RVDContext](x.rdd))

    val fastKeys = ContextRDD.union(sc, crdds.map(_.cmapPartitions { (ctx, it) =>
      val region = ctx.region
      val rvb = new RegionValueBuilder(region)
      val rv = RegionValue(region)

      it.flatMap { case (_, record) =>
        val contig = record.getContig
        val pos = record.getPosition
        val alleles = record.getAlleles
        val contigRecoded = contigRecoding.getOrElse(contig, contig)

        if (skipInvalidLoci && !rg.forall(_.isValidLocus(contigRecoded, pos)))
          None
        else {
          rvb.start(kType)
          rvb.startStruct()
          rvb.addAnnotation(kType.types(0), Locus.annotation(contigRecoded, pos, rg))

          val nAlleles = alleles.length
          rvb.startArray(nAlleles)
          var i = 0
          while (i < nAlleles) {
            rvb.addString(alleles(i))
            i += 1
          }
          rvb.endArray()
          rvb.endStruct()

          rv.setOffset(rvb.end())
          Some(rv)
        }
      }
    }))

    val rdd2 = ContextRDD.union(sc, crdds.map(_.cmapPartitions { (ctx, it) =>
      val region = ctx.region
      val rvb = new RegionValueBuilder(region)
      val rv = RegionValue(region)

      it.flatMap { case (_, record) =>
        val contig = record.getContig
        val pos = record.getPosition
        val alleles = record.getAlleles

        val contigRecoded = contigRecoding.getOrElse(contig, contig)

        if (skipInvalidLoci && !rg.forall(_.isValidLocus(contigRecoded, pos)))
          None
        else {
          rvb.start(rowType)
          rvb.startStruct()
          rvb.addAnnotation(kType.types(0), Locus.annotation(contigRecoded, pos, rg))

          val nAlleles = alleles.length
          rvb.startArray(nAlleles)
          var i = 0
          while (i < nAlleles) {
            rvb.addString(alleles(i))
            i += 1
          }
          rvb.endArray()

          if (includeRsid)
            rvb.addString(record.getRsid)
          if (includeLid)
            rvb.addString(record.getLid)
          if (includeFileRowIdx)
            rvb.addLong(record.getFileRowIdx)
          record.getValue(rvb) // gs

          rvb.endStruct()
          rv.setOffset(rvb.end())
          Some(rv)
        }
      }
    }))

    new MatrixTable(hc, matrixType,
      BroadcastRow(Row.empty, matrixType.globalType, sc),
      BroadcastIndexedSeq(sampleIds.map(x => Annotation(x)), TArray(matrixType.colType), sc),
      OrderedRVD.coerce(matrixType.orvdType, rdd2, Some(fastKeys), None))
  }

  def index(hConf: org.apache.hadoop.conf.Configuration, file: String) {
    val indexFile = file + ".idx"

    val bState = readState(hConf, file)

    val dataBlockStarts = new Array[Long](bState.nVariants + 1)
    var position: Long = bState.dataStart

    dataBlockStarts(0) = position

    hConf.readFile(file) { is =>
      val reader = new HadoopFSDataBinaryReader(is)
      reader.seek(0)

      for (i <- 1 to bState.nVariants) {
        reader.seek(position)

        if (bState.version == 1)
          reader.readInt() // nRows for v1.1 only

        val snpid = reader.readLengthAndString(2)
        val rsid = reader.readLengthAndString(2)
        val chr = reader.readLengthAndString(2)
        val pos = reader.readInt()

        val nAlleles = if (bState.version == 2) reader.readShort() else 2
        assert(nAlleles >= 2, s"Number of alleles must be greater than or equal to 2. Found $nAlleles alleles for variant '$snpid'")
        (0 until nAlleles).foreach { i => reader.readLengthAndString(4) }

        position = bState.version match {
          case 1 =>
            if (bState.compressed)
              reader.readInt() + reader.getPosition
            else
              reader.getPosition + 6 * bState.nSamples
          case 2 =>
            reader.readInt() + reader.getPosition
        }

        dataBlockStarts(i) = position
      }
    }

    IndexBTree.write(dataBlockStarts, indexFile, hConf)

  }

  def readSamples(hConf: org.apache.hadoop.conf.Configuration, file: String): Array[String] = {
    val bState = readState(hConf, file)
    if (bState.hasIds) {
      hConf.readFile(file) { is =>
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
      warn(s"BGEN file `$file' contains no sample ID block and no sample ID file given.\n" +
        s"  Using _0, _1, ..., _N as sample IDs.")
      (0 until bState.nSamples).map(i => s"_$i").toArray
    }
  }

  def readSampleFile(hConf: org.apache.hadoop.conf.Configuration, file: String): Array[String] = {
    hConf.readFile(file) { s =>
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

  def readState(hConf: org.apache.hadoop.conf.Configuration, file: String): BgenHeader = {
    hConf.readFile(file) { is =>
      val reader = new HadoopFSDataBinaryReader(is)
      readState(reader)
    }
  }

  def readState(reader: HadoopFSDataBinaryReader): BgenHeader = {
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
    if (version != 1 && version != 2)
      fatal(s"Hail supports BGEN version 1.1 and 1.2, got version 1.$version")

    val hasIds = (flags >> 31 & 1) != 0
    BgenHeader(isCompressed, nSamples, nVariants, headerLength, dataStart, hasIds, version)
  }

  private[bgen] def encodeInts(a: Array[Int]): String = {
    val b = new Array[Byte](a.length * 4)
    var i = 0
    while (i < a.length) {
      b(4 * i) = (a(i) & 0xff).asInstanceOf[Byte]
      b(4 * i + 1) = ((a(i) >>> 8) & 0xff).asInstanceOf[Byte]
      b(4 * i + 2) = ((a(i) >>> 16) & 0xff).asInstanceOf[Byte]
      b(4 * i + 3) = ((a(i) >>> 24) & 0xff).asInstanceOf[Byte]
      i += 1
    }
    Base64.encodeBase64String(b)
  }

  private[bgen] def decodeInts(a: String): Array[Int] = {
    val b = Base64.decodeBase64(a)
    val c = new Array[Int](b.length / 4)
    var i = 0
    while (i < c.length) {
      c(i) =
        ((b(i * 4 + 3) & 0xff).asInstanceOf[Int] << 24) |
        ((b(i * 4 + 2) & 0xff).asInstanceOf[Int] << 16) |
        ((b(i * 4 + 1) & 0xff).asInstanceOf[Int] << 8) |
        (b(i * 4).asInstanceOf[Int] & 0xff)
      i += 1
    }
    c
  }

  private[bgen] def encodeLongs(a: Array[Long]): String = {
    val b = new Array[Byte](a.length * 8)
    var i = 0
    while (i < a.length) {
      b(8 * i) = (a(i) & 0xff).asInstanceOf[Byte]
      b(8 * i + 1) = ((a(i) >>> 8) & 0xff).asInstanceOf[Byte]
      b(8 * i + 2) = ((a(i) >>> 16) & 0xff).asInstanceOf[Byte]
      b(8 * i + 3) = ((a(i) >>> 24) & 0xff).asInstanceOf[Byte]
      b(8 * i + 4) = ((a(i) >>> 32) & 0xff).asInstanceOf[Byte]
      b(8 * i + 5) = ((a(i) >>> 40) & 0xff).asInstanceOf[Byte]
      b(8 * i + 6) = ((a(i) >>> 48) & 0xff).asInstanceOf[Byte]
      b(8 * i + 7) = ((a(i) >>> 56) & 0xff).asInstanceOf[Byte]
      i += 1
    }
    Base64.encodeBase64String(b)
  }

  private[bgen] def decodeLongs(a: String): Array[Long] = {
    val b = Base64.decodeBase64(a)
    val c = new Array[Long](b.length / 8)
    var i = 0
    while (i < c.length) {
      c(i) =
        ((b(i * 8 + 7) & 0xff).asInstanceOf[Long] << 56) |
        ((b(i * 8 + 6) & 0xff).asInstanceOf[Long] << 48) |
        ((b(i * 8 + 5) & 0xff).asInstanceOf[Long] << 40) |
        ((b(i * 8 + 4) & 0xff).asInstanceOf[Long] << 32) |
        ((b(i * 8 + 3) & 0xff).asInstanceOf[Long] << 24) |
        ((b(i * 8 + 2) & 0xff).asInstanceOf[Long] << 16) |
        ((b(i * 8 + 1) & 0xff).asInstanceOf[Long] << 8) |
        (b(i * 8).asInstanceOf[Long] & 0xff)
      i += 1
    }
    c
  }
}
