package is.hail

import java.io.InputStream
import java.util.Properties

import is.hail.annotations._
import is.hail.expr.ir.MatrixRead
import is.hail.expr.types._
import is.hail.io.{CodecSpec, Decoder, LoadMatrix}
import is.hail.io.bgen.{LoadBgen, MatrixBGENReader}
import is.hail.io.gen.LoadGen
import is.hail.io.plink.{FamFileConfig, LoadPlink}
import is.hail.io.vcf._
import is.hail.rvd.RVDContext
import is.hail.table.Table
import is.hail.sparkextras.ContextRDD
import is.hail.utils.{log, _}
import is.hail.variant.{MatrixTable, ReferenceGenome, VSMSubgen}
import org.apache.hadoop
import org.apache.log4j.{ConsoleAppender, LogManager, PatternLayout, PropertyConfigurator}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark._
import org.apache.spark.executor.InputMetrics

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.language.existentials
import scala.reflect.ClassTag

case class FilePartition(index: Int, file: String) extends Partition

object HailContext {
  val tera: Long = 1024L * 1024L * 1024L * 1024L

  val logFormat: String = "%d{yyyy-MM-dd HH:mm:ss} %c{1}: %p: %m%n"

  private val contextLock = new Object()

  private var theContext: HailContext = _

  def get: HailContext = contextLock.synchronized { theContext }

  def checkSparkCompatibility(jarVersion: String, sparkVersion: String): Unit = {
    def majorMinor(version: String): String = version.split("\\.", 3).take(2).mkString(".")

    if (majorMinor(jarVersion) != majorMinor(sparkVersion))
      fatal(s"This Hail JAR was compiled for Spark $jarVersion, cannot run with Spark $sparkVersion.\n" +
        s"  The major and minor versions must agree, though the patch version can differ.")
    else if (jarVersion != sparkVersion)
      warn(s"This Hail JAR was compiled for Spark $jarVersion, running with Spark $sparkVersion.\n" +
        s"  Compatibility is not guaranteed.")
  }

  def configureAndCreateSparkContext(appName: String, master: Option[String],
    local: String, blockSize: Long): SparkContext = {
    require(blockSize >= 0)
    checkSparkCompatibility(is.hail.HAIL_SPARK_VERSION, org.apache.spark.SPARK_VERSION)

    val conf = new SparkConf().setAppName(appName)

    master match {
      case Some(m) =>
        conf.setMaster(m)
      case None =>
        if (!conf.contains("spark.master"))
          conf.setMaster(local)
    }

    conf.set("spark.logConf", "true")
    conf.set("spark.ui.showConsoleProgress", "false")

    conf.set(
      "spark.hadoop.io.compression.codecs",
      "org.apache.hadoop.io.compress.DefaultCodec," +
        "is.hail.io.compress.BGzipCodec," +
        "org.apache.hadoop.io.compress.GzipCodec")

    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.kryo.registrator", "is.hail.kryo.HailKryoRegistrator")

    conf.set("spark.hadoop.mapreduce.input.fileinputformat.split.minsize", (blockSize * 1024L * 1024L).toString)

    // load additional Spark properties from HAIL_SPARK_PROPERTIES
    val hailSparkProperties = System.getenv("HAIL_SPARK_PROPERTIES")
    if (hailSparkProperties != null) {
      hailSparkProperties
        .split(",")
        .foreach { p =>
          p.split("=") match {
            case Array(k, v) =>
              log.info(s"set Spark property from HAIL_SPARK_PROPERTIES: $k=$v")
              conf.set(k, v)
            case _ =>
              warn(s"invalid key-value property pair in HAIL_SPARK_PROPERTIES: $p")
          }
        }
    }

    val sc = new SparkContext(conf)
    sc
  }

  def checkSparkConfiguration(sc: SparkContext) {
    val conf = sc.getConf

    val problems = new ArrayBuffer[String]

    val serializer = conf.getOption("spark.serializer")
    val kryoSerializer = "org.apache.spark.serializer.KryoSerializer"
    if (!serializer.contains(kryoSerializer))
      problems += s"Invalid configuration property spark.serializer: required $kryoSerializer.  " +
        s"Found: ${ serializer.getOrElse("empty parameter") }."

    if (!conf.getOption("spark.kryo.registrator").exists(_.split(",").contains("is.hail.kryo.HailKryoRegistrator")))
      problems += s"Invalid config parameter: spark.kryo.registrator must include is.hail.kryo.HailKryoRegistrator." +
        s"Found ${ conf.getOption("spark.kryo.registrator").getOrElse("empty parameter.") }"

    if (problems.nonEmpty)
      fatal(
        s"""Found problems with SparkContext configuration:
           |  ${ problems.mkString("\n  ") }""".stripMargin)
  }

  def configureLogging(logFile: String, quiet: Boolean, append: Boolean) {
    val logProps = new Properties()

    logProps.put("log4j.rootLogger", "INFO, logfile")
    logProps.put("log4j.appender.logfile", "org.apache.log4j.FileAppender")
    logProps.put("log4j.appender.logfile.append", append.toString)
    logProps.put("log4j.appender.logfile.file", logFile)
    logProps.put("log4j.appender.logfile.threshold", "INFO")
    logProps.put("log4j.appender.logfile.layout", "org.apache.log4j.PatternLayout")
    logProps.put("log4j.appender.logfile.layout.ConversionPattern", HailContext.logFormat)

    LogManager.resetConfiguration()
    PropertyConfigurator.configure(logProps)

    if (!quiet)
      consoleLog.addAppender(new ConsoleAppender(new PatternLayout(HailContext.logFormat), "System.err"))
  }

  /**
   * If a HailContext has already been initialized, this function returns it regardless of the
   * parameters with which it was initialized.
   *
   * Otherwise, it initializes and returns a new HailContext.
   */
  def getOrCreate(sc: SparkContext = null,
    appName: String = "Hail",
    master: Option[String] = None,
    local: String = "local[*]",
    logFile: String = "hail.log",
    quiet: Boolean = false,
    append: Boolean = false,
    minBlockSize: Long = 1L,
    branchingFactor: Int = 50,
    tmpDir: String = "/tmp"): HailContext = contextLock.synchronized {

    if (get != null) {
      val hc = get
      if (sc == null) {
        warn("Requested that Hail be initialized with a new SparkContext, but Hail " +
          "has already been initialized. Different configuration settings will be ignored.")
      }
      val paramsDiff = (Map(
        "tmpDir" -> Seq(tmpDir, hc.tmpDir),
        "branchingFactor" -> Seq(branchingFactor, hc.branchingFactor),
        "minBlockSize" -> Seq(minBlockSize, hc.sc.getConf.getLong("spark.hadoop.mapreduce.input.fileinputformat.split.minsize", 0L) / 1024L / 1024L)
       ) ++ master.map(m => "master" -> Seq(m, hc.sc.master))).filter(_._2.areDistinct())
      val paramsDiffStr = paramsDiff.map { case (name, Seq(provided, existing)) =>
        s"Param: $name, Provided value: $provided, Existing value: $existing"
      }.mkString("\n")
      if (paramsDiff.nonEmpty) {
        warn("Found differences between requested and initialized parameters. Ignoring requested " +
          s"parameters.\n$paramsDiffStr")
      }

      hc
    } else {
      apply(sc, appName, master, local, logFile, quiet, append, minBlockSize, branchingFactor,
        tmpDir)
    }
  }

  def apply(sc: SparkContext = null,
    appName: String = "Hail",
    master: Option[String] = None,
    local: String = "local[*]",
    logFile: String = "hail.log",
    quiet: Boolean = false,
    append: Boolean = false,
    minBlockSize: Long = 1L,
    branchingFactor: Int = 50,
    tmpDir: String = "/tmp"): HailContext = contextLock.synchronized {
    require(theContext == null)

    val javaVersion = raw"(\d+)\.(\d+)\.(\d+).*".r
    val versionString = System.getProperty("java.version")
    versionString match {
      // old-style version: 1.MAJOR.MINOR
      // new-style version: MAJOR.MINOR.SECURITY (started in JRE 9)
      // see: https://docs.oracle.com/javase/9/migrate/toc.htm#JSMIG-GUID-3A71ECEF-5FC5-46FE-9BA9-88CBFCE828CB
      case javaVersion("1", major, minor) =>
        if (major.toInt < 8)
          fatal(s"Hail requires at least Java 1.8, found $versionString")
      case javaVersion(major, minor, security) =>
      case _ =>
        fatal(s"Unknown JVM version string: $versionString")
    }

    {
      import breeze.linalg._
      import breeze.linalg.operators.{BinaryRegistry, OpMulMatrix}

      implicitly[BinaryRegistry[DenseMatrix[Double], Vector[Double], OpMulMatrix.type, DenseVector[Double]]].register(
        DenseMatrix.implOpMulMatrix_DMD_DVD_eq_DVD)
    }

    configureLogging(logFile, quiet, append)

    val sparkContext = if (sc == null)
      configureAndCreateSparkContext(appName, master, local, minBlockSize)
    else {
      checkSparkConfiguration(sc)
      sc
    }

    sparkContext.hadoopConfiguration.set("io.compression.codecs",
      "org.apache.hadoop.io.compress.DefaultCodec," +
        "is.hail.io.compress.BGzipCodec," +
        "org.apache.hadoop.io.compress.GzipCodec"
    )

    if (!quiet)
      ProgressBarBuilder.build(sparkContext)

    val sqlContext = new org.apache.spark.sql.SQLContext(sparkContext)
    val hailTempDir = TempDir.createTempDir(tmpDir, sparkContext.hadoopConfiguration)
    val hc = new HailContext(sparkContext, sqlContext, hailTempDir, branchingFactor)
    sparkContext.uiWebUrl.foreach(ui => info(s"SparkUI: $ui"))

    info(s"Running Hail version ${ hc.version }")
    theContext = hc
    hc
  }

  def clear() {
    theContext = null
  }

  def startProgressBar(sc: SparkContext) {
    ProgressBarBuilder.build(sc)
  }

  def readRowsPartition(
    makeDec: (InputStream) => Decoder
  )(ctx: RVDContext,
    in: InputStream,
    metrics: InputMetrics = null
  ): Iterator[RegionValue] =
    new Iterator[RegionValue] {
      private val region = ctx.region
      private val rv = RegionValue(region)

      private val trackedIn = new ByteTrackingInputStream(in)
      private val dec =
        try {
          makeDec(trackedIn)
        } catch {
          case e: Exception =>
            in.close()
            throw e
        }

      private var cont: Byte = dec.readByte()
      if (cont == 0)
        dec.close()

      // can't throw
      def hasNext: Boolean = cont != 0

      def next(): RegionValue = {
        // !hasNext => cont == 0 => dec has been closed
        if (!hasNext)
          throw new NoSuchElementException("next on empty iterator")

        try {
          rv.setOffset(dec.readRegionValue(region))
          cont = dec.readByte()
          if (metrics != null) {
            ExposedMetrics.incrementRecord(metrics)
            ExposedMetrics.incrementBytes(metrics, trackedIn.bytesReadAndClear())
          }

          if (cont == 0)
            dec.close()

          rv
        } catch {
          case e: Exception =>
            dec.close()
            throw e
        }
      }

      override def finalize(): Unit = {
        dec.close()
      }
    }
}

class HailContext private(val sc: SparkContext,
  val sqlContext: SQLContext,
  val tmpDir: String,
  val branchingFactor: Int) {
  val hadoopConf: hadoop.conf.Configuration = sc.hadoopConfiguration

  def version: String = is.hail.HAIL_PRETTY_VERSION

  def grep(regex: String, files: Seq[String], maxLines: Int = 100) {
    val regexp = regex.r
    sc.textFilesLines(hadoopConf.globAll(files))
      .filter(line => regexp.findFirstIn(line.value).isDefined)
      .take(maxLines)
      .groupBy(_.source.asInstanceOf[Context].file)
      .foreach { case (file, lines) =>
        info(s"$file: ${ lines.length } ${ plural(lines.length, "match", "matches") }:")
        lines.map(_.value).foreach { line =>
          val (screen, logged) = line.truncatable().strings
          log.info("\t" + logged)
          println(s"\t$screen")
        }
      }
  }

  def getTemporaryFile(nChar: Int = 10, prefix: Option[String] = None, suffix: Option[String] = None): String =
    sc.hadoopConfiguration.getTemporaryFile(tmpDir, nChar, prefix, suffix)

  def importBgens(files: Seq[String],
    sampleFile: Option[String] = None,
    includeGT: Boolean = true,
    includeGP: Boolean = true,
    includeDosage: Boolean = false,
    includeLid: Boolean = true,
    includeRsid: Boolean = true,
    includeFileRowIdx: Boolean = false,
    nPartitions: Option[Int] = None,
    blockSizeInMB: Option[Int] = None,
    rg: Option[String] = None,
    contigRecoding: Map[String, String] = null,
    skipInvalidLoci: Boolean = false,
    includedVariantsPerUnresolvedFilePath: Map[String, Seq[Int]] = Map.empty
  ): MatrixTable = {
    val referenceGenome = rg.map(ReferenceGenome.getReference)

    val typedRowFields = Array(
      (true, "locus" -> TLocus.schemaFromRG(referenceGenome)),
      (true, "alleles" -> TArray(TString())),
      (includeRsid, "rsid" -> TString()),
      (includeLid, "varid" -> TString()),
      (includeFileRowIdx, "file_row_idx" -> TInt64()))
      .withFilter(_._1).map(_._2)

    val typedEntryFields: Array[(String, Type)] =
      Array(
        (includeGT, "GT" -> TCall()),
        (includeGP, "GP" -> +TArray(+TFloat64())),
        (includeDosage, "dosage" -> +TFloat64()))
        .withFilter(_._1).map(_._2)

    val requestedType: MatrixType = MatrixType.fromParts(
      globalType = TStruct.empty(),
      colKey = Array("s"),
      colType = TStruct("s" -> TString()),
      rowType = TStruct(typedRowFields: _*),
      rowKey = Array("locus", "alleles"),
      rowPartitionKey = Array("locus"),
      entryType = TStruct(typedEntryFields: _*))

    val reader = MatrixBGENReader(
      files, sampleFile, nPartitions,
      if (nPartitions.isEmpty && blockSizeInMB.isEmpty)
        Some(128)
      else
        blockSizeInMB,
      rg,
      Option(contigRecoding).getOrElse(Map.empty[String, String]),
      skipInvalidLoci,
      includedVariantsPerUnresolvedFilePath)
    new MatrixTable(this, MatrixRead(requestedType, dropCols = false, dropRows = false, reader))
  }

  def importGen(file: String,
    sampleFile: String,
    chromosome: Option[String] = None,
    nPartitions: Option[Int] = None,
    tolerance: Double = 0.2,
    rg: Option[ReferenceGenome] = Some(ReferenceGenome.defaultReference),
    contigRecoding: Option[Map[String, String]] = None,
    skipInvalidLoci: Boolean = false): MatrixTable = {
    importGens(List(file), sampleFile, chromosome, nPartitions, tolerance, rg, contigRecoding, skipInvalidLoci)
  }

  def importGens(files: Seq[String],
    sampleFile: String,
    chromosome: Option[String] = None,
    nPartitions: Option[Int] = None,
    tolerance: Double = 0.2,
    rg: Option[ReferenceGenome] = Some(ReferenceGenome.defaultReference),
    contigRecoding: Option[Map[String, String]] = None,
    skipInvalidLoci: Boolean = false): MatrixTable = {
    val inputs = hadoopConf.globAll(files)

    inputs.foreach { input =>
      if (!hadoopConf.stripCodec(input).endsWith(".gen"))
        fatal(s"gen inputs must end in .gen[.bgz], found $input")
    }

    if (inputs.isEmpty)
      fatal(s"arguments refer to no files: ${ files.mkString(",") }")

    rg.foreach(ref => contigRecoding.foreach(ref.validateContigRemap))

    val samples = LoadBgen.readSampleFile(sc.hadoopConfiguration, sampleFile)
    val nSamples = samples.length

    //FIXME: can't specify multiple chromosomes
    val results = inputs.map(f => LoadGen(f, sampleFile, sc, rg, nPartitions,
      tolerance, chromosome, contigRecoding.getOrElse(Map.empty[String, String]), skipInvalidLoci))

    val unequalSamples = results.filter(_.nSamples != nSamples).map(x => (x.file, x.nSamples))
    if (unequalSamples.length > 0)
      fatal(
        s"""The following GEN files did not contain the expected number of samples $nSamples:
           |  ${ unequalSamples.map(x => s"""(${ x._2 } ${ x._1 }""").mkString("\n  ") }""".stripMargin)

    val noVariants = results.filter(_.nVariants == 0).map(_.file)
    if (noVariants.length > 0)
      fatal(
        s"""The following GEN files did not contain at least 1 variant:
           |  ${ noVariants.mkString("\n  ") })""".stripMargin)

    val nVariants = results.map(_.nVariants).sum

    info(s"Number of GEN files parsed: ${ results.length }")
    info(s"Number of variants in all GEN files: $nVariants")
    info(s"Number of samples in GEN files: $nSamples")

    val signature = TStruct(
      "locus" -> TLocus.schemaFromRG(rg),
      "alleles" -> TArray(TString()),
      "rsid" -> TString(), "varid" -> TString())

    val rdd = sc.union(results.map(_.rdd))

    MatrixTable.fromLegacy(this,
      MatrixType.fromParts(
        globalType = TStruct.empty(),
        colKey = Array("s"),
        colType = TStruct("s" -> TString()),
        rowPartitionKey = Array("locus"), rowKey = Array("locus", "alleles"),
        rowType = signature,
        entryType = TStruct("GT" -> TCall(),
          "GP" -> TArray(TFloat64()))),
      Annotation.empty,
      samples.map(Annotation(_)),
      rdd)
  }

  def importTable(inputs: java.util.ArrayList[String],
    keyNames: java.util.ArrayList[String],
    nPartitions: java.lang.Integer,
    types: java.util.HashMap[String, Type],
    comment: java.util.ArrayList[String],
    separator: String,
    missing: String,
    noHeader: Boolean,
    impute: Boolean,
    quote: java.lang.Character,
    skipBlankLines: Boolean,
    forceBGZ: Boolean
  ): Table = importTables(inputs.asScala,
    Option(keyNames).map(_.asScala.toFastIndexedSeq),
    if (nPartitions == null) None else Some(nPartitions), types.asScala.toMap, comment.asScala.toArray,
    separator, missing, noHeader, impute, quote, skipBlankLines, forceBGZ)

  def importTable(input: String,
    keyNames: Option[IndexedSeq[String]] = None,
    nPartitions: Option[Int] = None,
    types: Map[String, Type] = Map.empty[String, Type],
    comment: Array[String] = Array.empty[String],
    separator: String = "\t",
    missing: String = "NA",
    noHeader: Boolean = false,
    impute: Boolean = false,
    quote: java.lang.Character = null,
    skipBlankLines: Boolean = false,
    forceBGZ: Boolean = false
  ): Table = importTables(List(input), keyNames, nPartitions, types, comment,
    separator, missing, noHeader, impute, quote, skipBlankLines, forceBGZ)

  def importTables(inputs: Seq[String],
    keyNames: Option[IndexedSeq[String]] = None,
    nPartitions: Option[Int] = None,
    types: Map[String, Type] = Map.empty[String, Type],
    comment: Array[String] = Array.empty[String],
    separator: String = "\t",
    missing: String = "NA",
    noHeader: Boolean = false,
    impute: Boolean = false,
    quote: java.lang.Character = null,
    skipBlankLines: Boolean = false,
    forceBGZ: Boolean = false): Table = {
    require(nPartitions.forall(_ > 0), "nPartitions argument must be positive")

    val files = hadoopConf.globAll(inputs)
    if (files.isEmpty)
      fatal(s"Arguments referred to no files: '${ inputs.mkString(",") }'")

    maybeGZipAsBGZip(forceBGZ) {
      TextTableReader.read(this)(files, types, comment, separator, missing,
        noHeader, impute, nPartitions.getOrElse(sc.defaultMinPartitions), quote,
        skipBlankLines).keyBy(keyNames)
    }
  }

  def importPlink(bed: String, bim: String, fam: String,
    nPartitions: Option[Int] = None,
    delimiter: String = "\\\\s+",
    missing: String = "NA",
    quantPheno: Boolean = false,
    a2Reference: Boolean = true,
    rg: Option[ReferenceGenome] = Some(ReferenceGenome.defaultReference),
    contigRecoding: Option[Map[String, String]] = None,
    skipInvalidLoci: Boolean = false): MatrixTable = {

    rg.foreach(ref => contigRecoding.foreach(ref.validateContigRemap))

    val ffConfig = FamFileConfig(quantPheno, delimiter, missing)

    LoadPlink(this, bed, bim, fam,
      ffConfig, nPartitions, a2Reference, rg, contigRecoding.getOrElse(Map.empty[String, String]), skipInvalidLoci)
  }

  def importPlinkBFile(bfileRoot: String,
    nPartitions: Option[Int] = None,
    delimiter: String = "\\\\s+",
    missing: String = "NA",
    quantPheno: Boolean = false,
    a2Reference: Boolean = true,
    rg: Option[ReferenceGenome] = Some(ReferenceGenome.defaultReference),
    contigRecoding: Option[Map[String, String]] = None,
    skipInvalidLoci: Boolean = false): MatrixTable = {
    importPlink(bfileRoot + ".bed", bfileRoot + ".bim", bfileRoot + ".fam",
      nPartitions, delimiter, missing, quantPheno, a2Reference, rg, contigRecoding, skipInvalidLoci)
  }

  def read(file: String, dropCols: Boolean = false, dropRows: Boolean = false): MatrixTable = {
    MatrixTable.read(this, file, dropCols = dropCols, dropRows = dropRows)
  }

  def readVDS(file: String, dropSamples: Boolean = false, dropVariants: Boolean = false): MatrixTable =
    read(file, dropSamples, dropVariants)

  def readGDS(file: String, dropSamples: Boolean = false, dropVariants: Boolean = false): MatrixTable =
    read(file, dropSamples, dropVariants)

  def readTable(path: String): Table = Table.read(this, path)

  def readPartitions[T: ClassTag](
    path: String,
    partFiles: Array[String],
    read: (Int, InputStream, InputMetrics) => Iterator[T],
    optPartitioner: Option[Partitioner] = None): RDD[T] = {
    val nPartitions = partFiles.length

    val sHadoopConfBc = sc.broadcast(new SerializableHadoopConfiguration(sc.hadoopConfiguration))

    new RDD[T](sc, Nil) {
      def getPartitions: Array[Partition] =
        Array.tabulate(nPartitions)(i => FilePartition(i, partFiles(i)))

      override def compute(split: Partition, context: TaskContext): Iterator[T] = {
        val p = split.asInstanceOf[FilePartition]
        val filename = path + "/parts/" + p.file
        val in = sHadoopConfBc.value.value.unsafeReader(filename)
        read(p.index, in, context.taskMetrics().inputMetrics)
      }

      @transient override val partitioner: Option[Partitioner] = optPartitioner
    }
  }

  def readRows(
    path: String,
    t: TStruct,
    codecSpec: CodecSpec,
    partFiles: Array[String],
    requestedType: TStruct
  ): ContextRDD[RVDContext, RegionValue] = {
    val makeDec = codecSpec.buildDecoder(t, requestedType)
    ContextRDD.weaken[RVDContext](readPartitions(path, partFiles, (_, is, m) => Iterator.single(is -> m)))
      .cmapPartitions { (ctx, it) =>
        assert(it.hasNext)
        val (is, m) = it.next
        assert(!it.hasNext)
        HailContext.readRowsPartition(makeDec)(ctx, is, m)
      }
  }

  def parseVCFMetadata(file: String): Map[String, Map[String, Map[String, String]]] = {
    val reader = new HtsjdkRecordReader(Set.empty)
    LoadVCF.parseHeaderMetadata(this, reader, file)
  }

  private[this] val codecsKey = "io.compression.codecs"
  private[this] val hadoopGzipCodec = "org.apache.hadoop.io.compress.GzipCodec"
  private[this] val hailGzipAsBGZipCodec = "is.hail.io.compress.BGzipCodecGZ"

  def maybeGZipAsBGZip[T](force: Boolean)(body: => T): T = {
    if (!force)
      body
    else {
      val defaultCodecs = hadoopConf.get(codecsKey)
      hadoopConf.set(codecsKey, defaultCodecs.replaceAllLiterally(hadoopGzipCodec, hailGzipAsBGZipCodec))
      try {
        body
      } finally {
        hadoopConf.set(codecsKey, defaultCodecs)
      }
    }
  }

  def importVCF(file: String, force: Boolean = false,
    forceBGZ: Boolean = false,
    headerFile: Option[String] = None,
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false,
    callFields: Set[String] = Set.empty[String],
    rg: Option[ReferenceGenome] = Some(ReferenceGenome.defaultReference),
    contigRecoding: Option[Map[String, String]] = None,
    arrayElementsRequired: Boolean = true,
    skipInvalidLoci: Boolean = false): MatrixTable = {
    val addedReference = rg.exists { referenceGenome =>
      if (!ReferenceGenome.hasReference(referenceGenome.name)) {
        ReferenceGenome.addReference(referenceGenome)
        true
      } else false
    }

    val reader = MatrixVCFReader(
      Array(file),
      callFields,
      headerFile,
      nPartitions,
      rg.map(_.name),
      contigRecoding.getOrElse(Map.empty[String, String]),
      arrayElementsRequired,
      skipInvalidLoci,
      forceBGZ,
      force
    )
    if (addedReference)
      ReferenceGenome.removeReference(rg.get.name)
    new MatrixTable(HailContext.get, MatrixRead(reader.fullType, dropSamples, false, reader))
  }

  def importMatrix(files: java.util.ArrayList[String],
    rowFields: java.util.HashMap[String, Type],
    keyNames: java.util.ArrayList[String],
    cellType: Type,
    missingVal: String,
    minPartitions: Option[Int],
    noHeader: Boolean,
    forceBGZ: Boolean,
    sep: String = "\t"): MatrixTable =
    importMatrices(files.asScala, rowFields.asScala.toMap, keyNames.asScala.toArray,
      cellType, missingVal, minPartitions, noHeader, forceBGZ, sep)

  def importMatrices(files: Seq[String],
    rowFields: Map[String, Type],
    keyNames: Array[String],
    cellType: Type,
    missingVal: String = "NA",
    nPartitions: Option[Int],
    noHeader: Boolean,
    forceBGZ: Boolean,
    sep: String = "\t"): MatrixTable = {
    assert(sep.length == 1)

    val inputs = hadoopConf.globAll(files)

    maybeGZipAsBGZip(forceBGZ) {
      LoadMatrix(this, inputs, rowFields, keyNames, cellType = TStruct("x" -> cellType), missingVal, nPartitions, noHeader, sep(0))
    }
  }

  def indexBgen(file: String) {
    indexBgen(List(file))
  }

  def indexBgen(files: Seq[String]) {
    val inputs = hadoopConf.globAll(files).flatMap { file =>
      if (!file.endsWith(".bgen"))
        warn(s"Input file does not have .bgen extension: $file")

      if (hadoopConf.isDir(file))
        hadoopConf.listStatus(file)
          .map(_.getPath.toString)
          .filter(p => ".*part-[0-9]+".r.matches(p))
      else
        Array(file)
    }

    if (inputs.isEmpty)
      fatal(s"arguments refer to no files: '${ files.mkString(",") }'")

    val conf = new SerializableHadoopConfiguration(hadoopConf)

    sc.parallelize(inputs, numSlices = inputs.length).foreach { in =>
      LoadBgen.index(conf.value, in)
    }

    info(s"Number of BGEN files indexed: ${ inputs.length }")
  }

  def genDataset(): MatrixTable = VSMSubgen.realistic.gen(this).sample()
}
