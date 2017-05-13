package is.hail

import java.util.Properties

import is.hail.annotations.Annotation
import is.hail.expr.{EvalContext, Parser, TStruct, Type, _}
import is.hail.io.bgen.BgenLoader
import is.hail.io.gen.{GenLoader, GenReport}
import is.hail.io.plink.{FamFileConfig, PlinkLoader}
import is.hail.io.vcf._
import is.hail.keytable.KeyTable
import is.hail.methods.DuplicateReport
import is.hail.stats.{BaldingNicholsModel, Distribution, UniformDist}
import is.hail.utils.{log, _}
import is.hail.variant.{GenericDataset, Genotype, VSMSubgen, Variant, VariantDataset, VariantMetadata, VariantSampleMatrix}
import org.apache.hadoop
import org.apache.log4j.{LogManager, PropertyConfigurator}
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{ProgressBarBuilder, SparkConf, SparkContext}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

object HailContext {

  val tera = 1024L * 1024L * 1024L * 1024L

  def configureAndCreateSparkContext(appName: String, master: Option[String], local: String,
    parquetCompression: String, blockSize: Long): SparkContext = {
    require(blockSize >= 0)
    require(is.hail.HAIL_SPARK_VERSION == org.apache.spark.SPARK_VERSION,
      s"""This Hail JAR was compiled for Spark ${ is.hail.HAIL_SPARK_VERSION },
         |  but the version of Spark available at runtime is ${ org.apache.spark.SPARK_VERSION }.""".stripMargin)

    val conf = new SparkConf().setAppName(appName)

    master match {
      case Some(m) =>
        conf.setMaster(m)
      case None =>
        if (!conf.contains("spark.master"))
          conf.setMaster(local)
    }

    conf.set("spark.ui.showConsoleProgress", "false")

    conf.set(
      "spark.hadoop.io.compression.codecs",
      "org.apache.hadoop.io.compress.DefaultCodec," +
        "is.hail.io.compress.BGzipCodec," +
        "org.apache.hadoop.io.compress.GzipCodec")

    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    conf.set("spark.sql.parquet.compression.codec", parquetCompression)
    conf.set("spark.sql.files.openCostInBytes", tera.toString)
    conf.set("spark.sql.files.maxPartitionBytes", tera.toString)

    conf.set("spark.hadoop.mapreduce.input.fileinputformat.split.minsize", (blockSize * 1024L * 1024L).toString)

    /* `DataFrame.write` writes one file per partition.  Without this, read will split files larger than the default
     * parquet block size into multiple partitions.  This causes `OrderedRDD` to fail since the per-partition range
     * no longer line up with the RDD partitions.
     *
     * For reasons we don't understand, the DataFrame code uses `SparkHadoopUtil.get.conf` instead of the Hadoop
     * configuration in the SparkContext.  Set both for consistency.
     */
    conf.set("spark.hadoop.parquet.block.size", tera.toString)

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

    val enoughGigs = 1024L * 1024L * 1024L * 50
    val sqlFileKeys = List("spark.sql.files.openCostInBytes",
      "spark.sql.files.maxPartitionBytes")

    sqlFileKeys.foreach { k =>
      val param = conf.getLong(k, 0)
      if (param < enoughGigs)
        problems += s"Invalid config parameter '$k=': too small. Found $param, require at least 50G"
    }

    if (problems.nonEmpty)
      fatal(
        s"""Found problems with SparkContext configuration:
           |  ${ problems.mkString("\n  ") }""".stripMargin)
  }

  def configureLogging(logFile: String, quiet: Boolean, append: Boolean) {
    val logProps = new Properties()
    if (quiet) {
      logProps.put("log4j.rootLogger", "OFF, stderr")
      logProps.put("log4j.appender.stderr", "org.apache.log4j.ConsoleAppender")
      logProps.put("log4j.appender.stderr.Target", "System.err")
      logProps.put("log4j.appender.stderr.threshold", "OFF")
      logProps.put("log4j.appender.stderr.layout", "org.apache.log4j.PatternLayout")
      logProps.put("log4j.appender.stderr.layout.ConversionPattern", "%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n")
    } else {
      logProps.put("log4j.rootLogger", "INFO, logfile")
      logProps.put("log4j.appender.logfile", "org.apache.log4j.FileAppender")
      logProps.put("log4j.appender.logfile.append", append.toString)
      logProps.put("log4j.appender.logfile.file", logFile)
      logProps.put("log4j.appender.logfile.threshold", "INFO")
      logProps.put("log4j.appender.logfile.layout", "org.apache.log4j.PatternLayout")
      logProps.put("log4j.appender.logfile.layout.ConversionPattern", "%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n")
    }

    LogManager.resetConfiguration()
    PropertyConfigurator.configure(logProps)
  }

  def apply(sc: SparkContext = null,
    appName: String = "Hail",
    master: Option[String] = None,
    local: String = "local[*]",
    logFile: String = "hail.log",
    quiet: Boolean = false,
    append: Boolean = false,
    parquetCompression: String = "snappy",
    minBlockSize: Long = 1L,
    branchingFactor: Int = 50,
    tmpDir: String = "/tmp"): HailContext = {

    val javaVersion = System.getProperty("java.version")
    if (!javaVersion.startsWith("1.8"))
      fatal(s"Hail requires Java 1.8, found version $javaVersion")

    {
      import breeze.linalg._
      import breeze.linalg.operators.{BinaryRegistry, OpMulMatrix}

      implicitly[BinaryRegistry[DenseMatrix[Double], Vector[Double], OpMulMatrix.type, DenseVector[Double]]].register(
        DenseMatrix.implOpMulMatrix_DMD_DVD_eq_DVD)
    }

    configureLogging(logFile, quiet, append)

    val sparkContext = if (sc == null)
      configureAndCreateSparkContext(appName, master, local, parquetCompression, minBlockSize)
    else {
      SparkHadoopUtil.get.conf.setLong("parquet.block.size", 1024L * 1024L * 1024L * 1024L)
      checkSparkConfiguration(sc)
      sc
    }

    SparkHadoopUtil.get.conf.setLong("parquet.block.size", tera)
    sparkContext.hadoopConfiguration.set("io.compression.codecs",
      "org.apache.hadoop.io.compress.DefaultCodec," +
        "is.hail.io.compress.BGzipCodec," +
        "org.apache.hadoop.io.compress.GzipCodec"
    )

    sparkContext.uiWebUrl.foreach(ui => info(s"SparkUI: $ui"))
    ProgressBarBuilder.build(sparkContext)

    log.info(s"Spark properties: ${
      sparkContext.getConf.getAll.map { case (k, v) =>
        s"$k=$v"
      }.mkString(", ")
    }")

    val sqlContext = new org.apache.spark.sql.SQLContext(sparkContext)
    val hc = new HailContext(sparkContext, sqlContext, tmpDir, branchingFactor)
    val welcomeMessage =
      """Welcome to
        |     __  __     <>__
        |    / /_/ /__  __/ /
        |   / __  / _ `/ / /
        |  /_/ /_/\_,_/_/_/   version """.stripMargin + is.hail.HAIL_PRETTY_VERSION
    println(welcomeMessage)
    log.info(welcomeMessage)
    hc
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
      .groupBy(_.source.asInstanceOf[TextContext].file)
      .foreach { case (file, lines) =>
        info(s"$file: ${ lines.length } ${ plural(lines.length, "match", "matches") }:")
        lines.map(_.value).foreach { line =>
          val (screen, logged) = line.truncatable().strings
          log.info("\t" + logged)
          println(s"\t$screen")
        }
      }
  }

  def importBgen(file: String,
    sampleFile: Option[String] = None,
    tolerance: Double = 0.2,
    nPartitions: Option[Int] = None): VariantDataset = {
    importBgens(List(file), sampleFile, tolerance, nPartitions)
  }

  def importBgens(files: Seq[String],
    sampleFile: Option[String] = None,
    tolerance: Double = 0.2,
    nPartitions: Option[Int] = None): VariantDataset = {

    val inputs = hadoopConf.globAll(files)

    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    inputs.foreach { input =>
      if (!input.endsWith(".bgen"))
        fatal("unknown input file type")
    }

    BgenLoader.load(this, inputs, sampleFile, tolerance, nPartitions)
  }

  def importGen(file: String,
    sampleFile: String,
    chromosome: Option[String] = None,
    nPartitions: Option[Int] = None,
    tolerance: Double = 0.2): VariantDataset = {
    importGens(List(file), sampleFile, chromosome, nPartitions, tolerance)
  }

  def importGens(files: Seq[String],
    sampleFile: String,
    chromosome: Option[String] = None,
    nPartitions: Option[Int] = None,
    tolerance: Double = 0.2): VariantDataset = {
    val inputs = hadoopConf.globAll(files)

    inputs.foreach { input =>
      if (!hadoopConf.stripCodec(input).endsWith(".gen"))
        fatal(s"gen inputs must end in .gen[.bgz], found $input")
    }

    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    val samples = BgenLoader.readSampleFile(sc.hadoopConfiguration, sampleFile)
    val nSamples = samples.length

    //FIXME: can't specify multiple chromosomes
    val results = inputs.map(f => GenLoader(f, sampleFile, sc, nPartitions,
      tolerance, chromosome))

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

    val signature = TStruct("rsid" -> TString, "varid" -> TString)

    VariantSampleMatrix(this, VariantMetadata(samples).copy(isLinearScale = true),
      sc.union(results.map(_.rdd)).toOrderedRDD)
      .copy(vaSignature = signature, wasSplit = true)
  }

  def importTable(inputs: java.util.ArrayList[String],
    keyNames: java.util.ArrayList[String],
    nPartitions: java.lang.Integer,
    types: java.util.HashMap[String, Type],
    commentChar: String,
    separator: String,
    missing: String,
    noHeader: Boolean,
    impute: Boolean): KeyTable = importTables(inputs.asScala, keyNames.asScala.toArray, if (nPartitions == null) None else Some(nPartitions),
    types.asScala.toMap, Option(commentChar), separator, missing, noHeader, impute)

  def importTable(input: String,
    keyNames: Array[String] = Array.empty[String],
    nPartitions: Option[Int] = None,
    types: Map[String, Type] = Map.empty[String, Type],
    commentChar: Option[String] = None,
    separator: String = "\t",
    missing: String = "NA",
    noHeader: Boolean = false,
    impute: Boolean = false): KeyTable = {
    importTables(List(input), keyNames, nPartitions, types, commentChar, separator, missing, noHeader, impute)
  }

  def importTables(inputs: Seq[String],
    keyNames: Array[String] = Array.empty[String],
    nPartitions: Option[Int] = None,
    types: Map[String, Type] = Map.empty[String, Type],
    commentChar: Option[String] = None,
    separator: String = "\t",
    missing: String = "NA",
    noHeader: Boolean = false,
    impute: Boolean = false): KeyTable = {
    require(nPartitions.forall(_ > 0), "nPartitions argument must be positive")

    val files = hadoopConf.globAll(inputs)
    if (files.isEmpty)
      fatal("Arguments referred to no files")

    val (struct, rdd) =
      TextTableReader.read(sc)(files, types, commentChar, separator, missing,
        noHeader, impute, nPartitions.getOrElse(sc.defaultMinPartitions))

    KeyTable(this, rdd.map(_.value), struct, keyNames)
  }

  def importPlink(bed: String, bim: String, fam: String,
    nPartitions: Option[Int] = None,
    delimiter: String = "\\\\s+",
    missing: String = "NA",
    quantPheno: Boolean = false): VariantDataset = {

    val ffConfig = FamFileConfig(quantPheno, delimiter, missing)

    PlinkLoader(this, bed, bim, fam,
      ffConfig, nPartitions)
  }

  def importPlinkBFile(bfileRoot: String,
    nPartitions: Option[Int] = None,
    delimiter: String = "\\\\s+",
    missing: String = "NA",
    quantPheno: Boolean = false): VariantDataset = {
    importPlink(bfileRoot + ".bed", bfileRoot + ".bim", bfileRoot + ".fam",
      nPartitions, delimiter, missing, quantPheno)
  }

  def checkDatasetSchemasCompatible[T](datasets: Array[VariantSampleMatrix[T]], inputs: Array[String]) {
    val sampleIds = datasets.head.sampleIds
    val vaSchema = datasets.head.vaSignature
    val wasSplit = datasets.head.wasSplit
    val genotypeSchema = datasets.head.genotypeSignature
    val isGenericGenotype = datasets.head.isGenericGenotype
    val reference = inputs(0)

    datasets.indices.tail.foreach { i =>
      val vds = datasets(i)
      val ids = vds.sampleIds
      val vas = vds.vaSignature
      val gsig = vds.genotypeSignature
      val isGenGt = vds.isGenericGenotype
      val path = inputs(i)
      if (ids != sampleIds) {
        fatal(
          s"""cannot read datasets with different sample IDs or sample ordering
             |  IDs in reference file $reference: @1
             |  IDs in file $path: @2""".stripMargin, sampleIds, ids)
      } else if (wasSplit != vds.wasSplit) {
        fatal(
          s"""cannot combine split and unsplit datasets
             |  Reference file $reference split status: $wasSplit
             |  File $path split status: ${ vds.wasSplit }""".stripMargin)
      } else if (vas != vaSchema) {
        fatal(
          s"""cannot read datasets with different variant annotation schemata
             |  Schema in reference file $reference: @1
             |  Schema in file $path: @2""".stripMargin,
          vaSchema.toPrettyString(compact = true, printAttrs = true),
          vas.toPrettyString(compact = true, printAttrs = true)
        )
      } else if (gsig != genotypeSchema) {
        fatal(
          s"""cannot read datasets with different genotype schemata
             |  Schema in reference file $reference: @1
             |  Schema in file $path: @2""".stripMargin,
          genotypeSchema.toPrettyString(compact = true, printAttrs = true),
          gsig.toPrettyString(compact = true, printAttrs = true)
        )
      } else if (isGenGt != isGenericGenotype) {
        fatal(
          s"""cannot read datasets with different data formats
             |  Generic genotypes in reference file $reference: @1
             |  Generic genotypes in file $path: @2""".stripMargin,
          isGenericGenotype.toString,
          isGenGt.toString
        )
      }
    }

    if (datasets.length > 1)
      info(s"Using sample and global annotations from ${ inputs(0) }")
  }

  def readMetadata(file: String): (VariantMetadata, Boolean) = VariantDataset.readMetadata(hadoopConf, file)

  def readAllMetadata(files: Seq[String]): Array[(VariantMetadata, Boolean)] = files.map(readMetadata).toArray

  def read(file: String, dropSamples: Boolean = false, dropVariants: Boolean = false,
    metadata: Option[Array[(VariantMetadata, Boolean)]] = None): VariantDataset =
    readAll(List(file), dropSamples, dropVariants, metadata)

  def readAll(files: Seq[String], dropSamples: Boolean = false, dropVariants: Boolean = false,
    metadata: Option[Array[(VariantMetadata, Boolean)]] = None): VariantDataset = {
    val inputs = hadoopConf.globAll(files)
    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    val (mdArray, pqgtArray) = metadata match {
      case Some(data) => data.unzip
      case _ => inputs.map(VariantDataset.readMetadata(sc.hadoopConfiguration, _)).unzip
    }

    val vdses = inputs.zipWithIndex.map { case (input, i) =>
      VariantDataset.read(this, input, mdArray(i), pqgtArray(i),
        dropSamples = dropSamples, dropVariants = dropVariants)
    }

    checkDatasetSchemasCompatible(vdses, inputs)

    vdses(0).copy(rdd = sc.union(vdses.map(_.rdd)).toOrderedRDD)
  }

  def readGDS(file: String, dropSamples: Boolean = false, dropVariants: Boolean = false,
    metadata: Option[Array[(VariantMetadata, Boolean)]] = None): GenericDataset =
    readAllGDS(List(file), dropSamples, dropVariants, metadata)

  def readAllGDS(files: Seq[String], dropSamples: Boolean = false, dropVariants: Boolean = false,
    metadata: Option[Array[(VariantMetadata, Boolean)]] = None): GenericDataset = {
    val inputs = hadoopConf.globAll(files)
    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    val (mdArray, pqgtArray) = metadata match {
      case Some(data) => data.unzip
      case _ => inputs.map(VariantDataset.readMetadata(sc.hadoopConfiguration, _)).unzip
    }

    val gdses = inputs.zipWithIndex.map { case (input, i) =>
      GenericDataset.read(this, input, mdArray(i), pqgtArray(i),
        dropSamples = dropSamples, dropVariants = dropVariants)
    }

    checkDatasetSchemasCompatible(gdses, inputs)

    gdses(0).copy(rdd = sc.union(gdses.map(_.rdd)).toOrderedRDD)
  }

  def readKeyTable(path: String): KeyTable =
    KeyTable.read(this, path)

  /**
    *
    * @param path path to Kudu database
    * @param table table name
    * @param master Kudu master address
    */
  def readKudu(path: String, table: String, master: String): VariantDataset = {
    VariantDataset.readKudu(this, path, table, master)
  }

  def writePartitioning(path: String) {
    VariantSampleMatrix.writePartitioning(sqlContext, path)
  }

  def importVCF(file: String, force: Boolean = false,
    forceBGZ: Boolean = false,
    headerFile: Option[String] = None,
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false,
    storeGQ: Boolean = false,
    ppAsPL: Boolean = false,
    skipBadAD: Boolean = false): VariantDataset = {
    importVCFs(List(file), force, forceBGZ, headerFile, nPartitions, dropSamples,
      storeGQ, ppAsPL, skipBadAD)
  }

  def importVCFs(files: Seq[String], force: Boolean = false,
    forceBGZ: Boolean = false,
    headerFile: Option[String] = None,
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false,
    storeGQ: Boolean = false,
    ppAsPL: Boolean = false,
    skipBadAD: Boolean = false): VariantDataset = {

    val inputs = LoadVCF.globAllVCFs(hadoopConf.globAll(files), hadoopConf, force || forceBGZ)

    val header = headerFile.getOrElse(inputs.head)

    val codecs = sc.hadoopConfiguration.get("io.compression.codecs")

    if (forceBGZ)
      hadoopConf.set("io.compression.codecs",
        codecs.replaceAllLiterally("org.apache.hadoop.io.compress.GzipCodec", "is.hail.io.compress.BGzipCodecGZ"))

    val settings = VCFSettings(storeGQ, dropSamples, ppAsPL, skipBadAD)
    val reader = new GenotypeRecordReader(settings)
    val vds = LoadVCF(this, reader, header, inputs, nPartitions, dropSamples)

    hadoopConf.set("io.compression.codecs", codecs)

    vds
  }

  def importVCFGeneric(file: String, force: Boolean = false,
    forceBGZ: Boolean = false,
    headerFile: Option[String] = None,
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false,
    callFields: Set[String] = Set.empty[String]): GenericDataset = {
    importVCFsGeneric(List(file), force, forceBGZ, headerFile, nPartitions, dropSamples, callFields)
  }

  def importVCFsGeneric(files: Seq[String], force: Boolean = false,
    forceBGZ: Boolean = false,
    headerFile: Option[String] = None,
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false,
    callFields: Set[String] = Set.empty[String]): GenericDataset = {

    val inputs = LoadVCF.globAllVCFs(hadoopConf.globAll(files), hadoopConf, force || forceBGZ)

    val header = headerFile.getOrElse(inputs.head)

    val codecs = sc.hadoopConfiguration.get("io.compression.codecs")

    if (forceBGZ)
      hadoopConf.set("io.compression.codecs",
        codecs.replaceAllLiterally("org.apache.hadoop.io.compress.GzipCodec", "is.hail.io.compress.BGzipCodecGZ"))

    val reader = new GenericRecordReader(callFields)
    val gds = LoadVCF(this, reader, header, inputs, nPartitions, dropSamples)

    hadoopConf.set("io.compression.codecs", codecs)

    gds
  }

  def indexBgen(file: String) {
    indexBgen(List(file))
  }

  def indexBgen(files: Seq[String]) {
    val inputs = hadoopConf.globAll(files)

    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    inputs.foreach { input =>
      if (!input.endsWith(".bgen")) {
        fatal(s"unknown input file: $input")
      }
    }

    val conf = new SerializableHadoopConfiguration(hadoopConf)

    sc.parallelize(inputs).foreach { in =>
      BgenLoader.index(conf.value, in)
    }

    info(s"Number of BGEN files indexed: ${ inputs.length }")
  }

  def baldingNicholsModel(populations: Int,
    samples: Int,
    variants: Int,
    nPartitions: Option[Int] = None,
    popDist: Option[Array[Double]] = None,
    fst: Option[Array[Double]] = None,
    afDist: Distribution = UniformDist(0.1, 0.9),
    seed: Int = 0): VariantDataset =
    BaldingNicholsModel(this, populations, samples, variants, popDist, fst, seed, nPartitions, afDist)

  def genDataset(): VariantDataset = VSMSubgen.realistic.gen(this).sample()

  def eval(expr: String): (Annotation, Type) = {
    val ec = EvalContext()
    val (t, f) = Parser.parseExpr(expr, ec)
    (f(), t)
  }

  def report() {
    VCFReport.report()
    GenReport.report()
    DuplicateReport.report()
  }

  def stop() {
    sc.stop()
  }
}
