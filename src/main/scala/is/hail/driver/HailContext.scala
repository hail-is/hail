package is.hail.driver

import java.util.Properties

import org.apache.hadoop
import is.hail.annotations.Annotation
import is.hail.expr.{EvalContext, Parser, TStruct, Type}
import is.hail.io.bgen.BgenLoader
import is.hail.io.gen.{GenLoader, GenReport}
import is.hail.expr._
import is.hail.io.plink.{FamFileConfig, PlinkLoader}
import is.hail.io.vcf.{LoadVCF, VCFReport}
import is.hail.keytable.KeyTable
import is.hail.methods.DuplicateReport
import is.hail.misc.SeqrServer
import is.hail.stats.{BaldingNicholsModel, Distribution, UniformDist}
import is.hail.utils.{log, _}
import is.hail.variant.{Genotype, VSMSubgen, Variant, VariantDataset, VariantMetadata, VariantSampleMatrix}
import org.apache.log4j.{LogManager, PropertyConfigurator}
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{ProgressBarBuilder, SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

object HailContext {
  def configureAndCreateSparkContext(appName: String, master: Option[String], local: String,
    parquetCompression: String, blockSize: Long): SparkContext = {
    require(blockSize >= 0)

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

    val tera = 1024L * 1024L * 1024L * 1024L

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
    SparkHadoopUtil.get.conf.setLong("parquet.block.size", tera)
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

    val codecKey = "spark.hadoop.io.compression.codecs"
    val codecs = conf.get(codecKey).split(",").toSet
    val requiredCodecs = List("org.apache.hadoop.io.compress.DefaultCodec",
      "is.hail.io.compress.BGzipCodec",
      "org.apache.hadoop.io.compress.GzipCodec")
    requiredCodecs.foreach { codec =>
      if (!codecs.contains(codec))
        problems += s"Invalid config parameter '$codecKey': missing codec '$codec'"
    }

    val enoughGigs = 1024L * 1024L * 1024L * 50
    val sqlFileKeys = List("spark.sql.files.openCostInBytes",
      "spark.sql.files.maxPartitionBytes")

    sqlFileKeys.foreach { k =>
      if (conf.getLong(k, 0) < enoughGigs)
        problems += s"Invalid config paramter '$k': too small, require >50G"
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
    parquetCompression: String = "uncompressed",
    blockSize: Long = 1L,
    branchingFactor: Int = 50,
    tmpDir: String = "/tmp"): HailContext = {

    {
      import breeze.linalg._
      import breeze.linalg.operators.{OpMulMatrix, BinaryRegistry}

      implicitly[BinaryRegistry[DenseMatrix[Double], Vector[Double], OpMulMatrix.type, DenseVector[Double]]].register(
        DenseMatrix.implOpMulMatrix_DMD_DVD_eq_DVD)
    }

    configureLogging(logFile, quiet, append)

    val sparkContext = if (sc == null)
      configureAndCreateSparkContext(appName, master, local, parquetCompression, blockSize)
    else {
      SparkHadoopUtil.get.conf.setLong("parquet.block.size", 1024L * 1024L * 1024L * 1024L)
      checkSparkConfiguration(sc)
      sc
    }

    ProgressBarBuilder.build(sparkContext)

    log.info(s"Spark properties: ${
      sparkContext.getConf.getAll.map { case (k, v) =>
        s"$k=$v"
      }.mkString(", ")
    }")

    val sqlContext = new org.apache.spark.sql.SQLContext(sparkContext)
    HailContext(sparkContext, sqlContext, tmpDir, branchingFactor)
  }
}

case class HailContext private(sc: SparkContext,
  sqlContext: SQLContext,
  tmpDir: String,
  branchingFactor: Int) {
  val hadoopConf: hadoop.conf.Configuration = sc.hadoopConfiguration

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

  def importAnnotationsTable(path: String,
    variantExpr: String,
    code: Option[String] = None,
    nPartitions: Option[Int] = None,
    config: TextTableConfiguration = TextTableConfiguration()): VariantDataset = {
    importAnnotationsTables(List(path), variantExpr, code, nPartitions, config)
  }

  def importAnnotationsTables(paths: Seq[String],
    variantExpr: String,
    code: Option[String] = None,
    nPartitions: Option[Int] = None,
    config: TextTableConfiguration = TextTableConfiguration()): VariantDataset = {
    val files = hadoopConf.globAll(paths)
    if (files.isEmpty)
      fatal("Arguments referred to no files")

    val (struct, rdd) = nPartitions match {
      case Some(n) =>
        if (n < 1)
          fatal("requested number of partitions in -n/--npartitions must be positive")
        else
          TextTableReader.read(sc)(files, config, n)
      case None =>
        TextTableReader.read(sc)(files, config)
    }

    val (finalType, fn): (Type, (Annotation, Option[Annotation]) => Annotation) = code.map { code =>
      val ec = EvalContext(Map(
        "va" -> (0, TStruct.empty),
        "table" -> (1, struct)))
      Annotation.buildInserter(code, TStruct.empty, ec, Annotation.VARIANT_HEAD)
    }.getOrElse((struct, (_: Annotation, anno: Option[Annotation]) => anno.orNull))

    val ec = EvalContext(struct.fields.map(f => (f.name, f.typ)): _*)
    val variantFn = Parser.parseTypedExpr[Variant](variantExpr, ec)

    val keyedRDD = rdd.flatMap {
      _.map { a =>
        ec.setAll(a.asInstanceOf[Row].toSeq: _*)
        variantFn().map(v => (v, (fn(null, Some(a)), Iterable.empty[Genotype])))
      }.value
    }.toOrderedRDD

    VariantSampleMatrix(this, VariantMetadata(Array.empty[String], IndexedSeq.empty[Annotation], Annotation.empty,
      TStruct.empty, finalType, TStruct.empty), keyedRDD)
  }

  def importBgen(file: String,
    sampleFile: Option[String] = None,
    tolerance: Double = 0.2,
    nPartitions: Option[Int] = None,
    compress: Boolean = true): VariantDataset = {
    importBgens(List(file), sampleFile, tolerance, nPartitions, compress)
  }

  def importBgens(files: Seq[String],
    sampleFile: Option[String] = None,
    tolerance: Double = 0.2,
    nPartitions: Option[Int] = None,
    compress: Boolean = true): VariantDataset = {

    val inputs = hadoopConf.globAll(files)

    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    inputs.foreach { input =>
      if (!input.endsWith(".bgen"))
        fatal("unknown input file type")
    }

    BgenLoader.load(this, inputs, sampleFile, tolerance, compress, nPartitions)
  }

  def importGen(file: String,
    sampleFile: String,
    chromosome: Option[String] = None,
    nPartitions: Option[Int] = None,
    tolerance: Double = 0.2,
    compress: Boolean = true): VariantDataset = {
    importGens(List(file), sampleFile, chromosome, nPartitions, tolerance, compress)
  }

  def importGens(files: Seq[String],
    sampleFile: String,
    chromosome: Option[String] = None,
    nPartitions: Option[Int] = None,
    tolerance: Double = 0.2,
    compress: Boolean = true): VariantDataset = {
    val inputs = hadoopConf.globAll(files)

    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    inputs.foreach { input =>
      if (!input.endsWith(".gen")) {
        fatal("unknown input file type")
      }
    }

    val samples = BgenLoader.readSampleFile(sc.hadoopConfiguration, sampleFile)
    val nSamples = samples.length

    //FIXME: can't specify multiple chromosomes
    val results = inputs.map(f => GenLoader(f, sampleFile, sc, nPartitions,
      tolerance, compress, chromosome))

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

    VariantSampleMatrix(this, VariantMetadata(samples).copy(isDosage = true),
      sc.union(results.map(_.rdd)).toOrderedRDD)
      .copy(vaSignature = signature, wasSplit = true)
  }

  def importKeytable(files: Seq[String],
    keyNames: Seq[String],
    nPartitions: Option[Int] = None,
    config: TextTableConfiguration = TextTableConfiguration()): KeyTable = {

    val inputs = hadoopConf.globAll(files)
    KeyTable.importTextTable(this, inputs, keyNames.mkString(","), nPartitions.getOrElse(sc.defaultMinPartitions), config)
  }

  def importPlink(bed: String, bim: String, fam: String,
    nPartitions: Option[Int] = None,
    delimiter: String = "\\\\s+",
    missing: String = "NA",
    quantPheno: Boolean = false,
    compress: Boolean = true): VariantDataset = {

    val ffConfig = FamFileConfig(quantPheno, delimiter, missing)
    hadoopConf.setBoolean("compressGS", compress)

    PlinkLoader(this, bed, bim, fam,
      ffConfig, nPartitions)
  }

  def importPlinkBFile(bfileRoot: String,
    nPartitions: Option[Int] = None,
    delimiter: String = "\\\\s+",
    missing: String = "NA",
    quantPheno: Boolean = false,
    compress: Boolean = true): VariantDataset = {
    importPlink(bfileRoot + ".bed", bfileRoot + ".bim", bfileRoot + ".fam",
      nPartitions, delimiter, missing, quantPheno, compress)
  }

  def read(file: String, sitesOnly: Boolean = false, samplesOnly: Boolean = false): VariantDataset = {
    readAll(List(file), sitesOnly, samplesOnly)
  }

  def readAll(files: Seq[String], sitesOnly: Boolean = false, samplesOnly: Boolean = false): VariantDataset = {
    val inputs = hadoopConf.globAll(files)
    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    val vdses = inputs.map(input => VariantDataset.read(this, input,
      skipGenotypes = sitesOnly, skipVariants = samplesOnly))

    val sampleIds = vdses.head.sampleIds
    val vaSchema = vdses.head.vaSignature
    val wasSplit = vdses.head.wasSplit
    val reference = inputs(0)

    vdses.indices.tail.foreach { i =>
      val vds = vdses(i)
      val ids = vds.sampleIds
      val vas = vds.vaSignature
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
      }
    }

    if (vdses.length > 1)
      info(s"Using sample and global annotations from ${ inputs(0) }")

    vdses(0).copy(rdd = sc.union(vdses.map(_.rdd)).toOrderedRDD)
  }

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
    sitesOnly: Boolean = false,
    storeGQ: Boolean = false,
    ppAsPL: Boolean = false,
    skipBadAD: Boolean = false,
    compress: Boolean = true): VariantDataset = {
    importVCFs(List(file), force, forceBGZ, headerFile, nPartitions, sitesOnly,
      storeGQ, ppAsPL, skipBadAD, compress)
  }

  def importVCFs(files: Seq[String], force: Boolean = false,
    forceBGZ: Boolean = false,
    headerFile: Option[String] = None,
    nPartitions: Option[Int] = None,
    sitesOnly: Boolean = false,
    storeGQ: Boolean = false,
    ppAsPL: Boolean = false,
    skipBadAD: Boolean = false,
    compress: Boolean = true): VariantDataset = {

    val inputs = LoadVCF.globAllVCFs(hadoopConf.globAll(files), hadoopConf, force || forceBGZ)

    val header = headerFile.getOrElse(inputs.head)

    val codecs = sc.hadoopConfiguration.get("io.compression.codecs")

    if (forceBGZ)
      hadoopConf.set("io.compression.codecs",
        codecs.replaceAllLiterally("org.apache.hadoop.io.compress.GzipCodec", "is.hail.io.compress.BGzipCodecGZ"))

    val vds = LoadVCF(this,
      header,
      inputs,
      storeGQ,
      compress,
      nPartitions,
      sitesOnly,
      ppAsPL,
      skipBadAD)

    hadoopConf.set("io.compression.codecs", codecs)

    vds
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

  def dataframeToKeytable(df: DataFrame, keys: Array[String] = Array.empty[String]): KeyTable =
    KeyTable.fromDF(this, df, keys)

  def genDataset(): VariantDataset = VSMSubgen.realistic.gen(this).sample()

  /**
    *
    * @param collection SolrCloud collection
    * @param url Solr instance (URL) to connect to
    * @param zkHost Zookeeper host string to connect to
    * @param jsonFields Comma-separated list of JSON-encoded fields
    * @param solrOnly Return results directly queried from Solr
    * @param address Cassandra contact point to connect to
    * @param keyspace Cassandra keyspace
    * @param table Cassandra table
    */
  def seqrServer(collection: String = null,
    url: String = null,
    zkHost: String = null,
    jsonFields: String = null,
    solrOnly: Boolean = false,
    address: String = null,
    keyspace: String = null,
    table: String = null) {
    SeqrServer.start(collection, url, zkHost, jsonFields, solrOnly, address, keyspace, table)
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