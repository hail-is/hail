package is.hail

import java.util.Properties

import is.hail.annotations.Annotation
import is.hail.expr.{EvalContext, Parser, TStruct, Type, _}
import is.hail.io.bgen.BgenLoader
import is.hail.io.gen.GenLoader
import is.hail.io.plink.{FamFileConfig, PlinkLoader}
import is.hail.io.vcf._
import is.hail.keytable.KeyTable
import is.hail.stats.{BaldingNicholsModel, Distribution, UniformDist}
import is.hail.utils.{log, _}
import is.hail.variant.{GenericDataset, GenomeReference, Genotype, Locus, VSMFileMetadata, VSMSubgen, Variant, VariantDataset, VariantSampleMatrix}
import org.apache.hadoop
import org.apache.log4j.{ConsoleAppender, LogManager, PatternLayout, PropertyConfigurator}
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.sql.SQLContext
import org.apache.spark.{ProgressBarBuilder, SparkConf, SparkContext}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.language.existentials
import scala.reflect.classTag

object HailContext {

  val tera = 1024L * 1024L * 1024L * 1024L

  val logFormat: String = "%d{yyyy-MM-dd HH:mm:ss} %c{1}: %p: %m%n"

  def configureAndCreateSparkContext(appName: String, master: Option[String],
    local: String, blockSize: Long): SparkContext = {
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

    val serializer = conf.get("spark.serializer")
    val kryoSerializer = "org.apache.spark.serializer.KryoSerializer"
    if (serializer != kryoSerializer)
      problems += s"Invalid configuration property spark.serializer: required $kryoSerializer.  Found: $serializer."

    if (!conf.getOption("spark.kryo.registrator").contains("is.hail.kryo.HailKryoRegistrator"))
      problems += s"Invalid config parameter: spark.kryo.registrator must be set to is.hail.kryo.HailKryoRegistrator"

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

  def apply(sc: SparkContext = null,
    appName: String = "Hail",
    master: Option[String] = None,
    local: String = "local[*]",
    logFile: String = "hail.log",
    quiet: Boolean = false,
    append: Boolean = false,
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

    ProgressBarBuilder.build(sparkContext)

    val sqlContext = new org.apache.spark.sql.SQLContext(sparkContext)
    val hc = new HailContext(sparkContext, sqlContext, tmpDir, branchingFactor)
    sparkContext.uiWebUrl.foreach(ui => info(s"SparkUI: $ui"))

    info(s"Running Hail version ${hc.version}")
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
    nPartitions: Option[Int] = None): GenericDataset = {
    importBgens(List(file), sampleFile, tolerance, nPartitions)
  }

  def importBgens(files: Seq[String],
    sampleFile: Option[String] = None,
    tolerance: Double = 0.2,
    nPartitions: Option[Int] = None): GenericDataset = {

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

    BgenLoader.load(this, inputs, sampleFile, tolerance, nPartitions)
  }

  def importGen(file: String,
    sampleFile: String,
    chromosome: Option[String] = None,
    nPartitions: Option[Int] = None,
    tolerance: Double = 0.2): GenericDataset = {
    importGens(List(file), sampleFile, chromosome, nPartitions, tolerance)
  }

  def importGens(files: Seq[String],
    sampleFile: String,
    chromosome: Option[String] = None,
    nPartitions: Option[Int] = None,
    tolerance: Double = 0.2): GenericDataset = {
    val inputs = hadoopConf.globAll(files)

    inputs.foreach { input =>
      if (!hadoopConf.stripCodec(input).endsWith(".gen"))
        fatal(s"gen inputs must end in .gen[.bgz], found $input")
    }

    if (inputs.isEmpty)
      fatal(s"arguments refer to no files: ${ files.mkString(",") }")

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

    val rdd = sc.union(results.map(_.rdd)).toOrderedRDD(TVariant(GenomeReference.GRCh37).orderedKey, classTag[(Annotation, Iterable[Annotation])])

    new GenericDataset(this,
      VSMFileMetadata(samples,
        vaSignature = signature,
        genotypeSignature = TStruct("GT" -> TCall,
          "GP" -> TArray(TFloat64)),
        wasSplit = true),
      rdd)
  }

  def importTable(inputs: java.util.ArrayList[String],
    keyNames: java.util.ArrayList[String],
    nPartitions: java.lang.Integer,
    types: java.util.HashMap[String, Type],
    commentChar: String,
    separator: String,
    missing: String,
    noHeader: Boolean,
    impute: Boolean,
    quote: java.lang.Character): KeyTable = importTables(inputs.asScala, keyNames.asScala.toArray, if (nPartitions == null) None else Some(nPartitions),
    types.asScala.toMap, Option(commentChar), separator, missing, noHeader, impute, quote)

  def importTable(input: String,
    keyNames: Array[String] = Array.empty[String],
    nPartitions: Option[Int] = None,
    types: Map[String, Type] = Map.empty[String, Type],
    commentChar: Option[String] = None,
    separator: String = "\t",
    missing: String = "NA",
    noHeader: Boolean = false,
    impute: Boolean = false,
    quote: java.lang.Character = null): KeyTable = {
    importTables(List(input), keyNames, nPartitions, types, commentChar, separator, missing, noHeader, impute, quote)
  }

  def importTables(inputs: Seq[String],
    keyNames: Array[String] = Array.empty[String],
    nPartitions: Option[Int] = None,
    types: Map[String, Type] = Map.empty[String, Type],
    commentChar: Option[String] = None,
    separator: String = "\t",
    missing: String = "NA",
    noHeader: Boolean = false,
    impute: Boolean = false,
    quote: java.lang.Character = null): KeyTable = {
    require(nPartitions.forall(_ > 0), "nPartitions argument must be positive")

    val files = hadoopConf.globAll(inputs)
    if (files.isEmpty)
      fatal(s"Arguments referred to no files: '${ files.mkString(",") }'")

    val (struct, rdd) =
      TextTableReader.read(sc)(files, types, commentChar, separator, missing,
        noHeader, impute, nPartitions.getOrElse(sc.defaultMinPartitions), quote)

    KeyTable(this, rdd.map(_.value), struct, keyNames)
  }

  def importPlink(bed: String, bim: String, fam: String,
    nPartitions: Option[Int] = None,
    delimiter: String = "\\\\s+",
    missing: String = "NA",
    quantPheno: Boolean = false): GenericDataset = {

    val ffConfig = FamFileConfig(quantPheno, delimiter, missing)

    PlinkLoader(this, bed, bim, fam,
      ffConfig, nPartitions)
  }

  def importPlinkBFile(bfileRoot: String,
    nPartitions: Option[Int] = None,
    delimiter: String = "\\\\s+",
    missing: String = "NA",
    quantPheno: Boolean = false): GenericDataset = {
    importPlink(bfileRoot + ".bed", bfileRoot + ".bim", bfileRoot + ".fam",
      nPartitions, delimiter, missing, quantPheno)
  }

  def checkDatasetSchemasCompatible(datasets: Array[VariantSampleMatrix[_, _, _]], inputs: Array[String]) {
    val sampleIds = datasets.head.sampleIds
    val vaSchema = datasets.head.vaSignature
    val wasSplit = datasets.head.wasSplit
    val genotypeSchema = datasets.head.genotypeSignature
    val reference = inputs(0)

    datasets.indices.tail.foreach { i =>
      val vds = datasets(i)
      val ids = vds.sampleIds
      val vas = vds.vaSignature
      val gsig = vds.genotypeSignature
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
      }
    }

    if (datasets.length > 1)
      info(s"Using sample and global annotations from ${ inputs(0) }")
  }

  def read(file: String, dropSamples: Boolean = false, dropVariants: Boolean = false): VariantSampleMatrix[_, _, _] =
    readAll(List(file), dropSamples, dropVariants)

  def readAll(files: Seq[String], dropSamples: Boolean = false, dropVariants: Boolean = false): VariantSampleMatrix[_, _, _] = {
    val inputs = hadoopConf.globAll(files)
    if (inputs.isEmpty)
      fatal(s"arguments refer to no files: '${ files.mkString(",") }'")

    val vsms = inputs.map { input =>
      VariantSampleMatrix.read(this, input, dropSamples = dropSamples, dropVariants = dropVariants)
    }

    if (vsms.length == 1)
      return vsms(0)

    checkDatasetSchemasCompatible(vsms, inputs)

    // I can't figure out how to write this with existentials -cs
    if (vsms(0).genotypeSignature == TGenotype) {
      val gdses = vsms.asInstanceOf[Array[VariantSampleMatrix[Annotation, Annotation, Annotation]]]
      implicit val kOk = gdses(0).kOk
      gdses(0).copy(
        rdd = sc.union(gdses.map(_.rdd)).toOrderedRDD)
    } else {
      val vdses = vsms.asInstanceOf[Array[VariantDataset]]
      implicit val kOk = vdses(0).kOk
      vdses(0).copy(
        rdd = sc.union(vdses.map(_.rdd)).toOrderedRDD)
    }
  }

  def readVDS(file: String, dropSamples: Boolean = false, dropVariants: Boolean = false): VariantDataset =
    readVDSAll(List(file), dropSamples, dropVariants)

  def readVDSAll(files: Seq[String], dropSamples: Boolean = false, dropVariants: Boolean = false): VariantDataset =
    readAll(files, dropSamples, dropVariants).toVDS

  def readGDS(file: String, dropSamples: Boolean = false, dropVariants: Boolean = false): GenericDataset =
    readAllGDS(List(file), dropSamples, dropVariants)

  def readAllGDS(files: Seq[String], dropSamples: Boolean = false, dropVariants: Boolean = false): GenericDataset =
    readAll(files, dropSamples, dropVariants).toGDS

  def readTable(path: String): KeyTable =
    KeyTable.read(this, path)

  def importVCF(file: String, force: Boolean = false,
    forceBGZ: Boolean = false,
    headerFile: Option[String] = None,
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false): VariantDataset = {
    importVCFs(List(file), force, forceBGZ, headerFile, nPartitions, dropSamples)
  }

  def importVCFs(files: Seq[String], force: Boolean = false,
    forceBGZ: Boolean = false,
    headerFile: Option[String] = None,
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false): VariantDataset = {
    val m = importVCFsGeneric(files, force, forceBGZ, headerFile, nPartitions, dropSamples)
    val extractG = Genotype.buildGenotypeExtractor(m.genotypeSignature)
    m.copy(
      rdd = m.rdd.mapValuesWithKey { case (v, (va, gs)) =>
        (va, gs.map { t =>
          val g = extractG(t)
          if (Genotype.ad(g).exists(_.length != v.nAlleles))
            null
          else {
            g.check(v.nAlleles)
            g
          }
        })
      },
      genotypeSignature = TGenotype)
  }

  def importVCFGeneric(file: String, force: Boolean = false,
    forceBGZ: Boolean = false,
    headerFile: Option[String] = None,
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false,
    callFields: Set[String] = Set.empty[String]): VariantSampleMatrix[Locus, Variant, Annotation] = {
    importVCFsGeneric(List(file), force, forceBGZ, headerFile, nPartitions, dropSamples, callFields)
  }

  def importVCFsGeneric(files: Seq[String], force: Boolean = false,
    forceBGZ: Boolean = false,
    headerFile: Option[String] = None,
    nPartitions: Option[Int] = None,
    dropSamples: Boolean = false,
    callFields: Set[String] = Set.empty[String]): VariantSampleMatrix[Locus, Variant, Annotation] = {

    val inputs = LoadVCF.globAllVCFs(hadoopConf.globAll(files), hadoopConf, force || forceBGZ)

    val header = headerFile.getOrElse(inputs.head)

    val codecs = sc.hadoopConfiguration.get("io.compression.codecs")

    if (forceBGZ)
      hadoopConf.set("io.compression.codecs",
        codecs.replaceAllLiterally("org.apache.hadoop.io.compress.GzipCodec", "is.hail.io.compress.BGzipCodecGZ"))

    val reader = new HtsjdkRecordReader(callFields)
    val vkds = LoadVCF(this, reader, header, inputs, nPartitions, dropSamples)

    hadoopConf.set("io.compression.codecs", codecs)

    vkds
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
    val ec = EvalContext(
      "v" -> TVariant(GenomeReference.GRCh37),
      "s" -> TString,
      "g" -> TGenotype,
      "sa" -> TStruct(
        "cohort" -> TString,
        "covariates" -> TStruct(
          "PC1" -> TFloat64,
          "PC2" -> TFloat64,
          "PC3" -> TFloat64,
          "age" -> TInt32,
          "isFemale" -> TBoolean
        )),
      "va" -> TStruct(
        "info" -> TStruct(
          "AC" -> TArray(TInt32),
          "AN" -> TInt32,
          "AF" -> TArray(TFloat64)),
        "transcripts" -> TArray(TStruct(
          "gene" -> TString,
          "isoform" -> TString,
          "canonical" -> TBoolean,
          "consequence" -> TString))))

    val v = Variant("16", 19200405, "C", Array("G", "CCC"))
    val s = "NA12878"
    val g = Genotype(3, 1, Array(14, 0, 12), 26, 60, Array(60, 65, 126, 0, 67, 65))
    val sa = Annotation("1KG", Annotation(0.102312, -0.61512, 0.3166666, 34, true))
    val va = Annotation(
      Annotation(IndexedSeq(40, 1), 5102, IndexedSeq(0.00784, 0.000196)),
      IndexedSeq(
        Annotation("GENE1", "GENE1.1", false, "SYN"),
        Annotation("GENE1", "GENE1.2", true, "LOF"),
        Annotation("GENE2", "GENE2.1", false, "MIS"),
        Annotation("GENE2", "GENE2.2", false, "MIS"),
        Annotation("GENE2", "GENE2.3", false, "MIS"),
        Annotation("GENE3", "GENE3.1", false, "SYN"),
        Annotation("GENE3", "GENE3.2", false, "SYN")))

    ec.set(0, v)
    ec.set(1, s)
    ec.set(2, g)
    ec.set(3, sa)
    ec.set(4, va)

    val (t, f) = Parser.parseExpr(expr, ec)
    (f(), t)
  }

  def stop() {
    sc.stop()
  }
}
