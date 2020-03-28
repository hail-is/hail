package is.hail.backend.spark

import is.hail.backend.{Backend, BroadcastValue, HailTaskContext}
import is.hail.io.fs.{FS, HadoopFS}
import is.hail.utils._
import org.apache.spark.{ProgressBarBuilder, SparkConf, SparkContext, TaskContext}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.SparkSession

import scala.collection.mutable
import scala.reflect.ClassTag

class SparkBroadcastValue[T](bc: Broadcast[T]) extends BroadcastValue[T] with Serializable {
  def value: T = bc.value
}

class SparkTaskContext(ctx: TaskContext) extends HailTaskContext {
  type BackendType = SparkBackend
  override def stageId(): Int = ctx.stageId()
  override def partitionId(): Int = ctx.partitionId()
  override def attemptNumber(): Int = ctx.attemptNumber()
}

object SparkBackend {
  private var theSparkBackend: SparkBackend = _

  def checkSparkCompatibility(jarVersion: String, sparkVersion: String): Unit = {
    def majorMinor(version: String): String = version.split("\\.", 3).take(2).mkString(".")

    if (majorMinor(jarVersion) != majorMinor(sparkVersion))
      fatal(s"This Hail JAR was compiled for Spark $jarVersion, cannot run with Spark $sparkVersion.\n" +
        s"  The major and minor versions must agree, though the patch version can differ.")
    else if (jarVersion != sparkVersion)
      warn(s"This Hail JAR was compiled for Spark $jarVersion, running with Spark $sparkVersion.\n" +
        s"  Compatibility is not guaranteed.")
  }

  def createSparkConf(appName: String, master: String,
    local: String, blockSize: Long): SparkConf = {
    require(blockSize >= 0)
    checkSparkCompatibility(is.hail.HAIL_SPARK_VERSION, org.apache.spark.SPARK_VERSION)

    val conf = new SparkConf().setAppName(appName)

    if (master != null)
      conf.setMaster(master)
    else {
      if (!conf.contains("spark.master"))
        conf.setMaster(local)
    }

    conf.set("spark.logConf", "true")
    conf.set("spark.ui.showConsoleProgress", "false")

    conf.set("spark.kryoserializer.buffer.max", "1g")
    conf.set("spark.driver.maxResultSize", "0")

    conf.set(
      "spark.hadoop.io.compression.codecs",
      "org.apache.hadoop.io.compress.DefaultCodec," +
        "is.hail.io.compress.BGzipCodec," +
        "is.hail.io.compress.BGzipCodecTbi," +
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
    conf
  }

  def configureAndCreateSparkContext(appName: String, master: String,
    local: String, blockSize: Long): SparkContext = {
    new SparkContext(createSparkConf(appName, master, local, blockSize))
  }

  def checkSparkConfiguration(sc: SparkContext) {
    val conf = sc.getConf

    val problems = new mutable.ArrayBuffer[String]

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

  def hailCompressionCodecs: Array[String] = Array(
    "org.apache.hadoop.io.compress.DefaultCodec",
    "is.hail.io.compress.BGzipCodec",
    "is.hail.io.compress.BGzipCodecTbi",
    "org.apache.hadoop.io.compress.GzipCodec")

  /**
    * If a SparkBackend has already been initialized, this function returns it regardless of the
    * parameters with which it was initialized.
    *
    * Otherwise, it initializes and returns a new HailContext.
    */
  def getOrCreate(sc: SparkContext = null,
    appName: String = "Hail",
    master: String = null,
    local: String = "local[*]",
    quiet: Boolean = false,
    minBlockSize: Long = 1L): SparkBackend = synchronized {
    if (theSparkBackend == null)
      return SparkBackend(sc, appName, master, local, quiet, minBlockSize)

    // there should be only one SparkContext
    assert(sc == null || (sc eq theSparkBackend.sc))

    val initializedMinBlockSize =
      theSparkBackend.sc.getConf.getLong("spark.hadoop.mapreduce.input.fileinputformat.split.minsize", 0L) / 1024L / 1024L
    if (minBlockSize != initializedMinBlockSize)
      warn(s"Requested minBlockSize $minBlockSize, but already initialized to $initializedMinBlockSize.  Ignoring requested setting.")

    if (master != null) {
      val initializedMaster = theSparkBackend.sc.master
      if (master != initializedMaster)
        warn(s"Requested master $master, but already initialized to $initializedMaster.  Ignoring requested setting.")
    }

    theSparkBackend
  }

  def apply(sc: SparkContext = null,
    appName: String = "Hail",
    master: String = null,
    local: String = "local[*]",
    quiet: Boolean = false,
    minBlockSize: Long = 1L): SparkBackend = synchronized {
    require(theSparkBackend == null)

    var sc1 = sc
    if (sc1 == null)
      sc1 = configureAndCreateSparkContext(appName, master, local, minBlockSize)

    sc1.hadoopConfiguration.set("io.compression.codecs", hailCompressionCodecs.mkString(","))

    checkSparkConfiguration(sc1)

    if (!quiet)
      ProgressBarBuilder.build(sc1)

    sc1.uiWebUrl.foreach(ui => info(s"SparkUI: $ui"))

    theSparkBackend = new SparkBackend(sc1)
    theSparkBackend
  }

  def clear(): Unit = synchronized {
    if (theSparkBackend != null) {
      theSparkBackend.sc.stop()
      theSparkBackend = null
    }
  }
}

class SparkBackend(val sc: SparkContext) extends Backend {
  lazy val sparkSession: SparkSession = SparkSession.builder().config(sc.getConf).getOrCreate()

  val fs: HadoopFS = new HadoopFS(new SerializableHadoopConfiguration(sc.hadoopConfiguration))

  val fsBc: Broadcast[FS] = sc.broadcast(fs)

  override val cache: SparkValueCache = SparkValueCache()

  def broadcast[T : ClassTag](value: T): BroadcastValue[T] = new SparkBroadcastValue[T](sc.broadcast(value))

  def parallelizeAndComputeWithIndex[T : ClassTag, U : ClassTag](collection: Array[T])(f: (T, Int) => U): Array[U] = {
    val rdd = sc.parallelize[T](collection, numSlices = collection.length)
    rdd.mapPartitionsWithIndex { (i, it) =>
      HailTaskContext.setTaskContext(new SparkTaskContext(TaskContext.get))
      val elt = it.next()
      assert(!it.hasNext)
      Iterator.single(f(elt, i))
    }.collect()
  }


  def startProgressBar() {
    ProgressBarBuilder.build(sc)
  }

  override def asSpark(): SparkBackend = this

  def clear(): Unit = SparkBackend.clear()
}
