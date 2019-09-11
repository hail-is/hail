package is.hail.backend.spark

import is.hail.HailContext.{createSparkConf, hailCompressionCodecs}
import is.hail.{HailContext, cxx}
import is.hail.backend.{Backend, BroadcastValue}
import is.hail.expr.ir._
import is.hail.io.fs.HadoopFS
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object SparkBackend {
  def executeJSON(ir: IR): String = HailContext.backend.executeJSON(ir)
}

class SparkBroadcastValue[T](bc: Broadcast[T]) extends BroadcastValue[T] with Serializable {
  def value: T = bc.value
}

class SparkBackend(maybeNullSparkContext: SparkContext, appName: String = "Hail", master: Option[String] = None,
                   local: String = "local[*]", minBlockSize: Long = 1L) extends Backend {

  val sc = if (maybeNullSparkContext == null)
    configureAndCreateSparkContext(appName, master, local, minBlockSize)
  else {
    checkSparkConfiguration(maybeNullSparkContext)
    maybeNullSparkContext
  }

  sc.hadoopConfiguration.set("io.compression.codecs", hailCompressionCodecs.mkString(","))

  var _hadoopFS: HadoopFS = null

  def broadcast[T : ClassTag](value: T): BroadcastValue[T] = new SparkBroadcastValue[T](sc.broadcast(value))

  def parallelizeAndComputeWithIndex[T : ClassTag, U : ClassTag](collection: Array[T])(f: (T, Int) => U): Array[U] = {
    val rdd = sc.parallelize[T](collection, numSlices = collection.length)
    rdd.mapPartitionsWithIndex { (i, it) =>
      val elt = it.next()
      assert(!it.hasNext)
      Iterator.single(f(elt, i))
    }.collect()
  }

  def getHadoopFS() = {
    if (_hadoopFS == null) {
      _hadoopFS = new HadoopFS(new SerializableHadoopConfiguration(sc.hadoopConfiguration))
    }
    _hadoopFS
  }

  override def asSpark(): SparkBackend = this

  def configureAndCreateSparkContext(appName: String, master: Option[String],
                                     local: String, blockSize: Long): SparkContext = {
    val sc = new SparkContext(createSparkConf(appName, master, local, blockSize))
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
}
