package org.broadinstitute.hail.variant.vsm

import java.nio.ByteBuffer
import org.apache.spark.{SparkEnv, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant._
import scala.collection.mutable
import scala.language.implicitConversions
import scala.reflect.ClassTag
import scala.reflect.runtime.universe._

object SparkyVSM {
  def read(sqlContext: SQLContext, dirname: String, metadata: VariantMetadata): SparkyVSM[Genotype, GenotypeStream] = {
    import RichRow._

    require(dirname.endsWith(".vds"))

    // val df = sqlContext.read.parquet(dirname + "/rdd.parquet")
    val df = sqlContext.parquetFile(dirname + "/rdd.parquet")
    new SparkyVSM(metadata, df.rdd.map(r => (r.getVariant(0), r.getGenotypeStream(1))))
  }
}

class SparkyVSM[T, S <: Iterable[(Int, T)]](val metadata: VariantMetadata,
  val rdd: RDD[(Variant, S)])
  (implicit ttt: TypeTag[T], stt: TypeTag[S], tct: ClassTag[T], sct: ClassTag[S],
    atct: ClassTag[Array[T]], vct: ClassTag[Variant],
    itct: ClassTag[(Int, T)])
  extends VariantSampleMatrix[T] {

  def nVariants: Long = rdd.count()

  def variants: RDD[Variant] = rdd.keys

  def cache(): SparkyVSM[T, S] =
    new SparkyVSM[T, S](metadata, rdd.cache())

  def repartition(nPartitions: Int) =
    new SparkyVSM[T, S](metadata, rdd.repartition(nPartitions))

  def nPartitions: Int = rdd.partitions.size

  def sparkContext: SparkContext = rdd.sparkContext

  def count(): Long = rdd.count() // should this be nVariants instead?

  def expand(): RDD[(Variant, Int, T)] =
    mapWithKeys((v, s, g) => (v, s, g))

  def write(sqlContext: SQLContext, dirname: String) {
    import sqlContext.implicits._

    require(dirname.endsWith(".vds"))

    val hConf = sparkContext.hadoopConfiguration
    hadoopMkdir(dirname, hConf)
    writeObjectFile(dirname + "/metadata.ser", hConf)(
      _.writeObject("sparky" -> metadata))

    // rdd.toDF().write.parquet(dirname + "/rdd.parquet")
    rdd.toDF().saveAsParquetFile(dirname + "/rdd.parquet")
  }

  def mapValuesWithKeys[U](f: (Variant, Int, T) => U)
    (implicit utt: TypeTag[U], uct: ClassTag[U], iuct: ClassTag[(Int, U)]): SparkyVSM[U, mutable.WrappedArray[(Int, U)]] = {
    new SparkyVSM[U, mutable.WrappedArray[(Int, U)]](metadata,
      rdd
        .map { case (v, gs) => (v, gs.map { case (s, t) => (s, f(v, s, t)) }.toArray(iuct): mutable.WrappedArray[(Int, U)]) })
  }

  def mapWithKeys[U](f: (Variant, Int, T) => U)(implicit uct: ClassTag[U]): RDD[U] =
    rdd
      .flatMap { case (v, gs) => gs.iterator.map { case (s, g) => f(v, s, g) } }

  def flatMapWithKeys[U](f: (Variant, Int, T) => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] =
    rdd
      .flatMap { case (v, gs) => gs.iterator.flatMap { case (s, g) => f(v, s, g) } }

  def filterVariants(p: (Variant) => Boolean) =
    new SparkyVSM[T, S](metadata,
      rdd.filter { case (v, gs) => p(v) })

  def filterSamples(p: (Int) => Boolean) = {
    val localitct = itct
    new SparkyVSM[T, mutable.WrappedArray[(Int, T)]](metadata,
      rdd.map { case (v, gs) =>
        (v, gs.iterator.filter { case (s, v) => p(s) }.toArray(localitct): mutable.WrappedArray[(Int, T)])
      })
  }

  def aggregateBySampleWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): RDD[(Int, U)] = {

    val localSamplesBc = sparkContext.broadcast(rdd.first()._2.map(_._1))

    // Serialize the zero value to a byte array so that we can get a new clone of it on each key
    val serializer = SparkEnv.get.serializer.newInstance()
    val zeroBuffer = serializer.serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    rdd
      .mapPartitions { (it: Iterator[(Variant, S)]) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        def createZero() = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))
        val byKeyZeroValue = Array.fill[U](localSamplesBc.value.size)(createZero())

        localSamplesBc.value.iterator
          .zip(it.foldLeft(byKeyZeroValue) { case (acc, (v, gs)) =>
            for ((sg, i) <- gs.zipWithIndex)
              acc(i) = seqOp(acc(i), v, sg._1, sg._2)
            acc
          }.iterator)
      }.foldByKey(zeroValue)(combOp)
  }

  def aggregateByVariantWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): RDD[(Variant, U)] = {

    // Serialize the zero value to a byte array so that we can get a new clone of it on each key
    val zeroBuffer = SparkEnv.get.serializer.newInstance().serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    rdd
      .map { case (v, gs) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        val zeroValue = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))

        (v, gs.foldLeft(zeroValue) { case (acc, (s, g)) =>
          seqOp(acc, v, s, g)
        })
      }
  }

  def foldBySample(zeroValue: T)(combOp: (T, T) => T): RDD[(Int, T)] = {

    val localSamplesBc = sparkContext.broadcast(rdd.first()._2.map(_._1))

    // Serialize the zero value to a byte array so that we can get a new clone of it on each key
    val serializer = SparkEnv.get.serializer.newInstance()
    val zeroBuffer = serializer.serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    rdd
      .mapPartitions { (it: Iterator[(Variant, S)]) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        def createZero() = serializer.deserialize[T](ByteBuffer.wrap(zeroArray))
        val byKeyZeroValue = Array.fill[T](localSamplesBc.value.size)(createZero())

        localSamplesBc.value.iterator
          .zip(it.foldLeft(byKeyZeroValue) { case (acc, (v, gs)) =>
            for ((sg, i) <- gs.zipWithIndex)
              acc(i) = combOp(acc(i), sg._2)
            acc
          }.iterator)
      }.foldByKey(zeroValue)(combOp)
  }

  def foldByVariant(zeroValue: T)(combOp: (T, T) => T): RDD[(Variant, T)] = {
    rdd
      .mapValues(_.foldLeft(zeroValue)((acc, ig) => combOp(acc, ig._2)))
  }
}
