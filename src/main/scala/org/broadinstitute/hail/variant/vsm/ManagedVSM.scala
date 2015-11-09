package org.broadinstitute.hail.variant.vsm

import java.io._

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant._

import scala.language.implicitConversions
import scala.reflect.ClassTag
import scala.reflect.runtime.universe._

object ManagedVSM {
  def read(sqlContext: SQLContext, dirname: String, metadata: VariantMetadata): ManagedVSM[Genotype] = {
    import RichRow._

    require(dirname.endsWith(".vds"))

    val df = sqlContext.read.parquet(dirname + "/rdd.parquet")
    new ManagedVSM(metadata, df.rdd.map(r => (r.getVariant(0), r.getGenotypeStream(1))), (v, s, g) => g, _ => true)
  }
}

class ManagedVSM[T](val metadata: VariantMetadata,
  val rdd: RDD[(Variant, GenotypeStream)],
  mapFn: (Variant, Int, Genotype) => T,
  samplePredicate: (Int) => Boolean)(implicit ttt: TypeTag[T], tct: ClassTag[T])
  extends VariantSampleMatrix[T] {

  def nVariants: Long = rdd.count()

  def variants: RDD[Variant] = rdd.keys

  def nPartitions: Int = rdd.partitions.size

  def repartition(nPartitions: Int) =
    new ManagedVSM[T](metadata, rdd.repartition(nPartitions), mapFn, samplePredicate)

  def sparkContext: SparkContext = rdd.sparkContext

  def cache(): ManagedVSM[T] = {
    val localSamplePredicate = samplePredicate
    val localMapFn = mapFn
    new ManagedVSM[T](metadata, rdd.cache(), localMapFn, localSamplePredicate)
  }

  def count(): Long = rdd.count() // should this be nVariants instead?

  def expand(): RDD[(Variant, Int, T)] = {
    val localSamplePredicate = samplePredicate
    val localMapFn = mapFn

    rdd.flatMap { case (v, gs) => gs
      .iterator
      .filter { case (s, g) => localSamplePredicate(s) }
      .map { case (s, g) => (v, s, localMapFn(v, s, g)) }
    }
  }

  def write(sqlContext: SQLContext, dirname: String) {
    import sqlContext.implicits._

    require(dirname.endsWith(".vds"))

    val hConf = sparkContext.hadoopConfiguration
    hadoopMkdir(dirname, hConf)
    writeObjectFile(dirname + "/metadata.ser", hConf)(
      _.writeObject("managed" -> metadata))

    rdd.toDF().write.parquet(dirname + "/rdd.parquet")
  }

  def mapValuesWithKeys[U](f: (Variant, Int, T) => U)(implicit utt: TypeTag[U], uct: ClassTag[U], iuct: ClassTag[(Int, U)]): ManagedVSM[U] = {
    val localSamplePredicate = samplePredicate
    val localMapFn = mapFn
    new ManagedVSM[U](metadata, rdd, (v, s, g) => f(v, s, localMapFn(v, s, g)), localSamplePredicate)
  }

  def mapWithKeys[U](f: (Variant, Int, T) => U)(implicit uct: ClassTag[U]): RDD[U] =
  // FIXME
    throw new NotImplementedError()


  def flatMapWithKeys[U](f: (Variant, Int, T) => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] =
  // FIXME
    throw new NotImplementedError()

  def filterVariants(p: (Variant) => Boolean) = {
    val localSamplePredicate = samplePredicate
    val localMapFn = mapFn
    new ManagedVSM[T](metadata, rdd.filter(t => p(t._1)), localMapFn, localSamplePredicate)
  }

  def filterSamples(p: (Int) => Boolean) = {
    val localSamplePredicate = samplePredicate
    val localMapFn = mapFn
    new ManagedVSM[T](metadata, rdd, localMapFn, id => localSamplePredicate(id) && p(id))
  }

  def aggregateBySampleWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): Map[Int, U] = {
    val localMapFn = mapFn

    val filteredNums = (0 until nSamples).filter(samplePredicate)

    val numFilteredNum = Array.fill[Int](nSamples)(-1)
    for ((i, j) <- filteredNums.zipWithIndex)
      numFilteredNum(i) = j
    val numFilteredNumBroadcast = rdd.sparkContext.broadcast(numFilteredNum)

    val a = rdd.aggregate(Vector.fill[U](filteredNums.length)(zeroValue))({
      case (acc, (v, gs)) => {
        gs
          .foldLeft(acc)({
          case (acc2, (s, g)) => {
            val j = numFilteredNumBroadcast.value(s)
            if (j != -1)
              acc2.updated(j, seqOp(acc2(j), v, s, localMapFn(v, s, g)))
            else
              acc2
          }
        })
      }
    },
    (v1: Vector[U], v2: Vector[U]) => v1.zipWith(v2, combOp))

    filteredNums
      .zip(a)
      .toMap
  }

  def aggregateByVariantWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): RDD[(Variant, U)] = {
    val localSamplePredicate = samplePredicate
    val localMapFn = mapFn

    rdd.aggregateByKey(zeroValue)(
      (x, gs) => gs
        .iterator
        .filter({ case (s, g) => localSamplePredicate(s) })
        .aggregate(x)({ case (x2, (s, g)) => seqOp(x2, gs.variant, s, localMapFn(gs.variant, s, g)) }, combOp),
      combOp)
  }

  def foldBySample(zeroValue: T)(combOp: (T, T) => T): Map[Int, T] = {
    val localSamplePredicate = samplePredicate
    val localMapFn = mapFn

    val filteredNums = (0 until nSamples).filter(samplePredicate)
    val nFilteredNums = filteredNums.length
    println("nFilteredNums: " + nFilteredNums)

    val numFilteredNum = Array.fill[Int](nSamples)(-1)
    for ((i, j) <- filteredNums.zipWithIndex)
      numFilteredNum(i) = j
    val numFilteredNumBroadcast = rdd.sparkContext.broadcast(numFilteredNum)

    val vCombOp: (Vector[T], Vector[T]) => Vector[T] =
      (a1, a2) => a1.iterator.zip(a2.iterator).map(combOp.tupled).toVector
    val a = rdd
      .aggregate(Vector.fill[T](nFilteredNums)(zeroValue))({
      case (acc, (v, gs)) => acc.iterator.zip(gs.iterator
        .filter({ case (s, g) => localSamplePredicate(s) })
        .map({ case (s, g) => localMapFn(v, s, g) }))
        .map(combOp.tupled)
        .toVector
    },
    vCombOp)

    filteredNums.toVector.zip(a).toMap
  }

  def foldByVariant(zeroValue: T)(combOp: (T, T) => T): RDD[(Variant, T)] = {
    val localSamplePredicate = samplePredicate
    val localMapFn = mapFn

    rdd
      .mapValues(gs => gs.iterator
      .filter({ case (s, g) => localSamplePredicate(s) })
      .map({ case (s, g) => localMapFn(gs.variant, s, g) })
      .fold(zeroValue)(combOp))
      .foldByKey(zeroValue)(combOp)
  }
}
