package org.broadinstitute.hail.variant.vsm

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.Utils._

import scala.language.implicitConversions
import scala.reflect.ClassTag
import scala.reflect.runtime.universe._

object TupleVSM {
  def apply(metadata: VariantMetadata,
    rdd: RDD[(Variant, GenotypeStream)]) =
    new TupleVSM(metadata,
      rdd.flatMap { case (v, gs) => gs.iterator.map { case (s, g) => (v, s, g) } })

  def read(sqlContext: SQLContext, dirname: String, metadata: VariantMetadata): TupleVSM[Genotype] = {
    import RichRow._
    require(dirname.endsWith(".vds"))

    // val df = sqlContext.read.parquet(dirname + "/rdd.parquet")
    val df = sqlContext.read.parquet(dirname + "/rdd.parquet")
    df.printSchema()
    new TupleVSM[Genotype](metadata,
      df
        .rdd
        .map(r => (r.getVariant(0), r.getInt(1), r.getGenotype(2))))
  }
}

class TupleVSM[T](val metadata: VariantMetadata,
  val rdd: RDD[(Variant, Int, T)])(implicit ttt: TypeTag[T], tct: ClassTag[T])
  extends VariantSampleMatrix[T] {

  override def nSamples: Int = sampleIds.length

  def variants: RDD[Variant] = rdd.map(_._1).distinct()

  def nVariants: Long = variants.count()

  def cache(): TupleVSM[T] =
    new TupleVSM[T](metadata, rdd.cache())

  def repartition(nPartitions: Int) =
    new TupleVSM[T](metadata, rdd.repartition(nPartitions))

  def nPartitions: Int = rdd.partitions.size

  def sparkContext: SparkContext = rdd.sparkContext

  def count(): Long = rdd.count() // should this be nVariants instead?

  def expand(): RDD[(Variant, Int, T)] = rdd

  def write(sqlContext: SQLContext, dirname: String) {
    import sqlContext.implicits._

    require(dirname.endsWith(".vds"))

    val hConf = sparkContext.hadoopConfiguration
    hadoopMkdir(dirname, hConf)
    writeObjectFile(dirname + "/metadata.ser", hConf)(
      _.writeObject("tuple" -> metadata)
    )

    // rdd.toDF().write.parquet(dirname + "/rdd.parquet")
    rdd.toDF().write.parquet(dirname + "/rdd.parquet")
  }

  def mapValuesWithKeys[U](f: (Variant, Int, T) => U)(implicit utt: TypeTag[U], uct: ClassTag[U], iuct: ClassTag[(Int, U)]): TupleVSM[U] = {
    new TupleVSM[U](metadata,
      rdd.map { case (v, s, g) => (v, s, f(v, s, g)) })
  }

  def mapWithKeys[U](f: (Variant, Int, T) => U)(implicit uct: ClassTag[U]): RDD[U] =
    rdd.map[U](f.tupled)

  def flatMapWithKeys[U](f: (Variant, Int, T) => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] =
    rdd.flatMap[U](f.tupled)

  def filterVariants(p: (Variant) => Boolean) = {
    new TupleVSM[T](metadata,
      rdd.filter { case (v, s, g) => p(v) })
  }

  def filterSamples(p: (Int) => Boolean) = {
    new TupleVSM[T](metadata,
      rdd.filter { case (v, s, g) => p(s) })
  }

  def aggregateBySampleWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): RDD[(Int, U)] = {

    rdd
      .map { case (v, s, g) => (s, (v, s, g)) }
      .aggregateByKey(zeroValue)({ case (u, (v, s, g)) => seqOp(u, v, s, g) }, combOp)
  }

  def aggregateByVariantWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): RDD[(Variant, U)] = {

    rdd
      .map { case (v, s, g) => (v, (v, s, g)) }
      .aggregateByKey(zeroValue)({ case (u, (v, s, g)) => seqOp(u, v, s, g) }, combOp)
  }

  def foldBySample(zeroValue: T)(combOp: (T, T) => T): RDD[(Int, T)] = {
    rdd
      .map { case (v, s, g) => (s, g) }
      .foldByKey(zeroValue)(combOp)
  }

  def foldByVariant(zeroValue: T)(combOp: (T, T) => T): RDD[(Variant, T)] = {
    rdd
      .map { case (v, s, g) => (v, g) }
      .foldByKey(zeroValue)(combOp)
  }
}
