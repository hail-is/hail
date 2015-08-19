package org.broadinstitute.k3.variant.vsm

import java.io._

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.broadinstitute.k3.variant._

import scala.language.implicitConversions
import scala.reflect.ClassTag
import scala.reflect.runtime.universe._

object TupleVSM {
  def apply(sampleIds: Array[String],
            rdd: RDD[(Variant, GenotypeStream)]) =
    new TupleVSM(sampleIds,
      rdd.flatMap { case (v, gs) => gs.iterator.map { case (s, g) => (v, s, g) } })

  def read(sqlContext: SQLContext, dirname: String): TupleVSM[Genotype] = {
    require(dirname.endsWith(".vds"))

    val metadataOis = new ObjectInputStream(new FileInputStream(dirname + "/metadata.ser"))

    val sampleIdsObj = metadataOis.readObject()
    val df = sqlContext.read.parquet(dirname + "/rdd.parquet")

    val sampleIds = sampleIdsObj match {
      case t: Array[String] => t
      case _ => throw new ClassCastException
    }

    import RichRow._
    this(sampleIds,
      df
      .rdd
      .map(r => (r.getVariant(0), r.getGenotypeStream(1))))
  }
}

class TupleVSM[T](val sampleIds: Array[String],
                  val rdd: RDD[(Variant, Int, T)])(implicit ttt: TypeTag[T], tct: ClassTag[T])
  extends VariantSampleMatrix[T] {

  def nSamples: Int = sampleIds.length

  // FIXME wrong
  def nVariants: Long = rdd.count()

  def cache(): TupleVSM[T] =
    new TupleVSM[T](sampleIds, rdd.cache())

  def repartition(nPartitions: Int) =
    new TupleVSM[T](sampleIds, rdd.repartition(nPartitions))

  def nPartitions: Int = rdd.partitions.size

  def variants: Array[Variant] = rdd.map(_._1).collect()

  def sparkContext: SparkContext = rdd.sparkContext

  def count(): Long = rdd.count() // should this be nVariants instead?

  def expand(): RDD[(Variant, Int, T)] = rdd

  def write(sqlContext: SQLContext, dirname: String) {
    require(dirname.endsWith(".vds"))

    new File(dirname).mkdir()

    val metadataOos = new ObjectOutputStream(new FileOutputStream(dirname + "/metadata.ser"))
    metadataOos.writeObject(sampleIds)

    import sqlContext.implicits._

    val df = rdd.toDF()
    df.write.parquet(dirname + "/rdd.parquet")
  }

  def mapValuesWithKeys[U](f: (Variant, Int, T) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): TupleVSM[U] = {
    new TupleVSM[U](sampleIds,
      rdd.map { case (v, s, g) => (v, s, f(v, s, g)) })
  }

  def filterVariants(p: (Variant) => Boolean) = {
    new TupleVSM[T](sampleIds,
      rdd.filter { case (v, s, g) => p(v) })
  }

  def filterSamples(p: (Int) => Boolean) = {
    new TupleVSM[T](sampleIds,
      rdd.filter { case (v, s, g) => p(s) })
  }

  def aggregateBySampleWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): Map[Int, U] = {

    rdd
    .map { case (v, s, g) => (s, (v, s, g)) }
    .aggregateByKey(zeroValue)({ case (u, (v, s, g)) => seqOp(u, v, s, g) }, combOp)
    .collectAsMap().toMap
  }

  def aggregateByVariantWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): RDD[(Variant, U)] = {

    rdd
    .map { case (v, s, g) => (v, (v, s, g)) }
    .aggregateByKey(zeroValue)({ case (u, (v, s, g)) => seqOp(u, v, s, g) }, combOp)
  }

  def foldBySample(zeroValue: T)(combOp: (T, T) => T): Map[Int, T] = {
    rdd
    .map { case (v, s, g) => (s, g) }
    .foldByKey(zeroValue)(combOp)
    .collectAsMap().toMap
  }

  def foldByVariant(zeroValue: T)(combOp: (T, T) => T): RDD[(Variant, T)] = {
    rdd
    .map { case (v, s, g) => (v, g) }
    .foldByKey(zeroValue)(combOp)
  }
}
