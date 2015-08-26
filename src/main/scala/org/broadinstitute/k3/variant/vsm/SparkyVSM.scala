package org.broadinstitute.k3.variant.vsm

import java.io._

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.broadinstitute.k3.Utils._
import org.broadinstitute.k3.variant._

import scala.language.implicitConversions
import scala.reflect.ClassTag
import scala.reflect.runtime.universe._

object SparkyVSM {
  def read(sqlContext: SQLContext, dirname: String): SparkyVSM[Genotype, GenotypeStream] = {
    require(dirname.endsWith(".vds"))

    val metadataOis = new ObjectInputStream(new FileInputStream(dirname + "/metadata.ser"))

    val sampleIdsObj = metadataOis.readObject()
    val df = sqlContext.read.parquet(dirname + "/rdd.parquet")

    val sampleIds = sampleIdsObj match {
      case t: Array[String] => t
      case _ => throw new ClassCastException
    }

    import RichRow._
    new SparkyVSM(sampleIds, df.rdd.map(r => (r.getVariant(0), r.getGenotypeStream(1))))
  }
}

class SparkyVSM[T, S <: Iterable[(Int, T)]](val sampleIds: Array[String],
                                            val rdd: RDD[(Variant, S)])
                                           (implicit ttt: TypeTag[T], stt: TypeTag[S], tct: ClassTag[T], sct: ClassTag[S])
  extends VariantSampleMatrix[T] {
  def nSamples: Int = sampleIds.length

  def nVariants: Long = rdd.count()

  def cache(): SparkyVSM[T, S] =
    new SparkyVSM[T, S](sampleIds, rdd.cache())

  def repartition(nPartitions: Int) =
    new SparkyVSM[T, S](sampleIds, rdd.repartition(nPartitions))

  def nPartitions: Int = rdd.partitions.size

  def variants: Array[Variant] = rdd.map(_._1).collect()

  def sparkContext: SparkContext = rdd.sparkContext

  def count(): Long = rdd.count() // should this be nVariants instead?

  def expand(): RDD[(Variant, Int, T)] = {
    rdd.flatMap { case (v, gs) =>
      gs.map { case (s, g) => (v, s, g) }
    }
  }

  def write(sqlContext: SQLContext, dirname: String) {
    require(dirname.endsWith(".vds"))

    new File(dirname).mkdir()

    val metadataOos = new ObjectOutputStream(new FileOutputStream(dirname + "/metadata.ser"))
    metadataOos.writeObject(sampleIds)

    import sqlContext.implicits._

    val df = rdd.toDF()
    df.write.parquet(dirname + "/rdd.parquet")
  }

  def mapValuesWithKeys[U](f: (Variant, Int, T) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): SparkyVSM[U, Vector[(Int, U)]] = {
    new SparkyVSM[U, Vector[(Int, U)]](sampleIds,
      rdd
      .map { case (v, gs) => (v, gs.map { case (s, t) => (s, f(v, s, t)) }.toVector) })
  }

  def flatMapWithKeys[U](f: (Variant, Int, T) => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] = {
    rdd
    .flatMap{ case (v, gs) => gs.flatMap{ case (s, g) => f(v, s, g) }}
  }

  // FIXME push down into reader: add VariantSampleDataframeMatrix?
  def filterVariants(p: (Variant) => Boolean) = {
    new SparkyVSM[T, S](sampleIds,
      rdd.filter { case (v, gs) => p(v) })
  }

  def filterSamples(p: (Int) => Boolean) = {
    new SparkyVSM[T, Vector[(Int, T)]](sampleIds,
      rdd.map { case (v, gs) =>
        (v, gs.filter { case (s, v) => p(s) }.toVector)
      })
  }

  def aggregateBySampleWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): Map[Int, U] = {

    val zeroValueBySample = rdd.first()._2.map { case (s, g) => (s, zeroValue) }.toVector

    rdd
    .aggregate(zeroValueBySample)({
      case (acc, (v, gs)) =>
        acc.zipWith[(Int, T), (Int, U)](gs, { case ((s1, a), (s2, g)) => {
          assert(s1 == s2)
          (s1, seqOp(a, v, s1, g))
        }
        })
    },
    (acc1, acc2) => acc1.zipWith[(Int, U), (Int, U)](acc2, { case ((s1, a1), (s2, a2)) => (s1, combOp(a1, a2)) }))
    .toMap
  }

  def aggregateByVariantWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): RDD[(Variant, U)] = {
    rdd
    .map { case (v, gs) => (v, (v, gs)) }
    .aggregateByKey(zeroValue)({
      case (acc, (v, gs)) => gs.aggregate(acc)({ case (acc2, (s, g)) => seqOp(acc2, v, s, g) }, combOp)
    },
    combOp)
  }

  def foldBySample(zeroValue: T)(combOp: (T, T) => T): Map[Int, T] = {
    val bySampleZeroValue = rdd.first()._2.map { case (s, g) => (s, zeroValue) }.toVector
    rdd
    .aggregate(bySampleZeroValue)({
      case (acc, (v, gs)) => acc.zipWith[(Int, T), (Int, T)](gs, { case ((s1, a), (s2, g)) => {
        assert(s1 == s2)
        (s1, combOp(a, g))
      }
      })
    }, (a1, a2) => a1.zipWith[(Int, T), (Int, T)](a2, { case ((s1, a1), (s2, a2)) => (s1, combOp(a1, a2)) }))
    .toMap
  }

  def foldByVariant(zeroValue: T)(combOp: (T, T) => T): RDD[(Variant, T)] = {
    rdd
    .mapValues(gs => gs.map {
      case (s, gs) => gs
    }.fold(zeroValue)(combOp))
    .foldByKey(zeroValue)(combOp)
  }
}
