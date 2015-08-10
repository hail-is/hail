package org.broadinstitute.k3.variant

import java.io._

import org.apache.spark.broadcast.Broadcast

import scala.collection.Map
import scala.language.implicitConversions

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

import scala.reflect.ClassTag

import org.broadinstitute.k3.Utils._

// FIXME implement variants WithKeys
class VariantSampleMatrix[T, S <: Iterable[(Int, T)]](val sampleIds: Array[String],
                                                      val rdd: RDD[(Variant, S)]) {
  def nSamples: Int = sampleIds.length
  def nVariants: Long = rdd.count()

  def cache(): VariantSampleMatrix[T, S] = {
    new VariantSampleMatrix[T, S](sampleIds, rdd.cache())
  }

  def count(): Long = rdd.count()

  def expand(): RDD[(Variant, Int, T)] = {
    rdd.flatMap { case (v, gs) =>
      gs.map { case (s, g) => (v, s, g) }
    }
  }

  def mapValuesWithKeys[U](f: (Variant, Int, T) => U): VariantSampleMatrix[U, Vector[(Int, U)]] = {
    new VariantSampleMatrix[U, Vector[(Int, U)]](sampleIds,
      rdd
      .map{ case (v, gs) => (v, gs.map{ case (s, t) => (s, f(v, s, t)) }.toVector) })
  }

  def mapValues[U](f: (T) => U): VariantSampleMatrix[U, Vector[(Int, U)]] = {
    mapValuesWithKeys((v, s, g) => f(g))
  }

  // FIXME push down into reader: add VariantSampleDataframeMatrix?
  def filterVariants(p: (Variant) => Boolean) = {
    new VariantSampleMatrix[T, S](sampleIds,
      rdd.filter { case (v, gs) => p(v) })
  }

  def filterSamples(p: (Int) => Boolean) = {
    new VariantSampleMatrix[T, Vector[(Int, T)]](sampleIds,
      rdd.map { case (v, gs) =>
        (v, gs.filter { case (s, v) => p(s) }.toVector)
      })
  }

  def aggregateBySampleWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U): Map[Int, U] = {

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

  def aggregateBySample[U](zeroValue: U)(
    seqOp: (U, T) => U,
    combOp: (U, U) => U): Map[Int, U] =
    aggregateBySampleWithKeys(zeroValue)((e, v, s, g) => seqOp(e, g), combOp)

  def aggregateByVariantWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit ut: ClassTag[U]): Map[Variant, U] = {
    rdd
    .map { case (v, gs) => (v, (v, gs)) }
    .aggregateByKey(zeroValue)({
      case (acc, (v, gs)) => gs.aggregate(acc)({ case (acc2, (s, g)) => seqOp(acc2, v, s, g) }, combOp)
    },
    combOp)
    .collectAsMap()
  }

  def aggregateByVariant[U](zeroValue: U)(
    seqOp: (U, T) => U,
    combOp: (U, U) => U)(implicit ut: ClassTag[U]): Map[Variant, U] =
    aggregateByVariantWithKeys(zeroValue)((e, v, s, g) => seqOp(e, g), combOp)

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

  def reduceByVariant(combOp: (T, T) => T)(implicit tt: ClassTag[T], st: ClassTag[S]): Map[Variant, T] = {
    rdd
    .mapValues(gs => gs.map {
      case (s, gs) => gs
    }.reduce(combOp))
    .reduceByKey(combOp)
    .collectAsMap()
  }

  def foldByVariant(zeroValue: T)(combOp: (T, T) => T)(implicit tt: ClassTag[T], st: ClassTag[S]): Map[Variant, T] = {
    rdd
    .mapValues(gs => gs.map {
      case (s, gs) => gs
    }.fold(zeroValue)(combOp))
    .foldByKey(zeroValue)(combOp)
    .collectAsMap()
  }
}
