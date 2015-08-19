package org.broadinstitute.k3.variant

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.broadinstitute.k3.variant.vsm.{ManagedVSM, SparkyVSM, TupleVSM}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe._

object VariantSampleMatrix {
  def apply(vsmtype: String,
            sampleIds: Array[String],
            rdd: RDD[(Variant, GenotypeStream)]): VariantSampleMatrix[Genotype] = {
    vsmtype match {
      case "managed" => new ManagedVSM(sampleIds, rdd, (v, s, g) => g, _ => true)
      case "sparky" => new SparkyVSM(sampleIds, rdd)
      case "tuple" => TupleVSM(sampleIds, rdd)
    }
  }

  def read(sqlContext: SQLContext, vsmtype: String, dirname: String) = vsmtype match {
    case "managed" => ManagedVSM.read(sqlContext, dirname)
    case "sparky" => SparkyVSM.read(sqlContext, dirname)
    case "tuple" => TupleVSM.read(sqlContext, dirname)
  }
}

// FIXME all maps should become RDDs
abstract class VariantSampleMatrix[T] {
  def sampleIds: Array[String]

  def nSamples: Int

  def nVariants: Long

  def nPartitions: Int

  def sparkContext: SparkContext

  def cache(): VariantSampleMatrix[T]

  def repartition(nPartitions: Int): VariantSampleMatrix[T]

  def count(): Long

  def expand(): RDD[(Variant, Int, T)]

  def write(sqlContext: SQLContext, dirname: String)

  def mapValuesWithKeys[U](f: (Variant, Int, T) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): VariantSampleMatrix[U]

  def mapValues[U](f: (T) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): VariantSampleMatrix[U] = {
    mapValuesWithKeys((v, s, g) => f(g))
  }

  def filterVariants(p: (Variant) => Boolean): VariantSampleMatrix[T]

  def filterSamples(p: (Int) => Boolean): VariantSampleMatrix[T]

  def aggregateBySampleWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): Map[Int, U]

  def aggregateBySample[U](zeroValue: U)(
    seqOp: (U, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): Map[Int, U] =
    aggregateBySampleWithKeys(zeroValue)((e, v, s, g) => seqOp(e, g), combOp)

  def aggregateByVariantWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): RDD[(Variant, U)]

  def aggregateByVariant[U](zeroValue: U)(
    seqOp: (U, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): RDD[(Variant, U)] =
    aggregateByVariantWithKeys(zeroValue)((e, v, s, g) => seqOp(e, g), combOp)

  def foldBySample(zeroValue: T)(combOp: (T, T) => T): Map[Int, T]

  def foldByVariant(zeroValue: T)(combOp: (T, T) => T): RDD[(Variant, T)]
}
