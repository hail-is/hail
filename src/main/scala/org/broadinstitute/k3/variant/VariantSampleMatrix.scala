package org.broadinstitute.k3.variant

import java.io._

import org.apache.spark.broadcast.Broadcast

import scala.collection.Map
import scala.language.implicitConversions

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

import scala.reflect.ClassTag

// FIXME move to own file
class VariantSampleMatrix[T : ClassTag](sampleIds: Array[String],
                             rdd: RDD[(Variant, GenotypeStream)],
                             mapFn: (Genotype) => T,
                             samplePredicate: (Int) => Boolean) {
  def nSamples: Int = sampleIds.length

  def write(sqlContext: SQLContext, dirname: String) {
    require(dirname.endsWith(".vds"))

    new File(dirname).mkdir()

    val metadataOos = new ObjectOutputStream(new FileOutputStream(dirname + "/metadata.ser"))
    metadataOos.writeObject(sampleIds)

    import sqlContext.implicits._

    val df = rdd.toDF()
    df.write.parquet(dirname + "/rdd.parquet")
  }

  def cache(): VariantSampleMatrix[T] = new VariantSampleMatrix[T](sampleIds, rdd.cache(), mapFn, samplePredicate)

  def mapValues[U : ClassTag](f: (T) => U) : VariantSampleMatrix[U] = {
    new VariantSampleMatrix[U](sampleIds, rdd, f compose mapFn, samplePredicate)
  }

  // FIXME push down into reader: Add VariantSampleDataframeMatrix
  def filterVariants(p: (Variant) => Boolean) = {
    new VariantSampleMatrix(sampleIds, rdd.filter(t => p(t._1)), mapFn, samplePredicate)
  }

  def filterSamples(p: (Int) => Boolean) = {
    new VariantSampleMatrix(sampleIds, rdd, mapFn, id => samplePredicate(id) && p(id))
  }

  def aggregateBySample[U : ClassTag](zeroValue: U)(
    seqOp: (U, T) => U,
    combOp: (U, U) => U): Map[Int, U] = {
    val filteredNums = (0 until nSamples).filter(samplePredicate)

    val numFilteredNum = Array.fill[Int](nSamples)(-1)
    for ((i, j) <- filteredNums.zipWithIndex)
      numFilteredNum(i) = j
    val numFilteredNumBroadcast = rdd.sparkContext.broadcast(numFilteredNum)

    val a = rdd.aggregate(Array.fill[U](filteredNums.length)(zeroValue))(
      (a: Array[U], t: (Variant, GenotypeStream)) => {
        val gs = t._2
        for ((i, g) <- gs) {
          val j = numFilteredNumBroadcast.value(i)
          if (j != -1)
            a(j) = seqOp(a(j), mapFn(g))
        }
        a
      },
      (a1: Array[U], a2: Array[U]) => {
        for (j <- a1.indices)
          a1(j) = combOp(a1(j), a2(j))
        a1
      })

    filteredNums
      .zip(a)
      .toMap
  }

  def aggregateByVariant[U : ClassTag](zeroValue: U)(
    seqOp: (U, T) => U,
    combOp: (U, U) => U): Map[Variant, U] = {
    rdd.aggregateByKey(zeroValue)(
      (x, gs) => gs
        .filter(sg => samplePredicate(sg._1))
        .aggregate(x)((x2, sg) => seqOp(x2, mapFn(sg._2)), combOp),
      combOp)
      .collectAsMap()
  }

  def reduceBySample(combOp: (T, T) => T): Map[Int, T] = {
    val filteredNums = (0 until nSamples).filter(samplePredicate)

    val numFilteredNum = Array.fill[Int](nSamples)(-1)
    for ((i, j) <- filteredNums.zipWithIndex)
      numFilteredNum(i) = j
    val numFilteredNumBroadcast = rdd.sparkContext.broadcast(numFilteredNum)

    val a = rdd
      .map(t => t._2.filter(sg => samplePredicate(sg._1)).map(sg => mapFn(sg._2)).toArray)
      .reduce((a1, a2) => {
      for (i <- a1.indices)
        a1(i) = combOp(a1(i), a2(i))
      a1
    })

    filteredNums
      .zip(a)
      .toMap
  }

  def reduceByVariant(combOp: (T, T) => T): Map[Variant, T] = {
    rdd
      .mapValues(gs => gs.filter(sg => samplePredicate(sg._1)).map(sg => mapFn(sg._2)).reduce(combOp))
      .reduceByKey(combOp)
      .collectAsMap()
  }
}

class VariantDataset(val sampleIds: Array[String],
                     val rdd: RDD[(Variant, GenotypeStream)])
  extends VariantSampleMatrix[Genotype](sampleIds, rdd, x => x, _ => true)

object VariantDataset {
  def read(sqlContext: SQLContext, dirname: String): VariantDataset = {
    require(dirname.endsWith(".vds"))

    val metadataOis = new ObjectInputStream(new FileInputStream(dirname + "/metadata.ser"))

    val sampleIdsObj = metadataOis.readObject()
    val df = sqlContext.read.parquet(dirname + "/rdd.parquet")

    val sampleIds = sampleIdsObj match {
      case t: Array[String] => t
      case _ => throw new ClassCastException
    }

    import RichRow._
    new VariantDataset(sampleIds, df.rdd.map(_.toVariantGenotypeStreamTuple))
  }
}