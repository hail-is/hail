package org.broadinstitute.hail.variant

import java.nio.ByteBuffer

import org.apache.spark.{SparkEnv, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.Utils._
import scala.language.implicitConversions
import org.broadinstitute.hail.annotations._

import scala.reflect.ClassTag
import scala.reflect.runtime.universe._


object VariantSampleMatrix {
  def apply(metadata: VariantMetadata,
    rdd: RDD[(Variant, AnnotationData, Iterable[Genotype])]): VariantDataset = {
    new VariantSampleMatrix(metadata, rdd)

  }

  def read(sqlContext: SQLContext, dirname: String): VariantDataset = {
    require(dirname.endsWith(".vds"))
    import RichRow._

    val metadata = readObjectFile(dirname + "/metadata.ser", sqlContext.sparkContext.hadoopConfiguration)(
      _.readObject().asInstanceOf[VariantMetadata])

    // val df = sqlContext.read.parquet(dirname + "/rdd.parquet")
    val df = sqlContext.parquetFile(dirname + "/rdd.parquet")
    new VariantSampleMatrix[Genotype](metadata, df.rdd.map(r =>
      (r.getVariant(0), r.getVariantAnnotations(1), r.getGenotypeStream(2))))
  }
}

class VariantSampleMatrix[T](val metadata: VariantMetadata,
  val localSamples: Array[Int],
  val rdd: RDD[(Variant, AnnotationData, Iterable[T])])
  (implicit ttt: TypeTag[T], tct: ClassTag[T],
    vct: ClassTag[Variant]) {

  def this(metadata: VariantMetadata, rdd: RDD[(Variant, AnnotationData, Iterable[T])])
    (implicit ttt: TypeTag[T], tct: ClassTag[T]) =
    this(metadata, Array.range(0, metadata.nSamples), rdd)

  def sampleIds: IndexedSeq[String] = metadata.sampleIds

  def nSamples: Int = metadata.sampleIds.length

  def nLocalSamples: Int = localSamples.length

  def copy[U](metadata: VariantMetadata = this.metadata,
    localSamples: Array[Int] = this.localSamples,
    rdd: RDD[(Variant, AnnotationData, Iterable[U])] = this.rdd)
    (implicit ttt: TypeTag[U], tct: ClassTag[U]): VariantSampleMatrix[U] =
    new VariantSampleMatrix(metadata, localSamples, rdd)

  def sparkContext: SparkContext = rdd.sparkContext

  def cache(): VariantSampleMatrix[T] = copy[T](rdd = rdd.cache())

  def repartition(nPartitions: Int) = copy[T](rdd = rdd.repartition(nPartitions))

  def nPartitions: Int = rdd.partitions.length

  def variants: RDD[Variant] = rdd.map(_._1)

  def variantsAndAnnotations: RDD[(Variant, AnnotationData)] = rdd.map { case (v, va, gs) => (v, va) }

  def nVariants: Long = variants.count()

  def expand(): RDD[(Variant, Int, T)] =
    mapWithKeys[(Variant, Int, T)]((v, s, g) => (v, s, g))

  def expandWithAnnotation(): RDD[(Variant, AnnotationData, Int, T)] =
    mapWithAll[(Variant, AnnotationData, Int, T)]((v, va, s, g) => (v, va, s, g))

  def sampleVariants(fraction: Double): VariantSampleMatrix[T] =
    copy(rdd = rdd.sample(withReplacement = false, fraction, 1))

  def mapValues[U](f: (T) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): VariantSampleMatrix[U] = {
    mapValuesWithAll((v, va, s, g) => f(g))
  }

  def mapValuesWithKeys[U](f: (Variant, Int, T) => U)
    (implicit utt: TypeTag[U], uct: ClassTag[U]): VariantSampleMatrix[U] = {
    mapValuesWithAll((v, va, s, g) => f(v, s, g))
  }

  def mapValuesWithAll[U](f: (Variant, AnnotationData, Int, T) => U)
    (implicit utt: TypeTag[U], uct: ClassTag[U]): VariantSampleMatrix[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    copy(rdd = rdd.map { case (v, va, gs) =>
      (v, va, localSamplesBc.value.view.zip(gs.view)
        .map { case (s, t) => f(v, va, s, t) })

    })
  }

  def map[U](f: T => U)(implicit uct: ClassTag[U]): RDD[U] =
    mapWithKeys((v, s, g) => f(g))

  def mapWithKeys[U](f: (Variant, Int, T) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    rdd
      .flatMap { case (v, va, gs) => localSamplesBc.value.view.zip(gs.view)
        .map { case (s, g) => f(v, s, g) }
      }
  }

  def mapWithAll[U](f: (Variant, AnnotationData, Int, T) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    rdd
      .flatMap { case (v, va, gs) => localSamplesBc.value.view.zip(gs.view)
        .map { case (s, g) => f(v, va, s, g) }
      }
  }

  def flatMap[U](f: T => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] =
    flatMapWithKeys((v, s, g) => f(g))

  def flatMapWithKeys[U](f: (Variant, Int, T) => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    rdd
      .flatMap { case (v, va, gs) => localSamplesBc.value.view.zip(gs.view)
        .flatMap { case (s, g) => f(v, s, g) }
      }
  }

  def filterVariants(p: (Variant, Annotations[String]) => Boolean): VariantSampleMatrix[T] =
    copy(rdd = rdd.filter { case (v, va, gs) => p(v, va) })

  def filterVariants(ilist: IntervalList): VariantSampleMatrix[T] =
    filterVariants((v, va) => ilist.contains(v.contig, v.start))

  def filterSamples(p: (Int, AnnotationData) => Boolean): VariantSampleMatrix[T] = {
    val mask = localSamples.zip(metadata.sampleAnnotations).map { case (s, sa) => p(s, sa) }
    val maskBc = sparkContext.broadcast(mask)
    copy[T](localSamples = localSamples.zipWithIndex
      .filter { case (s, i) => mask(i) }
      .map(_._1),
      rdd = rdd.map { case (v, va, gs) =>
        (v, va, maskBc.value.view.zip(gs.view)
          .filter(_._1)
          .map(_._2))
      })
  }

  def aggregateBySample[U](zeroValue: U)(
    seqOp: (U, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): RDD[(Int, U)] =
    aggregateBySampleWithKeys(zeroValue)((e, v, s, g) => seqOp(e, g), combOp)

  def aggregateBySampleWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): RDD[(Int, U)] = {
    aggregateBySampleWithAll(zeroValue)((e, v, va, s, g) => seqOp(e, v, s, g), combOp)
  }

  def aggregateBySampleWithAll[U](zeroValue: U)(
    seqOp: (U, Variant, AnnotationData, Int, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): RDD[(Int, U)] = {

    val localSamplesBc = sparkContext.broadcast(localSamples)

    val serializer = SparkEnv.get.serializer.newInstance()
    val zeroBuffer = serializer.serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    rdd
      .mapPartitions { (it: Iterator[(Variant, AnnotationData, Iterable[T])]) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        def copyZeroValue() = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))
        val arrayZeroValue = Array.fill[U](localSamplesBc.value.length)(copyZeroValue())

        localSamplesBc.value.iterator
          .zip(it.foldLeft(arrayZeroValue) { case (acc, (v, va, gs)) =>
            for ((g, i) <- gs.zipWithIndex)
              acc(i) = seqOp(acc(i), v, va, localSamplesBc.value(i), g)
            acc
          }.iterator)
      }.foldByKey(zeroValue)(combOp)
  }

  def aggregateByVariant[U](zeroValue: U)(
    seqOp: (U, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): RDD[(Variant, U)] =
    aggregateByVariantWithAll(zeroValue)((e, v, va, s, g) => seqOp(e, g), combOp)

  def aggregateByVariantWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): RDD[(Variant, U)] = {
    aggregateByVariantWithAll(zeroValue)((e, v, va, s, g) => seqOp(e, v, s, g), combOp)
  }

  def aggregateByVariantWithAll[U](zeroValue: U)(
    seqOp: (U, Variant, AnnotationData, Int, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): RDD[(Variant, U)] = {

    val localSamplesBc = sparkContext.broadcast(localSamples)

    // Serialize the zero value to a byte array so that we can get a new clone of it on each key
    val zeroBuffer = SparkEnv.get.serializer.newInstance().serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    rdd
      .map { case (v, va, gs) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        val zeroValue = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))

        (v, gs.zipWithIndex.foldLeft(zeroValue) { case (acc, (g, i)) =>
          seqOp(acc, v, va, localSamplesBc.value(i), g)
        })
      }
  }

  def foldBySample(zeroValue: T)(combOp: (T, T) => T): RDD[(Int, T)] = {

    val localSamplesBc = sparkContext.broadcast(localSamples)
    val localtct = tct

    val serializer = SparkEnv.get.serializer.newInstance()
    val zeroBuffer = serializer.serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    rdd
      .mapPartitions { (it: Iterator[(Variant, AnnotationData, Iterable[T])]) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        def copyZeroValue() = serializer.deserialize[T](ByteBuffer.wrap(zeroArray))(localtct)
        val arrayZeroValue = Array.fill[T](localSamplesBc.value.length)(copyZeroValue())
        localSamplesBc.value.iterator
          .zip(it.foldLeft(arrayZeroValue) { case (acc, (v, va, gs)) =>
            for ((g, i) <- gs.zipWithIndex)
              acc(i) = combOp(acc(i), g)
            acc
          }.iterator)
      }.foldByKey(zeroValue)(combOp)
  }

  def foldByVariant(zeroValue: T)(combOp: (T, T) => T): RDD[(Variant, T)] =
    rdd.map { case (v, va, gs) => (v, gs.foldLeft(zeroValue)((acc, g) => combOp(acc, g))) }

  def same(that: VariantSampleMatrix[T]): Boolean = {
    println(metadata == that.metadata)
    metadata == that.metadata &&
      localSamples.sameElements(that.localSamples) &&
      rdd.map { case (v, va, gs) => (v, (va, gs)) }
        .fullOuterJoin(that.rdd.map { case (v, va, gs) => (v, (va, gs)) })
        .map { case (v, t) => t match {
          case (Some((va1, it1)), Some((va2, it2))) =>
            it1.sameElements(it2) && va1 == va2
          case _ => false
        }
        }.reduce(_ && _)
  }
    def mapAnnotationsWithAggregate[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U,
    mapOp: (AnnotationData, U) => AnnotationData)
    (implicit utt: TypeTag[U], uct: ClassTag[U]): VariantSampleMatrix[T] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    // Serialize the zero value to a byte array so that we can get a new clone of it on each key
    val zeroBuffer = SparkEnv.get.serializer.newInstance().serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    this.copy(rdd = rdd
      .map { case (v, va, gs) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        val zeroValue = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))

        (v, mapOp(va, gs.zipWithIndex.foldLeft(zeroValue) { case (acc, (g, i)) =>
          seqOp(acc, v, localSamplesBc.value(i), g)
        }), gs)
      })
  }

  def addVariantSignatures(maps: Map[String, Map[String, AnnotationSignature]] = Map.empty[String, Map[String, AnnotationSignature]],
    vals: Map[String, AnnotationSignature] = Map.empty[String, AnnotationSignature]): VariantSampleMatrix[T] = {
    this.copy(metadata = this.metadata.copy(variantAnnotationSignatures =
      this.metadata.variantAnnotationSignatures.addMaps(maps).addVals(vals)))
  }

  def addSampleSignatures(maps: Map[String, Map[String, AnnotationSignature]] = Map.empty[String, Map[String, AnnotationSignature]],
    vals: Map[String, AnnotationSignature] = Map.empty[String, AnnotationSignature]): VariantSampleMatrix[T] = {
    this.copy(metadata = this.metadata.copy(sampleAnnotationSignatures =
      this.metadata.sampleAnnotationSignatures.addMaps(maps).addVals(vals)))
  }
}

// FIXME AnyVal Scala 2.11
class RichVDS(vds: VariantDataset) {

  def write(sqlContext: SQLContext, dirname: String, compress: Boolean = true) {
    import sqlContext.implicits._

    require(dirname.endsWith(".vds"))

    val hConf = vds.sparkContext.hadoopConfiguration
    hadoopMkdir(dirname, hConf)
    writeObjectFile(dirname + "/metadata.ser", hConf)(
      _.writeObject(vds.metadata))

    // rdd.toDF().write.parquet(dirname + "/rdd.parquet")
    vds.rdd
      .map { case (v, va, gs) => (v, va, gs.toGenotypeStream(v, compress)) }
      .toDF()
      .saveAsParquetFile(dirname + "/rdd.parquet")
  }
}
