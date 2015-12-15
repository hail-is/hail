package org.broadinstitute.hail.variant

import java.nio.ByteBuffer

import org.apache.spark.{SparkEnv, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.Utils._
import org.scalacheck.{Arbitrary, Gen}
import org.scalacheck.Arbitrary._
import scala.language.implicitConversions

import scala.reflect.ClassTag
import scala.reflect.runtime.universe._


object VariantSampleMatrix {
  def apply[T](metadata: VariantMetadata,
    rdd: RDD[(Variant, Iterable[T])])(implicit ttt: TypeTag[T], tct: ClassTag[T]): VariantSampleMatrix[T] = {
    new VariantSampleMatrix(metadata, rdd)
  }

  def read(sqlContext: SQLContext, dirname: String): VariantDataset = {
    require(dirname.endsWith(".vds"))
    import RichRow._

    val metadata = readObjectFile(dirname + "/metadata.ser", sqlContext.sparkContext.hadoopConfiguration)(
      _.readObject().asInstanceOf[VariantMetadata])

    // val df = sqlContext.read.parquet(dirname + "/rdd.parquet")
    val df = sqlContext.parquetFile(dirname + "/rdd.parquet")
    new VariantSampleMatrix[Genotype](metadata, df.rdd.map(r => (r.getVariant(0), r.getGenotypeStream(1))))
  }

  def genValues[T](nSamples: Int, g: Gen[T]): Gen[Iterable[T]] =
    Gen.buildableOfN[Iterable[T], T](nSamples, g)

  def genValues[T](g: Gen[T]): Gen[Iterable[T]] =
    Gen.buildableOf[Iterable[T], T](g)

  def genVariantValues[T](nSamples: Int, g: (Variant) => Gen[T]): Gen[(Variant, Iterable[T])] =
    for (v <- Variant.gen;
      values <- genValues[T](nSamples, g(v)))
      yield (v, values)

  def genVariantValues[T](g: (Variant) => Gen[T]): Gen[(Variant, Iterable[T])] =
    for (v <- Variant.gen;
      values <- genValues[T](g(v)))
      yield (v, values)

  def genVariantGenotypes: Gen[(Variant, Iterable[Genotype])] =
    genVariantValues(Genotype.gen)

  def genVariantGenotypes(nSamples: Int): Gen[(Variant, Iterable[Genotype])] =
    genVariantValues(nSamples, Genotype.gen)

  def gen[T](sc: SparkContext,
    sampleIds: Array[String],
    variants: Array[Variant],
    g: (Variant) => Gen[T])(implicit ttt: TypeTag[T], tct: ClassTag[T]): Gen[VariantSampleMatrix[T]] = {

    for (rows <- Gen.sequence[Seq[(Variant, Iterable[T])], (Variant, Iterable[T])](variants.map(v => Gen.zip(Gen.const(v), genValues(g(v))))))
      yield VariantSampleMatrix[T](VariantMetadata(Map.empty, sampleIds), sc.parallelize(rows))
  }

  def gen[T](sc: SparkContext, g: (Variant) => Gen[T])(implicit ttt: TypeTag[T], tct: ClassTag[T]): Gen[VariantSampleMatrix[T]] = {
    val samplesVariantsGen =
      for (sampleIds <- genDistinctBuildableOf[Array[String], String](Gen.identifier);
        variants <- genDistinctBuildableOf[Array[Variant], Variant](Variant.gen))
        yield (sampleIds, variants)
    samplesVariantsGen.flatMap { case (sampleIds, variants) => gen(sc, sampleIds, variants, g) }
  }

  def gen[T](sc: SparkContext, sampleIds: Array[String], g: (Variant) => Gen[T])(implicit ttt: TypeTag[T], tct: ClassTag[T]): Gen[VariantSampleMatrix[T]] = {
    val variantsGen = genDistinctBuildableOf[Array[Variant], Variant](Variant.gen)
    variantsGen.flatMap(variants => gen(sc, sampleIds, variants, g))
  }

  def gen[T](sc: SparkContext, variants: Array[Variant], g: (Variant) => Gen[T])(implicit ttt: TypeTag[T], tct: ClassTag[T]): Gen[VariantSampleMatrix[T]] = {
    val samplesGen = genDistinctBuildableOf[Array[String], String](Gen.identifier)
    samplesGen.flatMap(sampleIds => gen(sc, sampleIds, variants, g))
  }
}

class VariantSampleMatrix[T](val metadata: VariantMetadata,
  val localSamples: Array[Int],
  val rdd: RDD[(Variant, Iterable[T])])
  (implicit ttt: TypeTag[T], tct: ClassTag[T],
    vct: ClassTag[Variant]) {

  def this(metadata: VariantMetadata, rdd: RDD[(Variant, Iterable[T])])
    (implicit ttt: TypeTag[T], tct: ClassTag[T]) =
    this(metadata, Array.range(0, metadata.nSamples), rdd)

  def sampleIds: IndexedSeq[String] = metadata.sampleIds

  def nSamples: Int = metadata.sampleIds.length

  def nLocalSamples: Int = localSamples.length

  def copy[U](metadata: VariantMetadata = this.metadata,
    localSamples: Array[Int] = this.localSamples,
    rdd: RDD[(Variant, Iterable[U])] = this.rdd)
    (implicit ttt: TypeTag[U], tct: ClassTag[U]): VariantSampleMatrix[U] =
    new VariantSampleMatrix(metadata, localSamples, rdd)

  def sparkContext: SparkContext = rdd.sparkContext

  def cache(): VariantSampleMatrix[T] = copy[T](rdd = rdd.cache())

  def repartition(nPartitions: Int) = copy[T](rdd = rdd.repartition(nPartitions))

  def nPartitions: Int = rdd.partitions.length

  def variants: RDD[Variant] = rdd.keys

  def nVariants: Long = variants.count()

  def expand(): RDD[(Variant, Int, T)] =
    mapWithKeys[(Variant, Int, T)]((v, s, g) => (v, s, g))

  def sampleVariants(fraction: Double): VariantSampleMatrix[T] =
    copy(rdd = rdd.sample(withReplacement = false, fraction, 1))

  def mapValues[U](f: (T) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): VariantSampleMatrix[U] = {
    mapValuesWithKeys((v, s, g) => f(g))
  }

  def mapValuesWithKeys[U](f: (Variant, Int, T) => U)
    (implicit utt: TypeTag[U], uct: ClassTag[U]): VariantSampleMatrix[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    copy(rdd = rdd.map { case (v, gs) =>
      (v, localSamplesBc.value.view.zip(gs.view)
        .map { case (s, t) => f(v, s, t) })
    })
  }

  def map[U](f: T => U)(implicit uct: ClassTag[U]): RDD[U] =
    mapWithKeys((v, s, g) => f(g))

  def mapWithKeys[U](f: (Variant, Int, T) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    rdd
      .flatMap { case (v, gs) => localSamplesBc.value.view.zip(gs.view)
        .map { case (s, g) => f(v, s, g) }
      }
  }

  def flatMap[U](f: T => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] =
    flatMapWithKeys((v, s, g) => f(g))

  def flatMapWithKeys[U](f: (Variant, Int, T) => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    rdd
      .flatMap { case (v, gs) => localSamplesBc.value.view.zip(gs.view)
        .flatMap { case (s, g) => f(v, s, g) }
      }
  }

  def filterVariants(ilist: IntervalList): VariantSampleMatrix[T] =
    filterVariants(v => ilist.contains(v.contig, v.start))

  def filterVariants(p: (Variant) => Boolean): VariantSampleMatrix[T] =
    copy(rdd = rdd.filter { case (v, _) => p(v) })

  def filterSamples(p: (Int) => Boolean) = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    copy[T](localSamples = localSamples.filter(p),
      rdd = rdd.map { case (v, gs) =>
        (v, localSamplesBc.value.view.zip(gs.view)
          .filter { case (s, _) => p(s) }
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

    val localSamplesBc = sparkContext.broadcast(localSamples)

    val serializer = SparkEnv.get.serializer.newInstance()
    val zeroBuffer = serializer.serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    rdd
      .mapPartitions { (it: Iterator[(Variant, Iterable[T])]) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        def copyZeroValue() = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))
        val arrayZeroValue = Array.fill[U](localSamplesBc.value.length)(copyZeroValue())

        localSamplesBc.value.iterator
          .zip(it.foldLeft(arrayZeroValue) { case (acc, (v, gs)) =>
            for ((g, i) <- gs.zipWithIndex)
              acc(i) = seqOp(acc(i), v, localSamplesBc.value(i), g)
            acc
          }.iterator)
      }.foldByKey(zeroValue)(combOp)
  }

  def aggregateByVariant[U](zeroValue: U)(
    seqOp: (U, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): RDD[(Variant, U)] =
    aggregateByVariantWithKeys(zeroValue)((e, v, s, g) => seqOp(e, g), combOp)

  def aggregateByVariantWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit utt: TypeTag[U], uct: ClassTag[U]): RDD[(Variant, U)] = {

    val localSamplesBc = sparkContext.broadcast(localSamples)

    // Serialize the zero value to a byte array so that we can get a new clone of it on each key
    val zeroBuffer = SparkEnv.get.serializer.newInstance().serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    rdd.mapPartitions { (it: Iterator[(Variant, Iterable[T])]) =>
      val serializer = SparkEnv.get.serializer.newInstance()

      it.map { case (v, gs) =>
        // FIXME shadow
        val zeroValue = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))
        (v, gs.iterator.zipWithIndex.foldLeft(zeroValue) { case (acc, (g, i)) =>
          seqOp(acc, v, localSamplesBc.value(i), g)
        })
      }
    }

    /*
        rdd
          .map { case (v, gs) =>
            val serializer = SparkEnv.get.serializer.newInstance()
            val zeroValue = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))

            (v, gs.zipWithIndex.foldLeft(zeroValue) { case (acc, (g, i)) =>
              seqOp(acc, v, localSamplesBc.value(i), g)
            })
          }
    */
  }

  def foldBySample(zeroValue: T)(combOp: (T, T) => T): RDD[(Int, T)] = {

    val localSamplesBc = sparkContext.broadcast(localSamples)
    val localtct = tct

    val serializer = SparkEnv.get.serializer.newInstance()
    val zeroBuffer = serializer.serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    rdd
      .mapPartitions { (it: Iterator[(Variant, Iterable[T])]) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        def copyZeroValue() = serializer.deserialize[T](ByteBuffer.wrap(zeroArray))(localtct)
        val arrayZeroValue = Array.fill[T](localSamplesBc.value.length)(copyZeroValue())
        localSamplesBc.value.iterator
          .zip(it.foldLeft(arrayZeroValue) { case (acc, (v, gs)) =>
            for ((g, i) <- gs.zipWithIndex)
              acc(i) = combOp(acc(i), g)
            acc
          }.iterator)
      }.foldByKey(zeroValue)(combOp)
  }

  def foldByVariant(zeroValue: T)(combOp: (T, T) => T): RDD[(Variant, T)] =
    rdd.mapValues(_.foldLeft(zeroValue)((acc, g) => combOp(acc, g)))

  def same(that: VariantSampleMatrix[T]): Boolean = {
    metadata == that.metadata &&
      localSamples.sameElements(that.localSamples) &&
      rdd.fullOuterJoin(that.rdd)
        .map { case (v, t) => t match {
          case (Some(it1), Some(it2)) =>
            it1.sameElements(it2)
          case _ => false
        }
        }.reduce(_ && _)
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
      .map { case (v, gs) => (v, gs.toGenotypeStream(v, compress)) }
      .toDF()
      .saveAsParquetFile(dirname + "/rdd.parquet")
  }
}
