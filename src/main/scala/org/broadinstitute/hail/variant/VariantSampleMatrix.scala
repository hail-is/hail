package org.broadinstitute.hail.variant

import java.nio.ByteBuffer
import org.apache.spark.{SparkEnv, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.check.Gen
import scala.language.implicitConversions
import org.broadinstitute.hail.annotations._
import scala.reflect.ClassTag

object VariantSampleMatrix {
  def apply[T](metadata: VariantMetadata,
    rdd: RDD[(Variant, Annotations, Iterable[T])])(implicit tct: ClassTag[T]): VariantSampleMatrix[T] = {
    new VariantSampleMatrix(metadata, rdd)
  }

  def read(sqlContext: SQLContext, dirname: String): VariantDataset = {
    require(dirname.endsWith(".vds"))
    import RichRow._

    val (localSamples, metadata) = readDataFile(dirname + "/metadata.ser",
      sqlContext.sparkContext.hadoopConfiguration) {
      dis => {
        val serializer = SparkEnv.get.serializer.newInstance()
        serializer.deserializeStream(dis).readObject[(Array[Int], VariantMetadata)]
      }
    }

    val df = sqlContext.read.parquet(dirname + "/rdd.parquet")
    // val df = sqlContext.parquetFile(dirname + "/rdd.parquet")

    new VariantSampleMatrix[Genotype](metadata,
      localSamples,
      df.rdd.mapPartitions(iter => {
        val ser = SparkEnv.get.serializer.newInstance()
        iter.map(r =>
          (r.getVariant(0), ser.deserialize[Annotations](ByteBuffer.wrap(r.getByteArray(1))), r.getGenotypeStream(2))
        )
      }))
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
    g: (Variant) => Gen[T])(implicit tct: ClassTag[T]): Gen[VariantSampleMatrix[T]] = {
    val nSamples = sampleIds.length
    for (rows <- Gen.sequence[Seq[(Variant, Annotations, Iterable[T])], (Variant, Annotations, Iterable[T])](
      variants.map(v => Gen.zip(
        Gen.const(v),
        Gen.const(Annotations.empty()),
        genValues(nSamples, g(v))))))
      yield VariantSampleMatrix[T](VariantMetadata(sampleIds), sc.parallelize(rows))
  }

  def gen[T](sc: SparkContext, g: (Variant) => Gen[T])(implicit tct: ClassTag[T]): Gen[VariantSampleMatrix[T]] = {
    val samplesVariantsGen =
      for (sampleIds <- Gen.distinctBuildableOf[Array[String], String](Gen.identifier);
        variants <- Gen.distinctBuildableOf[Array[Variant], Variant](Variant.gen))
        yield (sampleIds, variants)
    samplesVariantsGen.flatMap { case (sampleIds, variants) => gen(sc, sampleIds, variants, g) }
  }

  def gen[T](sc: SparkContext, sampleIds: Array[String], g: (Variant) => Gen[T])(implicit tct: ClassTag[T]): Gen[VariantSampleMatrix[T]] = {
    val variantsGen = Gen.distinctBuildableOf[Array[Variant], Variant](Variant.gen)
    variantsGen.flatMap(variants => gen(sc, sampleIds, variants, g))
  }

  def gen[T](sc: SparkContext, variants: Array[Variant], g: (Variant) => Gen[T])(implicit tct: ClassTag[T]): Gen[VariantSampleMatrix[T]] = {
    val samplesGen = Gen.distinctBuildableOf[Array[String], String](Gen.identifier)
    samplesGen.flatMap(sampleIds => gen(sc, sampleIds, variants, g))
  }
}

class VariantSampleMatrix[T](val metadata: VariantMetadata,
  val localSamples: Array[Int],
  val rdd: RDD[(Variant, Annotations, Iterable[T])])
  (implicit tct: ClassTag[T]) {

  def this(metadata: VariantMetadata, rdd: RDD[(Variant, Annotations, Iterable[T])])
    (implicit tct: ClassTag[T]) =
    this(metadata, Array.range(0, metadata.nSamples), rdd)

  def sampleIds: IndexedSeq[String] = metadata.sampleIds

  def nSamples: Int = metadata.sampleIds.length

  def nLocalSamples: Int = localSamples.length

  def copy[U](metadata: VariantMetadata = metadata,
    localSamples: Array[Int] = localSamples,
    rdd: RDD[(Variant, Annotations, Iterable[U])] = rdd)
    (implicit tct: ClassTag[U]): VariantSampleMatrix[U] =
    new VariantSampleMatrix[U](metadata, localSamples, rdd)

  def sparkContext: SparkContext = rdd.sparkContext

  def cache(): VariantSampleMatrix[T] = copy[T](rdd = rdd.cache())

  def repartition(nPartitions: Int) = copy[T](rdd = rdd.repartition(nPartitions)(null))

  def nPartitions: Int = rdd.partitions.length

  def variants: RDD[Variant] = rdd.map(_._1)

  def variantsAndAnnotations: RDD[(Variant, Annotations)] = rdd.map { case (v, va, gs) => (v, va) }

  def nVariants: Long = variants.count()

  def expand(): RDD[(Variant, Int, T)] =
    mapWithKeys[(Variant, Int, T)]((v, s, g) => (v, s, g))

  def expandWithAnnotation(): RDD[(Variant, Annotations, Int, T)] =
    mapWithAll[(Variant, Annotations, Int, T)]((v, va, s, g) => (v, va, s, g))

  def sampleVariants(fraction: Double): VariantSampleMatrix[T] =
    copy(rdd = rdd.sample(withReplacement = false, fraction, 1))

  def mapValues[U](f: (T) => U)(implicit uct: ClassTag[U]): VariantSampleMatrix[U] = {
    mapValuesWithAll((v, va, s, g) => f(g))
  }

  def mapValuesWithKeys[U](f: (Variant, Int, T) => U)
    (implicit uct: ClassTag[U]): VariantSampleMatrix[U] = {
    mapValuesWithAll((v, va, s, g) => f(v, s, g))
  }

  def mapValuesWithAll[U](f: (Variant, Annotations, Int, T) => U)
    (implicit uct: ClassTag[U]): VariantSampleMatrix[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    copy(rdd = rdd.map { case (v, va, gs) =>
      (v, va,
        localSamplesBc.value.toIterable.lazyMapWith(gs,
          (s: Int, g: T) => f(v, va, s, g)))
    })
  }

  def mapValuesWithPartialApplication[U](f: (Variant, Annotations) => ((Int, T) => U))
    (implicit uct: ClassTag[U]): VariantSampleMatrix[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    copy(rdd =
      rdd.mapPartitions[(Variant, Annotations, Iterable[U])] { it: Iterator[(Variant, Annotations, Iterable[T])] =>
        it.map { case (v, va, gs) =>
          val f2 = f(v, va)
          (v, va, localSamplesBc.value.toIterable.lazyMapWith(gs, f2))
        }
      })
  }

  def map[U](f: T => U)(implicit uct: ClassTag[U]): RDD[U] =
    mapWithKeys((v, s, g) => f(g))

  def mapWithKeys[U](f: (Variant, Int, T) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    rdd
      .flatMap { case (v, va, gs) =>
        localSamplesBc.value.toIterable.lazyMapWith(gs,
          (s: Int, g: T) => f(v, s, g))
      }
  }

  def mapWithAll[U](f: (Variant, Annotations, Int, T) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    rdd
      .flatMap { case (v, va, gs) =>
        localSamplesBc.value.toIterable.lazyMapWith(gs,
          (s: Int, g: T) => f(v, va, s, g))
      }
  }

  def mapPartitionsWithAll[U](f: Iterator[(Variant, Annotations, Int, T)] => Iterator[U])(implicit uct: ClassTag[U]): RDD[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    rdd.mapPartitions { it =>
      f(it.flatMap { case (v, va, gs) =>
        localSamplesBc.value.iterator.zip(gs.iterator)
          .map { case (s, g) => (v, va, s, g) }
      })
    }
  }

  def mapAnnotations(f: (Variant, Annotations) => Annotations): VariantSampleMatrix[T] =
    copy[T](rdd = rdd.map { case (v, va, gs) => (v, f(v, va), gs) })

  def flatMap[U](f: T => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] =
    flatMapWithKeys((v, s, g) => f(g))

  def flatMapWithKeys[U](f: (Variant, Int, T) => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    rdd
      .flatMap { case (v, va, gs) => localSamplesBc.value.toIterable.lazyFlatMapWith(gs,
        (s: Int, g: T) => f(v, s, g))
      }
  }

  def filterVariants(p: (Variant, Annotations) => Boolean): VariantSampleMatrix[T] =
    copy(rdd = rdd.filter { case (v, va, gs) => p(v, va) })

  def filterVariants(ilist: IntervalList): VariantSampleMatrix[T] =
    filterVariants((v, va) => ilist.contains(v.contig, v.start))

  // FIXME see if we can remove broadcasts elsewhere in the code
  def filterSamples(p: (Int, Annotations) => Boolean): VariantSampleMatrix[T] = {
    val mask = localSamples.map((s) => p(s, metadata.sampleAnnotations(s)))
    val maskBc = sparkContext.broadcast(mask)
    val localtct = tct
    copy[T](localSamples = localSamples.zipWithIndex
      .filter { case (s, i) => mask(i) }
      .map(_._1),
      rdd = rdd.map { case (v, va, gs) =>
        (v, va, gs.lazyFilterWith(maskBc.value.toIterable, (g: T, m: Boolean) => m))
      })
  }

  def aggregateBySample[U](zeroValue: U)(
    seqOp: (U, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Int, U)] =
    aggregateBySampleWithKeys(zeroValue)((e, v, s, g) => seqOp(e, g), combOp)

  def aggregateBySampleWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Int, U)] = {
    aggregateBySampleWithAll(zeroValue)((e, v, va, s, g) => seqOp(e, v, s, g), combOp)
  }

  def aggregateBySampleWithAll[U](zeroValue: U)(
    seqOp: (U, Variant, Annotations, Int, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Int, U)] = {

    val localSamplesBc = sparkContext.broadcast(localSamples)

    val serializer = SparkEnv.get.serializer.newInstance()
    val zeroBuffer = serializer.serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    rdd
      .mapPartitions { (it: Iterator[(Variant, Annotations, Iterable[T])]) =>
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
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Variant, U)] =
    aggregateByVariantWithAll(zeroValue)((e, v, va, s, g) => seqOp(e, g), combOp)

  def aggregateByVariantWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Variant, U)] = {
    aggregateByVariantWithAll(zeroValue)((e, v, va, s, g) => seqOp(e, v, s, g), combOp)
  }

  def aggregateByVariantWithAll[U](zeroValue: U)(
    seqOp: (U, Variant, Annotations, Int, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Variant, U)] = {

    val localSamplesBc = sparkContext.broadcast(localSamples)

    // Serialize the zero value to a byte array so that we can apply a new clone of it on each key
    val zeroBuffer = SparkEnv.get.serializer.newInstance().serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    rdd
      .mapPartitions { (it: Iterator[(Variant, Annotations, Iterable[T])]) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        it.map { case (v, va, gs) =>
          val zeroValue = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))
          (v, gs.iterator.zipWithIndex.foldLeft(zeroValue) { case (acc, (g, i)) =>
            seqOp(acc, v, va, localSamplesBc.value(i), g)
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
      .mapPartitions { (it: Iterator[(Variant, Annotations, Iterable[T])]) =>
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
    metadata == that.metadata &&
      localSamples.sameElements(that.localSamples) &&
      rdd.map { case (v, va, gs) => (v, (va, gs)) }
        .fullOuterJoin(that.rdd.map { case (v, va, gs) => (v, (va, gs)) })
        .map { case (v, t) => t match {
          case (Some((va1, it1)), Some((va2, it2))) =>
            it1.sameElements(it2) && va1 == va2
          case _ => false
        }
        }.fold(true)(_ && _)
  }

  def mapAnnotationsWithAggregate[U](zeroValue: U)(
    seqOp: (U, Variant, Int, T) => U,
    combOp: (U, U) => U,
    mapOp: (Annotations, U) => Annotations)
    (implicit uct: ClassTag[U]): VariantSampleMatrix[T] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    // Serialize the zero value to a byte array so that we can apply a new clone of it on each key
    val zeroBuffer = SparkEnv.get.serializer.newInstance().serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    copy(rdd = rdd
      .map { case (v, va, gs) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        val zeroValue = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))

        (v, mapOp(va, gs.zipWithIndex.foldLeft(zeroValue) { case (acc, (g, i)) =>
          seqOp(acc, v, localSamplesBc.value(i), g)
        }), gs)
      })
  }

  def addVariantAnnotationSignatures(name: String, sig: Any): VariantSampleMatrix[T] = {
    require(sig.isInstanceOf[AnnotationSignature] ||
      sig.isInstanceOf[Annotations] && Annotations.validSignatures(sig.asInstanceOf[Annotations]))
    copy(metadata = metadata.copy(variantAnnotationSignatures =
      metadata.variantAnnotationSignatures +(name, sig)))
  }

  def addSampleAnnotationSignatures(name: String, sig: Any): VariantSampleMatrix[T] = {
    require(sig.isInstanceOf[AnnotationSignature] ||
      sig.isInstanceOf[Annotations] && Annotations.validSignatures(sig.asInstanceOf[Annotations]))
    copy(metadata = metadata.copy(sampleAnnotationSignatures =
      metadata.sampleAnnotationSignatures +(name, sig)))
  }
}

// FIXME AnyVal Scala 2.11
class RichVDS(vds: VariantDataset) {

  def write(sqlContext: SQLContext, dirname: String, compress: Boolean = true) {
    import sqlContext.implicits._

    require(dirname.endsWith(".vds"))

    val hConf = vds.sparkContext.hadoopConfiguration
    hadoopMkdir(dirname, hConf)
    writeDataFile(dirname + "/metadata.ser", hConf) {
      dos => {
        val serializer = SparkEnv.get.serializer.newInstance()
        serializer.serializeStream(dos).writeObject((vds.localSamples, vds.metadata))
      }
    }

    vds.rdd
      .mapPartitions { iter =>
        val serializer = SparkEnv.get.serializer.newInstance()
        iter.map {
          case (v, va, gs) =>
            (v, serializer.serialize(va).array(), gs.toGenotypeStream(v, compress))
        }
      }
      .toDF()
      .write.parquet(dirname + "/rdd.parquet")
    // .saveAsParquetFile(dirname + "/rdd.parquet")
  }

  def eraseSplit: VariantDataset = {
    vds.copy(rdd =
      vds.rdd.map { case (v, va, gs) =>
        (v, va.copy(attrs = va.attrs - "multiallelic"),
          gs.map(g => g.copy(fakeRef = false))
          )
      })
  }
}
