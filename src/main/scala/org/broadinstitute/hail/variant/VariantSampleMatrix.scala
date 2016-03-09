package org.broadinstitute.hail.variant

import java.nio.ByteBuffer
import org.apache.spark.sql.catalyst.ScalaReflection.Schema
import org.apache.spark.{SparkEnv, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.check.Gen
import org.broadinstitute.hail.io.annotators.{SampleAnnotator, VariantAnnotator}
import scala.language.implicitConversions
import org.broadinstitute.hail.annotations._
import scala.reflect.ClassTag
import org.apache.spark.sql.types.{NullType, StructType, StructField, StringType}


object VariantSampleMatrix {
  def apply[T](metadata: VariantMetadata,
    rdd: RDD[(Variant, Annotation, Iterable[T])])(implicit tct: ClassTag[T]): VariantSampleMatrix[T] = {
    new VariantSampleMatrix(metadata, rdd)
  }

  def read(sqlContext: SQLContext, dirname: String): VariantDataset = {
    require(dirname.endsWith(".vds"))

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
      df.rdd.map(row => {
        (Variant.fromRow(row.getAs[Row](0)), row.get(1), GenotypeStream.fromRow(row.getAs[Row](2)))
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
    for (rows <- Gen.sequence[Seq[(Variant, Annotation, Iterable[T])], (Variant, Annotation, Iterable[T])](
      variants.map(v => Gen.zip(
        Gen.const(v),
        Gen.const(Annotation.empty),
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
  val rdd: RDD[(Variant, Annotation, Iterable[T])])
  (implicit tct: ClassTag[T]) {

  def this(metadata: VariantMetadata, rdd: RDD[(Variant, Annotation, Iterable[T])])
    (implicit tct: ClassTag[T]) =
    this(metadata, Array.range(0, metadata.nSamples), rdd)

  def sampleIds: IndexedSeq[String] = metadata.sampleIds

  def nSamples: Int = metadata.sampleIds.length

  def nLocalSamples: Int = localSamples.length

  def vaSignatures: Signature = metadata.vaSignatures

  def saSignatures: Signature = metadata.saSignatures

  def sampleAnnotations: IndexedSeq[Annotation] = metadata.sampleAnnotations

  def copy[U](localSamples: Array[Int] = localSamples,
    rdd: RDD[(Variant, Annotation, Iterable[U])] = rdd,
    filters: IndexedSeq[(String, String)] = metadata.filters,
    sampleIds: IndexedSeq[String] = metadata.sampleIds,
    sampleAnnotations: IndexedSeq[Annotation] = metadata.sampleAnnotations,
    saSignatures: Signature = metadata.saSignatures,
    vaSignatures: Signature = metadata.vaSignatures,
    wasSplit: Boolean = metadata.wasSplit)
    (implicit tct: ClassTag[U]): VariantSampleMatrix[U] =
    new VariantSampleMatrix[U](
      VariantMetadata(filters, sampleIds, sampleAnnotations, saSignatures, vaSignatures, wasSplit), localSamples, rdd)

  def sparkContext: SparkContext = rdd.sparkContext

  def cache(): VariantSampleMatrix[T] = copy[T](rdd = rdd.cache())

  def repartition(nPartitions: Int) = copy[T](rdd = rdd.repartition(nPartitions)(null))

  def nPartitions: Int = rdd.partitions.length

  def variants: RDD[Variant] = rdd.map(_._1)

  def variantsAndAnnotations: RDD[(Variant, Annotation)] = rdd.map { case (v, va, gs) => (v, va) }

  def nVariants: Long = variants.count()

  def expand(): RDD[(Variant, Int, T)] =
    mapWithKeys[(Variant, Int, T)]((v, s, g) => (v, s, g))

  def expandWithAnnotation(): RDD[(Variant, Annotation, Int, T)] =
    mapWithAll[(Variant, Annotation, Int, T)]((v, va, s, g) => (v, va, s, g))

  def sampleVariants(fraction: Double): VariantSampleMatrix[T] =
    copy(rdd = rdd.sample(withReplacement = false, fraction, 1))

  def mapValues[U](f: (T) => U)(implicit uct: ClassTag[U]): VariantSampleMatrix[U] = {
    mapValuesWithAll((v, va, s, g) => f(g))
  }

  def mapValuesWithKeys[U](f: (Variant, Int, T) => U)
    (implicit uct: ClassTag[U]): VariantSampleMatrix[U] = {
    mapValuesWithAll((v, va, s, g) => f(v, s, g))
  }

  def mapValuesWithAll[U](f: (Variant, Annotation, Int, T) => U)
    (implicit uct: ClassTag[U]): VariantSampleMatrix[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    copy(rdd = rdd.map { case (v, va, gs) =>
      (v, va,
        localSamplesBc.value.toIterable.lazyMapWith(gs,
          (s: Int, g: T) => f(v, va, s, g)))
    })
  }

  def mapValuesWithPartialApplication[U](f: (Variant, Annotation) => ((Int, T) => U))
    (implicit uct: ClassTag[U]): VariantSampleMatrix[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    copy(rdd =
      rdd.mapPartitions[(Variant, Annotation, Iterable[U])] { it: Iterator[(Variant, Annotation, Iterable[T])] =>
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

  def mapWithAll[U](f: (Variant, Annotation, Int, T) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    rdd
      .flatMap { case (v, va, gs) =>
        localSamplesBc.value.toIterable.lazyMapWith(gs,
          (s: Int, g: T) => f(v, va, s, g))
      }
  }

  def mapPartitionsWithAll[U](f: Iterator[(Variant, Annotation, Int, T)] => Iterator[U])(implicit uct: ClassTag[U]): RDD[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    rdd.mapPartitions { it =>
      f(it.flatMap { case (v, va, gs) =>
        localSamplesBc.value.iterator.zip(gs.iterator)
          .map { case (s, g) => (v, va, s, g) }
      })
    }
  }

  def mapAnnotations(f: (Variant, Annotation) => Annotation): VariantSampleMatrix[T] =
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

  def filterVariants(p: (Variant, Annotation) => Boolean): VariantSampleMatrix[T] =
    copy(rdd = rdd.filter { case (v, va, gs) => p(v, va) })

  def filterVariants(ilist: IntervalList): VariantSampleMatrix[T] =
    filterVariants((v, va) => ilist.contains(v.contig, v.start))

  // FIXME see if we can remove broadcasts elsewhere in the code
  def filterSamples(p: (Int, Annotation) => Boolean): VariantSampleMatrix[T] = {
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
    seqOp: (U, Variant, Annotation, Int, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Int, U)] = {

    val localSamplesBc = sparkContext.broadcast(localSamples)

    val serializer = SparkEnv.get.serializer.newInstance()
    val zeroBuffer = serializer.serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    rdd
      .mapPartitions { (it: Iterator[(Variant, Annotation, Iterable[T])]) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        def copyZeroValue() = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))
        val arrayZeroValue = Array.fill[U](localSamplesBc.value.length)(copyZeroValue())

        localSamplesBc.value.iterator
          .zip(it.foldLeft(arrayZeroValue) { case (acc, (v, va, gs)) =>
            for ((g, i) <- gs.iterator.zipWithIndex)
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
    seqOp: (U, Variant, Annotation, Int, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Variant, U)] = {

    val localSamplesBc = sparkContext.broadcast(localSamples)

    // Serialize the zero value to a byte array so that we can apply a new clone of it on each key
    val zeroBuffer = SparkEnv.get.serializer.newInstance().serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    rdd
      .mapPartitions { (it: Iterator[(Variant, Annotation, Iterable[T])]) =>
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
      .mapPartitions { (it: Iterator[(Variant, Annotation, Iterable[T])]) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        def copyZeroValue() = serializer.deserialize[T](ByteBuffer.wrap(zeroArray))(localtct)
        val arrayZeroValue = Array.fill[T](localSamplesBc.value.length)(copyZeroValue())
        localSamplesBc.value.iterator
          .zip(it.foldLeft(arrayZeroValue) { case (acc, (v, va, gs)) =>
            for ((g, i) <- gs.iterator.zipWithIndex)
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
    mapOp: (Annotation, U) => Annotation)
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

        (v, mapOp(va, gs.iterator.zipWithIndex.foldLeft(zeroValue) { case (acc, (g, i)) =>
          seqOp(acc, v, localSamplesBc.value(i), g)
        }), gs)
      })
  }

  def annotateVariants(annotator: VariantAnnotator): VariantSampleMatrix[T] = {
    val annotatorBc = sparkContext.broadcast(annotator)
    copy(rdd = rdd.mapPartitions(
      iter => {
        lazy val sz = SparkEnv.get.serializer.newInstance()
        val annotator = annotatorBc.value
        val toRemove = annotator.metadata()
        iter.map {
          case (v, va, gs) =>
            (v, annotator.annotate(v, va, sz), gs)
        }
      })
    )
      .copy(vaSignatures = annotator.metadata())
  }

  def annotateSamples(annotator: SampleAnnotator): VariantSampleMatrix[T] = {
    copy(metadata = metadata.annotateSamples(annotator))
  }

  def queryVA(args: String*): Querier = queryVA(args.toList)

  def queryVA(path: List[String]): Querier = {
    try {
      vaSignatures.query(path)
    } catch {
      case e: AnnotationPathException => fatal(s"Invalid variant annotations query: ${path.::("va").mkString(".")}")
    }
  }

  def querySA(args: String*): Querier = querySA(args.toList)

  def querySA(path: List[String]): Querier = {
    try {
      saSignatures.query(path)
    } catch {
      case e: AnnotationPathException => fatal(s"Invalid sample annotations query: ${path.::("sa").mkString(".")}")
    }
  }

  def deleteVA(args: String*): (Signature, Deleter) = deleteVA(args.toList)

  def deleteVA(path: List[String]): (Signature, Deleter) = {
    vaSignatures.delete(path) match {
      case (null, null) => (EmptySignature(), a => null)
      case x => x
    }
  }

  def deleteSA(args: String*): (Signature, Deleter) = deleteSA(args.toList)

  def deleteSA(path: List[String]): (Signature, Deleter) = {
    saSignatures.delete(path) match {
      case (null, null) => (EmptySignature(), a => null)
      case x => x
    }
  }

  def insertVA(sig: Signature, args: String*): (Signature, Inserter) = insertVA(sig, args.toList)

  def insertVA(sig: Signature, path: List[String]): (Signature, Inserter) = {
    vaSignatures.insert(path, sig)
  }

  def insertSA(sig: Signature, args: String*): (Signature, Inserter) = insertSA(sig, args.toList)

  def insertSA(sig: Signature, path: List[String]): (Signature, Inserter) = {
    saSignatures.insert(path, sig)
  }

  def vaSchema: String = {
    metadata.vaSignatures match {
      case null => "va: empty schema"
      case s => s.printSchema("va")
    }
  }

  def saSchema: String = {
    metadata.saSignatures match {
      case null => "sa: empty schema"
      case s => s.printSchema("sa")
    }
  }
}

// FIXME AnyVal Scala 2.11
class RichVDS(vds: VariantDataset) {

  def makeSchema(): StructType = {
    val vaStruct = vds.metadata.vaSignatures.getSchema

    val s = StructType(Array(
      StructField("variant", Variant.schema(), false),
      StructField("annotations", vds.metadata.vaSignatures.getSchema, false),
      StructField("gs", GenotypeStream.schema(), false)
    ))
    s
  }

  def write(sqlContext: SQLContext, dirname: String, compress: Boolean = true) {
    require(dirname.endsWith(".vds"))

    val hConf = vds.sparkContext.hadoopConfiguration
    hadoopMkdir(dirname, hConf)
    writeDataFile(dirname + "/metadata.ser", hConf) {
      dos => {
        val serializer = SparkEnv.get.serializer.newInstance()
        serializer.serializeStream(dos).writeObject((vds.localSamples, vds.metadata))
      }
    }

    val rowRDD = vds.rdd
      .map {
        case (v, va, gs) =>
          Row.fromSeq(Array(Variant.toRow(v), va.asInstanceOf[Row], GenotypeStream.toRow(gs.toGenotypeStream(v, compress))))
      }
    sqlContext.createDataFrame(rowRDD, makeSchema())
      .write.parquet(dirname + "/rdd.parquet")
    // .saveAsParquetFile(dirname + "/rdd.parquet")
  }

  def eraseSplit: VariantDataset = {
    val (newSignatures1, f1) = vds.deleteVA("wasSplit")
    val vds1 = vds.copy(vaSignatures = newSignatures1)
    val (newSignatures2, f2) = vds1.deleteVA("aIndex")
    vds1.copy(wasSplit = false,
      vaSignatures = newSignatures2,
      rdd = vds1.rdd.map {
        case (v, va, gs) =>
          (v, f2(f1(va)), gs.lazyMap(g => g.copy(fakeRef = false)))
      })
  }
}
