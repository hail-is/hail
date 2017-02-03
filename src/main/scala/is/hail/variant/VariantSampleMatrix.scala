package is.hail.variant

import java.nio.ByteBuffer

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.driver.HailContext
import is.hail.expr.{EvalContext, _}
import is.hail.io.annotators.IntervalListAnnotator
import is.hail.keytable.KeyTable
import is.hail.methods.Aggregators
import is.hail.sparkextras._
import is.hail.utils
import is.hail.utils._
import is.hail.variant.Variant.orderedKey
import org.apache.hadoop
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkContext, SparkEnv}
import org.json4s.jackson.Serialization

import scala.collection.JavaConverters._
import scala.language.implicitConversions
import scala.reflect.ClassTag

object VariantSampleMatrix {
  final val fileVersion: Int = 4

  def apply[T](hc: HailContext, metadata: VariantMetadata,
    rdd: OrderedRDD[Locus, Variant, (Annotation, Iterable[T])])(implicit tct: ClassTag[T]): VariantSampleMatrix[T] = {
    new VariantSampleMatrix(hc, metadata, rdd)
  }

  def writePartitioning(sqlContext: SQLContext, dirname: String): Unit = {
    val sc = sqlContext.sparkContext
    val hConf = sc.hadoopConfiguration

    if (hConf.exists(dirname + "/partitioner.json.gz")) {
      warn("write partitioning: partitioner.json.gz already exists, nothing to do")
      return
    }

    val parquetFile = dirname + "/rdd.parquet"

    val fastKeys = sqlContext.readParquetSorted(parquetFile, Some(Array("variant")))
      .map(_.getVariant(0))
    val kvRDD = fastKeys.map(k => (k, ()))

    val ordered = kvRDD.toOrderedRDD(fastKeys)

    hConf.writeTextFile(dirname + "/partitioner.json.gz") { out =>
      Serialization.write(ordered.orderedPartitioner.toJSON, out)
    }
  }

  def gen[T](hc: HailContext,
    gen: VSMSubgen[T])(implicit tct: ClassTag[T]): Gen[VariantSampleMatrix[T]] =
    gen.gen(hc)
}

case class VSMSubgen[T](
  sampleIdGen: Gen[IndexedSeq[String]],
  saSigGen: Gen[Type],
  vaSigGen: Gen[Type],
  globalSigGen: Gen[Type],
  saGen: (Type) => Gen[Annotation],
  vaGen: (Type) => Gen[Annotation],
  globalGen: (Type) => Gen[Annotation],
  vGen: Gen[Variant],
  tGen: (Int) => Gen[T],
  isDosage: Boolean = false,
  wasSplit: Boolean = false) {

  def gen(hc: HailContext)(implicit tct: ClassTag[T]): Gen[VariantSampleMatrix[T]] =
    for (size <- Gen.size;
      subsizes <- Gen.partitionSize(5).resize(size / 10);
      vaSig <- vaSigGen.resize(subsizes(0));
      saSig <- saSigGen.resize(subsizes(1));
      globalSig <- globalSigGen.resize(subsizes(2));
      global <- globalGen(globalSig).resize(subsizes(3));
      nPartitions <- Gen.choose(1, 10);

      (l, w) <- Gen.squareOfAreaAtMostSize.resize((size / 10) * 9);

      sampleIds <- sampleIdGen.resize(w);
      nSamples = sampleIds.length;
      saValues <- Gen.buildableOfN[IndexedSeq, Annotation](nSamples, saGen(saSig)).resize(subsizes(4));
      rows <- Gen.distinctBuildableOf[Seq, (Variant, (Annotation, Iterable[T]))](
        for (subsubsizes <- Gen.partitionSize(3);
          v <- vGen.resize(subsubsizes(0));
          va <- vaGen(vaSig).resize(subsubsizes(1));
          ts <- Gen.buildableOfN[Iterable, T](nSamples, tGen(v.nAlleles)).resize(subsubsizes(2)))
          yield (v, (va, ts))).resize(l))
      yield {
        VariantSampleMatrix[T](hc, VariantMetadata(sampleIds, saValues, global, saSig, vaSig, globalSig, wasSplit = wasSplit, isDosage = isDosage),
          hc.sc.parallelize(rows, nPartitions).toOrderedRDD)
      }
}

object VSMSubgen {
  val random = VSMSubgen[Genotype](
    sampleIdGen = Gen.distinctBuildableOf[IndexedSeq, String](Gen.identifier),
    saSigGen = Type.genArb,
    vaSigGen = Type.genArb,
    globalSigGen = Type.genArb,
    saGen = (t: Type) => t.genValue,
    vaGen = (t: Type) => t.genValue,
    globalGen = (t: Type) => t.genValue,
    vGen = Variant.gen,
    tGen = Genotype.genExtreme)

  val plinkSafeBiallelic = random.copy(
    sampleIdGen = Gen.distinctBuildableOf[IndexedSeq, String](Gen.plinkSafeIdentifier),
    vGen = VariantSubgen.plinkCompatible.copy(nAllelesGen = Gen.const(2)).gen,
    wasSplit = true)

  val realistic = random.copy(
    tGen = Genotype.genRealistic)

  val dosage = random.copy(
    tGen = Genotype.genDosage, isDosage = true)
}

class VariantSampleMatrix[T](val hc: HailContext, val metadata: VariantMetadata,
  val rdd: OrderedRDD[Locus, Variant, (Annotation, Iterable[T])])(implicit tct: ClassTag[T]) extends JoinAnnotator {

  def sampleIds: IndexedSeq[String] = metadata.sampleIds

  def sampleIdsAsArray: Array[String] = sampleIds.toArray

  lazy val sampleIdsBc = sparkContext.broadcast(sampleIds)

  def nSamples: Int = metadata.sampleIds.length

  def vaSignature: Type = metadata.vaSignature

  def saSignature: Type = metadata.saSignature

  def globalSignature: Type = metadata.globalSignature

  def globalAnnotation: Annotation = metadata.globalAnnotation

  def sampleAnnotations: IndexedSeq[Annotation] = metadata.sampleAnnotations

  def sampleIdsAndAnnotations: IndexedSeq[(String, Annotation)] = sampleIds.zip(sampleAnnotations)

  lazy val sampleAnnotationsBc = sparkContext.broadcast(sampleAnnotations)

  def wasSplit: Boolean = metadata.wasSplit

  def isDosage: Boolean = metadata.isDosage

  def copy[U](rdd: OrderedRDD[Locus, Variant, (Annotation, Iterable[U])] = rdd,
    sampleIds: IndexedSeq[String] = sampleIds,
    sampleAnnotations: IndexedSeq[Annotation] = sampleAnnotations,
    globalAnnotation: Annotation = globalAnnotation,
    saSignature: Type = saSignature,
    vaSignature: Type = vaSignature,
    globalSignature: Type = globalSignature,
    wasSplit: Boolean = wasSplit,
    isDosage: Boolean = isDosage)
    (implicit tct: ClassTag[U]): VariantSampleMatrix[U] =
    new VariantSampleMatrix[U](hc,
      VariantMetadata(sampleIds, sampleAnnotations, globalAnnotation,
        saSignature, vaSignature, globalSignature, wasSplit, isDosage), rdd)

  def sparkContext: SparkContext = hc.sc

  def hadoopConf: hadoop.conf.Configuration = hc.hadoopConf

  def nPartitions: Int = rdd.partitions.length

  def storageLevel: String = rdd.getStorageLevel.toReadableString()

  def variants: RDD[Variant] = rdd.keys

  def variantsAndAnnotations: OrderedRDD[Locus, Variant, Annotation] = rdd.mapValuesWithKey { case (v, (va, gs)) => va }.asOrderedRDD

  def countVariants(): Long = variants.count()

  def expand(): RDD[(Variant, String, T)] =
    mapWithKeys[(Variant, String, T)]((v, s, g) => (v, s, g))

  def expandWithAll(): RDD[(Variant, Annotation, String, Annotation, T)] =
    mapWithAll[(Variant, Annotation, String, Annotation, T)]((v, va, s, sa, g) => (v, va, s, sa, g))

  def sampleVariants(fraction: Double): VariantSampleMatrix[T] =
    copy(rdd = rdd.sample(withReplacement = false, fraction, 1).asOrderedRDD)

  def mapValues[U](f: (T) => U)(implicit uct: ClassTag[U]): VariantSampleMatrix[U] = {
    mapValuesWithAll((v, va, s, sa, g) => f(g))
  }

  def mapValuesWithKeys[U](f: (Variant, String, T) => U)
    (implicit uct: ClassTag[U]): VariantSampleMatrix[U] = {
    mapValuesWithAll((v, va, s, sa, g) => f(v, s, g))
  }

  def mapValuesWithAll[U](f: (Variant, Annotation, String, Annotation, T) => U)
    (implicit uct: ClassTag[U]): VariantSampleMatrix[U] = {
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc
    copy(rdd = rdd.mapValuesWithKey { case (v, (va, gs)) =>
      (va, localSampleIdsBc.value.lazyMapWith2[Annotation, T, U](localSampleAnnotationsBc.value, gs, {
        case (s, sa, g) => f(v, va, s, sa, g)
      }))
    }.asOrderedRDD)
  }

  def map[U](f: T => U)(implicit uct: ClassTag[U]): RDD[U] =
    mapWithKeys((v, s, g) => f(g))

  def mapWithKeys[U](f: (Variant, String, T) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc

    rdd
      .flatMap { case (v, (va, gs)) =>
        localSampleIdsBc.value.lazyMapWith[T, U](gs,
          (s, g) => f(v, s, g))
      }
  }

  def mapWithAll[U](f: (Variant, Annotation, String, Annotation, T) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    rdd
      .flatMap { case (v, (va, gs)) =>
        localSampleIdsBc.value.lazyMapWith2[Annotation, T, U](localSampleAnnotationsBc.value, gs, {
          case (s, sa, g) => f(v, va, s, sa, g)
        })
      }
  }

  def mapPartitionsWithAll[U](f: Iterator[(Variant, Annotation, String, Annotation, T)] => Iterator[U])
    (implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    rdd.mapPartitions { it =>
      f(it.flatMap { case (v, (va, gs)) =>
        localSampleIdsBc.value.lazyMapWith2[Annotation, T, (Variant, Annotation, String, Annotation, T)](
          localSampleAnnotationsBc.value, gs, { case (s, sa, g) => (v, va, s, sa, g) })
      })
    }
  }

  def mapAnnotations(f: (Variant, Annotation, Iterable[T]) => Annotation): VariantSampleMatrix[T] =
    copy[T](rdd = rdd.mapValuesWithKey { case (v, (va, gs)) => (f(v, va, gs), gs) }.asOrderedRDD)

  def flatMap[U](f: T => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] =
    flatMapWithKeys((v, s, g) => f(g))

  def flatMapWithKeys[U](f: (Variant, String, T) => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc

    rdd
      .flatMap { case (v, (va, gs)) => localSampleIdsBc.value.lazyFlatMapWith(gs,
        (s: String, g: T) => f(v, s, g))
      }
  }

  /**
    * The function {@code f} must be monotonic with respect to the ordering on {@code Locus}
    */
  def flatMapVariants(f: (Variant, Annotation, Iterable[T]) => TraversableOnce[(Variant, (Annotation, Iterable[T]))]): VariantSampleMatrix[T] =
    copy(rdd = rdd.flatMapMonotonic[(Annotation, Iterable[T])] { case (v, (va, gs)) => f(v, va, gs) })

  def filterVariants(p: (Variant, Annotation, Iterable[T]) => Boolean): VariantSampleMatrix[T] =
    copy(rdd = rdd.filter { case (v, (va, gs)) => p(v, va, gs) }.asOrderedRDD)

  def filterVariantsList(input: String, keep: Boolean): VariantSampleMatrix[T] = {
    copy(
      rdd = rdd
        .orderedLeftJoinDistinct(Variant.variantUnitRdd(sparkContext, input).toOrderedRDD)
        .mapPartitions({ it =>
          it.flatMap { case (v, ((va, gs), o)) =>
            o match {
              case Some(_) =>
                if (keep) Some((v, (va, gs))) else None
              case None =>
                if (keep) None else Some((v, (va, gs)))
            }
          }
        }, preservesPartitioning = true)
        .asOrderedRDD
    )
  }

  def filterIntervals(iList: IntervalTree[Locus], keep: Boolean): VariantSampleMatrix[T] = {
    if (keep)
      copy(rdd = rdd.filterIntervals(iList))
    else {
      val iListBc = sparkContext.broadcast(iList)
      filterVariants { (v, va, gs) => !iListBc.value.contains(v.locus)
      }
    }
  }

  def filterIntervals(path: String, keep: Boolean): VariantSampleMatrix[T] = {
    filterIntervals(IntervalListAnnotator.read(path, sparkContext.hadoopConfiguration, prune = true), keep)
  }

  def dropVariants(): VariantSampleMatrix[T] = copy(rdd = OrderedRDD.empty(sparkContext))

  def dropSamples(): VariantSampleMatrix[T] =
    copy(sampleIds = IndexedSeq.empty[String],
      sampleAnnotations = IndexedSeq.empty[Annotation],
      rdd = rdd.mapValues { case (va, gs) => (va, Iterable.empty[T]) }
        .asOrderedRDD)

  // FIXME see if we can remove broadcasts elsewhere in the code
  def filterSamples(p: (String, Annotation) => Boolean): VariantSampleMatrix[T] = {
    val mask = sampleIdsAndAnnotations.map { case (s, sa) => p(s, sa) }
    val maskBc = sparkContext.broadcast(mask)
    val localtct = tct
    copy[T](sampleIds = sampleIds.zipWithIndex
      .filter { case (s, i) => mask(i) }
      .map(_._1),
      sampleAnnotations = sampleAnnotations.zipWithIndex
        .filter { case (sa, i) => mask(i) }
        .map(_._1),
      rdd = rdd.mapValues { case (va, gs) =>
        (va, gs.lazyFilterWith(maskBc.value, (g: T, m: Boolean) => m))
      }.asOrderedRDD)
  }

  def aggregateBySample[U](zeroValue: U)(
    seqOp: (U, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(String, U)] =
    aggregateBySampleWithKeys(zeroValue)((e, v, s, g) => seqOp(e, g), combOp)

  def aggregateBySampleWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, String, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(String, U)] = {
    aggregateBySampleWithAll(zeroValue)((e, v, va, s, sa, g) => seqOp(e, v, s, g), combOp)
  }

  def aggregateBySampleWithAll[U](zeroValue: U)(
    seqOp: (U, Variant, Annotation, String, Annotation, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(String, U)] = {

    val serializer = SparkEnv.get.serializer.newInstance()
    val zeroBuffer = serializer.serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    rdd
      .mapPartitions { (it: Iterator[(Variant, (Annotation, Iterable[T]))]) =>
        val serializer = SparkEnv.get.serializer.newInstance()

        def copyZeroValue() = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))

        val arrayZeroValue = Array.fill[U](localSampleIdsBc.value.length)(copyZeroValue())

        localSampleIdsBc.value.iterator
          .zip(it.foldLeft(arrayZeroValue) { case (acc, (v, (va, gs))) =>
            for ((g, i) <- gs.iterator.zipWithIndex) {
              acc(i) = seqOp(acc(i), v, va,
                localSampleIdsBc.value(i), localSampleAnnotationsBc.value(i), g)
            }
            acc
          }.iterator)
      }.foldByKey(zeroValue)(combOp)
  }

  def aggregateByVariant[U](zeroValue: U)(
    seqOp: (U, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Variant, U)] =
    aggregateByVariantWithAll(zeroValue)((e, v, va, s, sa, g) => seqOp(e, g), combOp)

  def aggregateByVariantWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, String, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Variant, U)] = {
    aggregateByVariantWithAll(zeroValue)((e, v, va, s, sa, g) => seqOp(e, v, s, g), combOp)
  }

  def aggregateByVariantWithAll[U](zeroValue: U)(
    seqOp: (U, Variant, Annotation, String, Annotation, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Variant, U)] = {

    // Serialize the zero value to a byte array so that we can apply a new clone of it on each key
    val zeroBuffer = SparkEnv.get.serializer.newInstance().serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    rdd
      .mapPartitions({ (it: Iterator[(Variant, (Annotation, Iterable[T]))]) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        it.map { case (v, (va, gs)) =>
          val zeroValue = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))
          (v, gs.iterator.zipWithIndex.map { case (g, i) => (localSampleIdsBc.value(i), localSampleAnnotationsBc.value(i), g) }
            .foldLeft(zeroValue) { case (acc, (s, sa, g)) =>
              seqOp(acc, v, va, s, sa, g)
            })
        }
      }, preservesPartitioning = true)

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

  def aggregateByKey(keyCond: String, aggCond: String): KeyTable = {
    val aggregationST = Map(
      "global" -> (0, globalSignature),
      "v" -> (1, TVariant),
      "va" -> (2, vaSignature),
      "s" -> (3, TSample),
      "sa" -> (4, saSignature),
      "g" -> (5, TGenotype))

    val ec = EvalContext(aggregationST.map { case (name, (i, t)) => name -> (i, TAggregable(t, aggregationST)) })

    val keyEC = EvalContext(Map(
      "global" -> (0, globalSignature),
      "v" -> (1, TVariant),
      "va" -> (2, vaSignature),
      "s" -> (3, TSample),
      "sa" -> (4, saSignature),
      "g" -> (5, TGenotype)))

    val (keyNames, keyTypes, keyF) = Parser.parseNamedExprs(keyCond, keyEC)
    val (aggNames, aggTypes, aggF) = Parser.parseNamedExprs(aggCond, ec)

    val keySignature = TStruct((keyNames, keyTypes).zipped.map { case (n, t) => (n, t) }: _*)
    val valueSignature = TStruct((aggNames, aggTypes).zipped.map { case (n, t) => (n, t) }: _*)

    val (zVals, seqOp, combOp, resultOp) = Aggregators.makeFunctions[Annotation](ec, { case (ec, a) =>
      KeyTable.setEvalContext(ec, a, 6)
    })

    val localGlobalAnnotation = globalAnnotation

    val ktRDD = mapPartitionsWithAll { it =>
      it.map { case (v, va, s, sa, g) =>
        keyEC.setAll(localGlobalAnnotation, v, va, s, sa, g)
        val key = Annotation.fromSeq(keyF().map(_.orNull))
        (key, Annotation(localGlobalAnnotation, v, va, s, sa, g))
      }
    }.aggregateByKey(zVals)(seqOp, combOp)
      .map { case (k, agg) =>
        resultOp(agg)
        (k, Annotation.fromSeq(aggF().map(_.orNull)))
      }

    KeyTable(hc, ktRDD, keySignature, valueSignature)
  }

  def foldBySample(zeroValue: T)(combOp: (T, T) => T): RDD[(String, T)] = {

    val localtct = tct

    val serializer = SparkEnv.get.serializer.newInstance()
    val zeroBuffer = serializer.serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    val localSampleIdsBc = sampleIdsBc

    rdd
      .mapPartitions { (it: Iterator[(Variant, (Annotation, Iterable[T]))]) =>
        val serializer = SparkEnv.get.serializer.newInstance()

        def copyZeroValue() = serializer.deserialize[T](ByteBuffer.wrap(zeroArray))(localtct)

        val arrayZeroValue = Array.fill[T](localSampleIdsBc.value.length)(copyZeroValue())
        localSampleIdsBc.value.iterator
          .zip(it.foldLeft(arrayZeroValue) { case (acc, (v, (va, gs))) =>
            for ((g, i) <- gs.iterator.zipWithIndex)
              acc(i) = combOp(acc(i), g)
            acc
          }.iterator)
      }.foldByKey(zeroValue)(combOp)
  }

  def foldByVariant(zeroValue: T)(combOp: (T, T) => T): RDD[(Variant, T)] =
    rdd.mapValues { case (va, gs) => gs.foldLeft(zeroValue)((acc, g) => combOp(acc, g)) }

  def sampleAnnotationsSimilar(that: VariantSampleMatrix[T], tolerance: Double = utils.defaultTolerance): Boolean = {
    require(saSignature == that.saSignature)
    sampleAnnotations.zip(that.sampleAnnotations)
      .forall { case (s1, s2) => saSignature.valuesSimilar(s1, s2, tolerance) }
  }

  def same(that: VariantSampleMatrix[T], tolerance: Double = utils.defaultTolerance): Boolean = {
    var metadataSame = true
    if (vaSignature != that.vaSignature) {
      metadataSame = false
      println(
        s"""different va signature:
           |  left:  ${ vaSignature.toPrettyString(compact = true) }
           |  right: ${ that.vaSignature.toPrettyString(compact = true) }""".stripMargin)
    }
    if (saSignature != that.saSignature) {
      metadataSame = false
      println(
        s"""different sa signature:
           |  left:  ${ saSignature.toPrettyString(compact = true) }
           |  right: ${ that.saSignature.toPrettyString(compact = true) }""".stripMargin)
    }
    if (globalSignature != that.globalSignature) {
      metadataSame = false
      println(
        s"""different global signature:
           |  left:  ${ globalSignature.toPrettyString(compact = true) }
           |  right: ${ that.globalSignature.toPrettyString(compact = true) }""".stripMargin)
    }
    if (sampleIds != that.sampleIds) {
      metadataSame = false
      println(
        s"""different sample ids:
           |  left:  $sampleIds
           |  right: ${ that.sampleIds }""".stripMargin)
    }
    if (!sampleAnnotationsSimilar(that, tolerance)) {
      metadataSame = false
      println(
        s"""different sample annotations:
           |  left:  $sampleAnnotations
           |  right: ${ that.sampleAnnotations }""".stripMargin)
    }
    if (sampleIds != that.sampleIds) {
      metadataSame = false
      println(
        s"""different global annotation:
           |  left:  $globalAnnotation
           |  right: ${ that.globalAnnotation }""".stripMargin)
    }
    if (wasSplit != that.wasSplit) {
      metadataSame = false
      println(
        s"""different was split:
           |  left:  $wasSplit
           |  right: ${ that.wasSplit }""".stripMargin)
    }
    if (!metadataSame)
      println("metadata were not the same")
    val vaSignatureBc = sparkContext.broadcast(vaSignature)
    var printed = false
    metadataSame &&
      rdd
        .fullOuterJoin(that.rdd)
        .forall {
          case (v, (Some((va1, it1)), Some((va2, it2)))) =>
            val annotationsSame = vaSignatureBc.value.valuesSimilar(va1, va2, tolerance)
            if (!annotationsSame && !printed) {
              println(
                s"""at variant `$v', annotations were not the same:
                   |  $va1
                   |  $va2
                 """.stripMargin)
              printed = true
            }
            val genotypesSame = (it1, it2).zipped.forall { case (g1, g2) =>
              if (g1 != g2)
                println(s"genotypes $g1, $g2 were not the same")
              g1 == g2
            }
            annotationsSame && genotypesSame
          case (v, _) =>
            println(s"Found unmatched variant $v")
            false
        }
  }

  def mapAnnotationsWithAggregate[U](zeroValue: U, newVAS: Type)(
    seqOp: (U, Variant, Annotation, String, Annotation, T) => U,
    combOp: (U, U) => U,
    mapOp: (Annotation, U) => Annotation)
    (implicit uct: ClassTag[U]): VariantSampleMatrix[T] = {

    // Serialize the zero value to a byte array so that we can apply a new clone of it on each key
    val zeroBuffer = SparkEnv.get.serializer.newInstance().serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    copy(vaSignature = newVAS,
      rdd = rdd.mapValuesWithKey { case (v, (va, gs)) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        val zeroValue = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))

        (mapOp(va, gs.iterator
          .zip(localSampleIdsBc.value.iterator
            .zip(localSampleAnnotationsBc.value.iterator)).foldLeft(zeroValue) {
          case (acc, (g, (s, sa))) =>
            seqOp(acc, v, va, s, sa, g)
        }), gs)
      }.asOrderedRDD)
  }

  def annotateIntervals(is: IntervalTree[Locus],
    path: List[String]): VariantSampleMatrix[T] = {
    val isBc = sparkContext.broadcast(is)
    val (newSignature, inserter) = insertVA(TBoolean, path)
    copy(rdd = rdd.mapValuesWithKey { case (v, (va, gs)) =>
      (inserter(va, Some(isBc.value.contains(Locus(v.contig, v.start)))), gs)
    }.asOrderedRDD,
      vaSignature = newSignature)
  }

  def annotateIntervals(is: IntervalTree[Locus],
    t: Type,
    m: Map[Interval[Locus], List[String]],
    all: Boolean,
    path: List[String]): VariantSampleMatrix[T] = {
    val isBc = sparkContext.broadcast(is)

    val mBc = sparkContext.broadcast(m)
    val (newSignature, inserter) = insertVA(
      if (all) TSet(t) else t,
      path)
    copy(rdd = rdd.mapValuesWithKey { case (v, (va, gs)) =>
      val queries = isBc.value.query(v.locus)
      val toIns = if (all)
        Some(queries.flatMap(mBc.value))
      else {
        queries.flatMap(mBc.value).headOption
      }
      (inserter(va, toIns), gs)
    }.asOrderedRDD,
      vaSignature = newSignature)
  }

  def annotateVariants(otherRDD: OrderedRDD[Locus, Variant, Annotation], signature: Type,
    code: String): VariantSampleMatrix[T] = {
    val (newSignature, ins) = insertVA(signature, Parser.parseAnnotationRoot(code, Annotation.VARIANT_HEAD))
    annotateVariants(otherRDD, newSignature, ins)
  }


  def annotateVariants(otherRDD: OrderedRDD[Locus, Variant, Annotation], newSignature: Type,
    inserter: Inserter): VariantSampleMatrix[T] = {
    val newRDD = rdd.orderedLeftJoinDistinct(otherRDD)
      .mapValues { case ((va, gs), annotation) =>
        (inserter(va, annotation), gs)
      }.asOrderedRDD
    copy(rdd = newRDD, vaSignature = newSignature)
  }

  def annotateLoci(lociRDD: OrderedRDD[Locus, Locus, Annotation], newSignature: Type, inserter: Inserter): VariantSampleMatrix[T] = {

    import LocusImplicits.orderedKey

    val newRDD = rdd
      .mapMonotonic(OrderedKeyFunction(_.locus), { case (v, vags) => (v, vags) })
      .orderedLeftJoinDistinct(lociRDD)
      .map { case (l, ((v, (va, gs)), annotation)) => (v, (inserter(va, annotation), gs)) }

    // we safely use the non-shuffling apply method of OrderedRDD because orderedLeftJoinDistinct preserves the
    // (Variant) ordering of the left RDD
    val orderedRDD = OrderedRDD(newRDD, rdd.orderedPartitioner)
    copy(rdd = orderedRDD, vaSignature = newSignature)
  }

  def annotateSamples(annotations: Map[String, Annotation], signature: Type, code: String): VariantSampleMatrix[T] = {
    val (t, i) = insertSA(signature, Parser.parseAnnotationRoot(code, Annotation.SAMPLE_HEAD))
    annotateSamples(annotations.get _, t, i)
  }

  def annotateSamples(signature: Type, path: List[String], annotation: (String) => Option[Annotation]): VariantSampleMatrix[T] = {
    val (t, i) = insertSA(signature, path)
    annotateSamples(annotation, t, i)
  }

  def annotateSamples(annotation: (String) => Option[Annotation], newSignature: Type, inserter: Inserter): VariantSampleMatrix[T] = {
    val newAnnotations = sampleIds.zipWithIndex.map { case (id, i) =>
      val sa = sampleAnnotations(i)
      val newAnnotation = annotation(id)
      newAnnotation.foreach(newSignature.typeCheck)
      inserter(sa, newAnnotation)
    }

    copy(sampleAnnotations = newAnnotations, saSignature = newSignature)
  }

  def annotateGlobal(a: Annotation, t: Type, code: String): VariantSampleMatrix[T] = {
    val (newT, i) = insertGlobal(t, Parser.parseAnnotationRoot(code, Annotation.GLOBAL_HEAD))
    copy(globalSignature = newT, globalAnnotation = i(globalAnnotation, Option(a)))
  }

  def queryVA(code: String): (Type, Querier) = {

    val st = Map(Annotation.VARIANT_HEAD -> (0, vaSignature))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parseExpr(code, ec)

    val f2: Annotation => Option[Any] = { annotation =>
      a(0) = annotation
      f()
    }

    (t, f2)
  }

  def querySA(code: String): (Type, Querier) = {

    val st = Map(Annotation.SAMPLE_HEAD -> (0, saSignature))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parseExpr(code, ec)

    val f2: Annotation => Option[Any] = { annotation =>
      a(0) = annotation
      f()
    }

    (t, f2)
  }

  def queryGlobal(path: String): (Type, Option[Annotation]) = {
    val st = Map(Annotation.GLOBAL_HEAD -> (0, globalSignature))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parseExpr(path, ec)

    val f2: Annotation => Option[Any] = { annotation =>
      a(0) = annotation
      f()
    }

    (t, f2(globalAnnotation))
  }

  def deleteVA(args: String*): (Type, Deleter) = deleteVA(args.toList)

  def deleteVA(path: List[String]): (Type, Deleter) = vaSignature.delete(path)

  def deleteSA(args: String*): (Type, Deleter) = deleteSA(args.toList)

  def deleteSA(path: List[String]): (Type, Deleter) = saSignature.delete(path)

  def deleteGlobal(args: String*): (Type, Deleter) = deleteGlobal(args.toList)

  def deleteGlobal(path: List[String]): (Type, Deleter) = globalSignature.delete(path)

  def insertVA(sig: Type, args: String*): (Type, Inserter) = insertVA(sig, args.toList)

  def insertVA(sig: Type, path: List[String]): (Type, Inserter) = {
    vaSignature.insert(sig, path)
  }

  def insertSA(sig: Type, args: String*): (Type, Inserter) = insertSA(sig, args.toList)

  def insertSA(sig: Type, path: List[String]): (Type, Inserter) = saSignature.insert(sig, path)

  def insertGlobal(sig: Type, args: String*): (Type, Inserter) = insertGlobal(sig, args.toList)

  def insertGlobal(sig: Type, path: List[String]): (Type, Inserter) = {
    globalSignature.insert(sig, path)
  }

  override def toString = s"VariantSampleMatrix(metadata=$metadata, rdd=$rdd, sampleIds=$sampleIds, nSamples=$nSamples, vaSignature=$vaSignature, saSignature=$saSignature, globalSignature=$globalSignature, sampleAnnotations=$sampleAnnotations, sampleIdsAndAnnotations=$sampleIdsAndAnnotations, globalAnnotation=$globalAnnotation, wasSplit=$wasSplit)"

  def variantsKT(): KeyTable = {
    val localVASignature = vaSignature
    KeyTable(hc, rdd.map { case (v, (va, gs)) =>
      Annotation(v, va)
    },
      TStruct(
        "v" -> TVariant,
        "va" -> vaSignature),
      Array("v"))
  }

  def samplesKT(): KeyTable = {
    KeyTable(hc, sparkContext.parallelize(sampleIdsAndAnnotations)
      .map { case (s, sa) =>
        Annotation(s, sa)
      },
      TStruct(
        "s" -> TSample,
        "sa" -> saSignature),
      Array("s"))
  }

  def querySamples(expr: String): (Annotation, Type) = {
    val qs = querySamples(Array(expr))
    assert(qs.length == 1)
    qs.head
  }

  def querySamples(exprs: Array[String]): Array[(Annotation, Type)] = {
    val aggregationST = Map(
      "global" -> (0, globalSignature),
      "s" -> (1, TSample),
      "sa" -> (2, saSignature))
    val ec = EvalContext(Map(
      "global" -> (0, globalSignature),
      "samples" -> (1, TAggregable(TSample, aggregationST))))

    val ts = exprs.map(e => Parser.parseExpr(e, ec))

    val localGlobalAnnotation = globalAnnotation
    val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[(String, Annotation)](ec, { case (ec, (s, sa)) =>
      ec.setAll(localGlobalAnnotation, s, sa)
    })

    val results = sampleIdsAndAnnotations
      .aggregate(zVal)(seqOp, combOp)
    resOp(results)
    ec.set(0, localGlobalAnnotation)

    ts.map { case (t, f) => (f().orNull, t) }.toArray
  }

  def queryVariants(expr: String): (Annotation, Type) = {
    val qv = queryVariants(Array(expr))
    assert(qv.length == 1)
    qv.head
  }

  def queryVariants(exprs: Array[String]): Array[(Annotation, Type)] = {

    val aggregationST = Map(
      "global" -> (0, globalSignature),
      "v" -> (1, TVariant),
      "va" -> (2, vaSignature))
    val ec = EvalContext(Map(
      "global" -> (0, globalSignature),
      "variants" -> (1, TAggregable(TVariant, aggregationST))))

    val ts = exprs.map(e => Parser.parseExpr(e, ec))

    val localGlobalAnnotation = globalAnnotation
    val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[(Variant, Annotation)](ec, { case (ec, (v, va)) =>
      ec.setAll(localGlobalAnnotation, v, va)
    })

    val result = variantsAndAnnotations
      .treeAggregate(zVal)(seqOp, combOp, depth = treeAggDepth(hc, nPartitions))
    resOp(result)

    ec.setAll(localGlobalAnnotation)
    ts.map { case (t, f) => (f().orNull, t) }.toArray
  }

  def annotateVariantsKeyTable(kt: KeyTable, code: String): VariantSampleMatrix[T] = {
    val ktKeyTypes = kt.keySignature.fields.map(_.typ)

    if (ktKeyTypes.size != 1 || ktKeyTypes(0) != TVariant)
      fatal(s"Key signature of KeyTable must be 1 field with type `Variant'. Found `${ kt.keySignature }'")

    val ktSig = kt.signature

    val inserterEc = EvalContext(Map("va" -> (0, vaSignature), "table" -> (1, ktSig)))

    val (finalType, inserter) =
      buildInserter(code, vaSignature, inserterEc, Annotation.VARIANT_HEAD)

    val keyedRDD = kt.rdd.map { case (k: Row, v) => (k(0).asInstanceOf[Variant], kt.mergeKeyAndValue(k, v)) }

    val ordRdd = OrderedRDD(keyedRDD, None, None)

    annotateVariants(ordRdd, finalType, inserter)
  }

  def annotateVariantsKeyTable(kt: KeyTable, vdsKey: java.util.ArrayList[String], code: String): VariantSampleMatrix[T] =
    annotateVariantsKeyTable(kt, vdsKey.asScala, code)

  def annotateVariantsKeyTable(kt: KeyTable, vdsKey: Seq[String], code: String): VariantSampleMatrix[T] = {
    val vdsKeyEc = EvalContext(Map("v" -> (0, TVariant), "va" -> (1, vaSignature)))

    val (vdsKeyType, vdsKeyFs) = vdsKey.map(Parser.parseExpr(_, vdsKeyEc)).unzip

    val keyTypes = kt.keySignature.fields.map(_.typ)
    if (keyTypes != vdsKeyType)
      fatal(s"Key signature of KeyTable, `$keyTypes', must match type of computed key, `$vdsKeyType'.")

    val ktSig = kt.signature

    val inserterEc = EvalContext(Map("va" -> (0, vaSignature), "table" -> (1, ktSig)))

    val (finalType, inserter) =
      buildInserter(code, vaSignature, inserterEc, Annotation.VARIANT_HEAD)

    val ktRdd = kt.rdd.map { case (k, v) => (k, kt.mergeKeyAndValue(k, v)) }

    val thisRdd = rdd.map { case (v, (va, gs)) =>
      vdsKeyEc.setAll(v, va)
      (Annotation.fromSeq(vdsKeyFs.map(f => f().orNull)), (v, va))
    }

    val variantKeyedRdd = ktRdd.join(thisRdd)
      .map { case (_, (table, (v, va))) => (v, inserter(va, Some(table))) }

    val ordRdd = OrderedRDD(variantKeyedRdd, None, None)

    val newRdd = rdd.orderedLeftJoinDistinct(ordRdd)
      .mapValues { case ((va, gs), optVa) => (optVa.getOrElse(va), gs) }
      .asOrderedRDD

    copy(rdd = newRdd, vaSignature = finalType)
  }

  def minrep(maxShift: Int = 100): VariantSampleMatrix[T] = {
    require(maxShift > 0, s"invalid value for maxShift: $maxShift. Parameter must be a positive integer.")
    val minrepped = rdd.map {
      case (v, (va, gs)) =>
        (v.minrep, (va, gs))
    }
    copy(rdd = minrepped.smartShuffleAndSort(rdd.orderedPartitioner, maxShift))
  }
}
