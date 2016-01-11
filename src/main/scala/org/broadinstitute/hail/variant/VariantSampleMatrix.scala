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

private object JoinUtils {
  def reindexSamplesOption[T](vsm: VariantSampleMatrix[T], mergedSampleIds: IndexedSeq[String], mergedLocalSamples: Array[Int])
                             (implicit ttt: TypeTag[T], tct: ClassTag[T]): VariantSampleMatrix[Option[T]] = {
    val tctLocal = tct
    val indexMapping = for (i <- vsm.localSamples) yield mergedLocalSamples.indexOf(mergedSampleIds.indexOf(vsm.sampleIds(i)))
    new VariantSampleMatrix[Option[T]](new VariantMetadata(vsm.metadata.filters, mergedSampleIds, vsm.metadata.sampleAnnotations,
      vsm.metadata.sampleAnnotationSignatures, vsm.metadata.variantAnnotationSignatures),
      mergedLocalSamples, vsm.rdd.map { case (v, a, s) => (v, a, reorderGenotypesOption(s, mergedLocalSamples.length, indexMapping)(tctLocal)) })
  }

  def reindexSamplesNoOption[T](vsm: VariantSampleMatrix[T], mergedSampleIds: IndexedSeq[String], mergedLocalSamples: Array[Int])
                               (implicit ttt: TypeTag[T], tct: ClassTag[T]): VariantSampleMatrix[T] = {
    require(mergedSampleIds.exists { id => vsm.localSamples.contains(vsm.sampleIds.indexOf(id)) })
    val tctLocal = tct
    val indexMapping = for (i <- vsm.localSamples) yield mergedLocalSamples.indexOf(mergedSampleIds.indexOf(vsm.sampleIds(i)))
    new VariantSampleMatrix[T](new VariantMetadata(vsm.metadata.filters, mergedSampleIds, vsm.metadata.sampleAnnotations,
      vsm.metadata.sampleAnnotationSignatures, vsm.metadata.variantAnnotationSignatures),
      mergedLocalSamples, vsm.rdd.map { case (v, a, s) => (v, a, reorderGenotypesNoOption(s, mergedLocalSamples.length, indexMapping)(tctLocal)) })
  }

  def mergeLocalSamples[T, S](vsm1: VariantSampleMatrix[T], vsm2: VariantSampleMatrix[S], mergedSampleIds: IndexedSeq[String])
                             (implicit tct: ClassTag[T], sct: ClassTag[S]): Array[Int] = {
    val localIds = vsm1.localSamples.map(vsm1.sampleIds) ++ vsm2.localSamples.map(vsm2.sampleIds)
    for ((s, i) <- mergedSampleIds.toArray.zipWithIndex if localIds.contains(s)) yield i
  }

  def reorderGenotypesNoOption[T](gts: Iterable[T], nLocalSamples: Int, indexMapping: Array[Int])
                                 (implicit tct: ClassTag[T]): Iterable[T] = {
    val newGenotypes = new Array[T](nLocalSamples)
    for ((g, i) <- gts.zipWithIndex) {
      val newIndex = indexMapping(i)
      if (newIndex != -1)
        newGenotypes(newIndex) = g
    }
    newGenotypes.toIterable
  }

  def reorderGenotypesOption[T](gts: Iterable[T], nLocalSamples: Int, indexMapping: Array[Int])
                               (implicit tct: ClassTag[T]): Iterable[Option[T]] = {
    val newGenotypes = Array.fill[Option[T]](nLocalSamples)(None)
    for ((g, i) <- gts.zipWithIndex) {
      val newIndex = indexMapping(i)
      if (newIndex != -1)
        newGenotypes(newIndex) = Some(g)
    }
    newGenotypes.toIterable
  }

  def sampleInnerJoin[T, S](vsm1: VariantSampleMatrix[T], vsm2: VariantSampleMatrix[S])
                           (implicit ttt: TypeTag[T], tct: ClassTag[T], stt: TypeTag[S], sct: ClassTag[S]): (VariantSampleMatrix[T], VariantSampleMatrix[S]) = {
    val mergedSampleIds = vsm1.sampleIds.toSet.intersect(vsm2.sampleIds.toSet).toIndexedSeq
    val mergedLocalSamples = mergeLocalSamples(vsm1, vsm2, mergedSampleIds)
    (reindexSamplesNoOption(vsm1, mergedSampleIds, mergedLocalSamples)(ttt, tct), reindexSamplesNoOption(vsm2, mergedSampleIds, mergedLocalSamples)(stt, sct))
  }

  def sampleOuterJoin[T, S](vsm1: VariantSampleMatrix[T], vsm2: VariantSampleMatrix[S])
                           (implicit ttt: TypeTag[T], tct: ClassTag[T], stt: TypeTag[S], sct: ClassTag[S]): (VariantSampleMatrix[Option[T]], VariantSampleMatrix[Option[S]]) = {
    val mergedSampleIds = vsm1.sampleIds.toSet.union(vsm2.sampleIds.toSet).toIndexedSeq
    val mergedLocalSamples = mergeLocalSamples(vsm1, vsm2, mergedSampleIds)
    (reindexSamplesOption(vsm1, mergedSampleIds, mergedLocalSamples)(ttt, tct), reindexSamplesOption(vsm2, mergedSampleIds, mergedLocalSamples)(stt, sct))
  }

  def sampleLeftJoin[T, S](vsm1: VariantSampleMatrix[T], vsm2: VariantSampleMatrix[S])
                          (implicit ttt: TypeTag[T], tct: ClassTag[T], stt: TypeTag[S], sct: ClassTag[S]): (VariantSampleMatrix[T], VariantSampleMatrix[Option[S]]) = {
    val mergedSampleIds = vsm1.sampleIds
    val mergedLocalSamples = mergeLocalSamples(vsm1, vsm2, mergedSampleIds)
    (reindexSamplesNoOption(vsm1, mergedSampleIds, mergedLocalSamples)(ttt,tct), reindexSamplesOption(vsm2, mergedSampleIds, mergedLocalSamples)(stt, sct))
  }

  def sampleRightJoin[T, S](vsm1: VariantSampleMatrix[T], vsm2: VariantSampleMatrix[S])
                           (implicit ttt: TypeTag[T], tct: ClassTag[T], stt: TypeTag[S], sct: ClassTag[S]): (VariantSampleMatrix[Option[T]], VariantSampleMatrix[S]) = {
    val mergedSampleIds = vsm2.sampleIds
    val mergedLocalSamples = mergeLocalSamples(vsm1, vsm2, mergedSampleIds)
    (reindexSamplesOption(vsm1, mergedSampleIds, mergedLocalSamples)(ttt, tct), reindexSamplesNoOption(vsm2, mergedSampleIds, mergedLocalSamples)(stt, sct))
  }

  def variantInnerJoin[T, S](vsm1: VariantSampleMatrix[T], vsm2: VariantSampleMatrix[S])
                            (implicit tct: ClassTag[T], sct: ClassTag[S]): RDD[(Variant, ((AnnotationData, Iterable[T]), (AnnotationData,Iterable[S])))] = {
    vsm1.rdd.map{case (v,a,gs) => (v,(a,gs))}.join(vsm2.rdd.map{case (v,a,gs) => (v,(a,gs))})
  }

  def variantLeftJoin[T, S](vsm1: VariantSampleMatrix[T], vsm2: VariantSampleMatrix[S])
                           (implicit tct: ClassTag[T], sct: ClassTag[S]): RDD[(Variant, ((AnnotationData,Iterable[T]), (Option[AnnotationData],Option[Iterable[S]])))] = {
    vsm1.rdd.map{case (v,a,gs) => (v,(a,gs))}.leftOuterJoin(vsm2.rdd.map{case (v,a,gs) => (v,(a,gs))})
      .map{case (v,(d1,d2)) => (v,(d1,d2.map{case (a,gs) => (Some(a),Some(gs))}.getOrElse((None,None))))}
  }

  def variantRightJoin[T, S](vsm1: VariantSampleMatrix[T], vsm2: VariantSampleMatrix[S])
                            (implicit tct: ClassTag[T], sct: ClassTag[S]): RDD[(Variant, ((Option[AnnotationData],Option[Iterable[T]]),(AnnotationData, Iterable[S])))] = {
    vsm1.rdd.map{case (v,a,gs) => (v,(a,gs))}.rightOuterJoin(vsm2.rdd.map{case (v,a,gs) => (v,(a,gs))})
      .map{case (v,(d1,d2)) => (v,(d1.map{case (a,gs) => (Some(a),Some(gs))}.getOrElse((None,None)),d2))}
  }

  def variantOuterJoin[T, S](vsm1: VariantSampleMatrix[T], vsm2: VariantSampleMatrix[S])
                            (implicit tct: ClassTag[T], sct: ClassTag[S]): RDD[(Variant, ((Option[AnnotationData],Option[Iterable[T]]),(Option[AnnotationData],Option[Iterable[S]])))] = {
    vsm1.rdd.map{case (v,a,gs) => (v,(a,gs))}.fullOuterJoin(vsm2.rdd.map{case (v,a,gs) => (v,(a,gs))})
      .map{case (v,(d1,d2)) => (v,(d1.map{case (a,gs) => (Some(a),Some(gs))}.getOrElse((None,None)),d2.map{case (a,gs) => (Some(a),Some(gs))}.getOrElse((None,None))))}
  }

  def annotationInnerJoin[T, S](a: AnnotationData, b: AnnotationData): AnnotationData = {
    a
  }

  def annotationLeftJoin[T, S](a: AnnotationData, b: Option[AnnotationData]): AnnotationData = {
    a
  }

  def annotationRightJoin[T, S](a: Option[AnnotationData], b: AnnotationData): AnnotationData = {
    b
  }

  def annotationOuterJoin[T, S](a: Option[AnnotationData], b: Option[AnnotationData]): AnnotationData = {
    Array(a,b).flatten.apply(0)
  }

  def genotypeInnerInnerJoin[T, S](a: Iterable[T], b: Iterable[S], nSamples: Int)
                                  (implicit tct: ClassTag[T], sct: ClassTag[S]): Iterable[(T, S)] = {
    require(a.size == b.size)
    a.zip(b)
  }

  def genotypeInnerLeftJoin[T, S](a: Iterable[T], b: Option[Iterable[S]], nSamples: Int)
                                 (implicit tct: ClassTag[T], sct: ClassTag[S]): Iterable[(T, Option[S])] = {
    require(nSamples == a.size)
    val bPrime: Iterable[Option[S]] = b.map(_.map(s => Some(s))).getOrElse(Array.fill[Option[S]](nSamples)(None).toIterable)
    require(a.size == bPrime.size)
    a.zip(bPrime)
  }

  def genotypeInnerRightJoin[T, S](a: Option[Iterable[T]], b: Iterable[S], nSamples: Int)
                                  (implicit tct: ClassTag[T], sct: ClassTag[S]): Iterable[(Option[T], S)] = {
    require(nSamples == b.size)
    val aPrime: Iterable[Option[T]] = a.map(_.map(s => Some(s))).getOrElse(Array.fill[Option[T]](nSamples)(None).toIterable)
    require(aPrime.size == b.size)
    aPrime.zip(b)
  }

  def genotypeInnerOuterJoin[T, S](a: Option[Iterable[T]], b: Option[Iterable[S]], nSamples: Int)
                                  (implicit tct: ClassTag[T], sct: ClassTag[S]): Iterable[(Option[T], Option[S])] = {
    val aPrime: Iterable[Option[T]] = a.map(_.map(s => Some(s))).getOrElse(Array.fill[Option[T]](nSamples)(None).toIterable)
    val bPrime: Iterable[Option[S]] = b.map(_.map(s => Some(s))).getOrElse(Array.fill[Option[S]](nSamples)(None).toIterable)
    require(aPrime.size == bPrime.size)
    aPrime.zip(bPrime)
  }

  def genotypeLeftInnerJoin[T, S](a: Iterable[T], b: Iterable[Option[S]], nSamples: Int)
                                 (implicit tct: ClassTag[T], sct: ClassTag[S]): Iterable[(T, Option[S])] = {
    require(a.size == b.size)
    a.zip(b)
  }

  def genotypeLeftLeftJoin[T, S](a: Iterable[T], b: Option[Iterable[Option[S]]], nSamples: Int)
                                (implicit tct: ClassTag[T], sct: ClassTag[S]): Iterable[(T, Option[S])] = {
    require(nSamples == a.size)
    val bPrime: Iterable[Option[S]] = b.getOrElse(Array.fill[Option[S]](nSamples)(None).toIterable)
    require(a.size == bPrime.size)
    a.zip(bPrime)
  }

  def genotypeLeftRightJoin[T, S](a: Option[Iterable[T]], b: Iterable[Option[S]], nSamples: Int)
                                 (implicit tct: ClassTag[T], sct: ClassTag[S]): Iterable[(Option[T], Option[S])] = {
    require(nSamples == b.size)
    val aPrime: Iterable[Option[T]] = a.map(_.map(t => Some(t))).getOrElse(Array.fill[Option[T]](nSamples)(None).toIterable)
    require(aPrime.size == b.size)
    aPrime.zip(b)
  }

  def genotypeLeftOuterJoin[T, S](a: Option[Iterable[T]], b: Option[Iterable[Option[S]]], nSamples: Int)
                                 (implicit tct: ClassTag[T], sct: ClassTag[S]): Iterable[(Option[T], Option[S])] = {
    val aPrime: Iterable[Option[T]] = a.map(_.map(t => Some(t))).getOrElse(Array.fill[Option[T]](nSamples)(None).toIterable)
    val bPrime: Iterable[Option[S]] = b.getOrElse(Array.fill[Option[S]](nSamples)(None).toIterable)
    require(aPrime.size == bPrime.size)
    aPrime.zip(bPrime)
  }

  def genotypeRightInnerJoin[T, S](a: Iterable[Option[T]], b: Iterable[S], nSamples: Int)
                                  (implicit tct: ClassTag[T], sct: ClassTag[S]): Iterable[(Option[T], S)] = {
    require(a.size == b.size)
    a.zip(b)
  }

  def genotypeRightLeftJoin[T, S](a: Iterable[Option[T]], b: Option[Iterable[S]], nSamples: Int)
                                 (implicit tct: ClassTag[T], sct: ClassTag[S]): Iterable[(Option[T], Option[S])] = {
    require(nSamples == a.size)
    val bPrime: Iterable[Option[S]] = b.map(_.map(s => Some(s))).getOrElse(Array.fill[Option[S]](nSamples)(None).toIterable)
    require(a.size == bPrime.size)
    a.zip(bPrime)
  }

  def genotypeRightRightJoin[T, S](a: Option[Iterable[Option[T]]], b: Iterable[S], nSamples: Int)
                                  (implicit tct: ClassTag[T], sct: ClassTag[S]): Iterable[(Option[T], S)] = {
    require(nSamples == b.size)
    val aPrime: Iterable[Option[T]] = a.getOrElse(Array.fill[Option[T]](nSamples)(None).toIterable)
    require(aPrime.size == b.size)
    aPrime.zip(b)
  }

  def genotypeRightOuterJoin[T, S](a: Option[Iterable[Option[T]]], b: Option[Iterable[S]], nSamples: Int)
                                  (implicit tct: ClassTag[T], sct: ClassTag[S]): Iterable[(Option[T], Option[S])] = {
    val aPrime: Iterable[Option[T]] = a.getOrElse(Array.fill[Option[T]](nSamples)(None).toIterable)
    val bPrime: Iterable[Option[S]] = b.map(_.map(s => Some(s))).getOrElse(Array.fill[Option[S]](nSamples)(None).toIterable)
    require(aPrime.size == bPrime.size)
    aPrime.zip(bPrime)
  }

  def genotypeOuterInnerJoin[T, S](a: Iterable[Option[T]], b: Iterable[Option[S]], nSamples: Int)
                                  (implicit tct: ClassTag[T], sct: ClassTag[S]): Iterable[(Option[T], Option[S])] = {
    require(a.size == b.size)
    a.zip(b)
  }

  def genotypeOuterLeftJoin[T, S](a: Iterable[Option[T]], b: Option[Iterable[Option[S]]], nSamples: Int)
                                 (implicit tct: ClassTag[T], sct: ClassTag[S]): Iterable[(Option[T], Option[S])] = {
    require(nSamples == a.size)
    val bPrime: Iterable[Option[S]] = b.getOrElse(Array.fill[Option[S]](nSamples)(None).toIterable)
    require(a.size == bPrime.size)
    a.zip(bPrime)
  }

  def genotypeOuterRightJoin[T, S](a: Option[Iterable[Option[T]]], b: Iterable[Option[S]], nSamples: Int)
                                  (implicit tct: ClassTag[T], sct: ClassTag[S]): Iterable[(Option[T], Option[S])] = {
    require(nSamples == b.size)
    val aPrime: Iterable[Option[T]] = a.getOrElse(Array.fill[Option[T]](nSamples)(None).toIterable)
    require(aPrime.size == b.size)
    aPrime.zip(b)
  }

  def genotypeOuterOuterJoin[T, S](a: Option[Iterable[Option[T]]], b: Option[Iterable[Option[S]]], nSamples: Int)
                                  (implicit tct: ClassTag[T], sct: ClassTag[S]): Iterable[(Option[T], Option[S])] = {
    val aPrime: Iterable[Option[T]] = a.getOrElse(Array.fill[Option[T]](nSamples)(None).toIterable)
    val bPrime: Iterable[Option[S]] = b.getOrElse(Array.fill[Option[S]](nSamples)(None).toIterable)
    require(aPrime.size == bPrime.size)
    aPrime.zip(bPrime)
  }
}

object VariantSampleMatrix {
  def apply(metadata: VariantMetadata,
            rdd: RDD[(Variant, AnnotationData, Iterable[Genotype])]): VariantDataset = {
    new VariantSampleMatrix(metadata, rdd)

  }

  def read(sqlContext: SQLContext, dirname: String): VariantDataset = {
    require(dirname.endsWith(".vds"))
    import RichRow._

    val (localSamples, metadata) = readObjectFile(dirname + "/metadata.ser",
      sqlContext.sparkContext.hadoopConfiguration) { s =>
      (s.readObject().asInstanceOf[Array[Int]],
        s.readObject().asInstanceOf[VariantMetadata])
    }

    // val df = sqlContext.read.parquet(dirname + "/rdd.parquet")
    val df = sqlContext.parquetFile(dirname + "/rdd.parquet")
    // FIXME annotations
    new VariantSampleMatrix[Genotype](metadata,
      localSamples,
      df.rdd.map(r =>
        (r.getVariant(0), r.getVariantAnnotations(1), r.getGenotypeStream(2))))
  }
}

class VariantSampleMatrix[T](val metadata: VariantMetadata,
                             val localSamples: Array[Int],
                             val rdd: RDD[(Variant, AnnotationData, Iterable[T])])
                            (implicit ttt: TypeTag[T], tct: ClassTag[T],
                             vct: ClassTag[Variant]) {

  import JoinUtils._

  def this(metadata: VariantMetadata, rdd: RDD[(Variant, AnnotationData, Iterable[T])])
          (implicit ttt: TypeTag[T], tct: ClassTag[T]) =
    this(metadata, Array.range(0, metadata.nSamples), rdd)

  def sampleIds: IndexedSeq[String] = metadata.sampleIds

  def nSamples: Int = metadata.sampleIds.length

  def nLocalSamples: Int = localSamples.length

  def copy[U](metadata: VariantMetadata = metadata,
              localSamples: Array[Int] = localSamples,
              rdd: RDD[(Variant, AnnotationData, Iterable[U])] = rdd)
             (implicit ttt: TypeTag[U], tct: ClassTag[U]): VariantSampleMatrix[U] =
    new VariantSampleMatrix(metadata, localSamples, rdd)

  def sparkContext: SparkContext = rdd.sparkContext

  def cache(): VariantSampleMatrix[T] = copy[T](rdd = rdd.cache())

  def repartition(nPartitions: Int) = copy[T](rdd = rdd.repartition(nPartitions)(null))

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

  def mapValuesWithPartialApplication[U](f: (Variant, AnnotationData) => ((Int, T) => U))
                                        (implicit utt: TypeTag[U], uct: ClassTag[U]): VariantSampleMatrix[U] = {
    val localSamplesBc = sparkContext.broadcast(localSamples)
    copy(rdd = rdd.map { case (v, va, gs) =>
      val f2 = f(v, va)
      (v, va, localSamplesBc.value.view.zip(gs.view)
        .map { case (s, t) => f2(s, t) })
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

  def mapAnnotations(f: (Variant, AnnotationData) => AnnotationData): VariantSampleMatrix[T] =
    copy[T](rdd = rdd.map { case (v, va, gs) => (v, f(v, va), gs) })

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

  // FIXME see if we can remove broadcasts elsewhere in the code
  def filterSamples(p: (Int, AnnotationData) => Boolean): VariantSampleMatrix[T] = {
    val mask = localSamples.zip(metadata.sampleAnnotations).map { case (s, sa) => p(s, sa) }
    val maskBc = sparkContext.broadcast(mask)
    val localtct = tct
    copy[T](localSamples = localSamples.zipWithIndex
      .filter { case (s, i) => mask(i) }
      .map(_._1),
      rdd = rdd.map { case (v, va, gs) =>
        (v, va, maskBc.value.iterator.zip(gs.iterator)
          .filter(_._1)
          .map(_._2)
          .toArray[T](localtct): Iterable[T])
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

    copy(rdd = rdd
      .map { case (v, va, gs) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        val zeroValue = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))

        (v, mapOp(va, gs.zipWithIndex.foldLeft(zeroValue) { case (acc, (g, i)) =>
          seqOp(acc, v, localSamplesBc.value(i), g)
        }), gs)
      })
  }

  def addVariantMapSignatures(mapName: String, map: Map[String, AnnotationSignature]): VariantSampleMatrix[T] = {
    copy(metadata = metadata.copy(variantAnnotationSignatures =
      metadata.variantAnnotationSignatures.addMap(mapName, map)))
  }

  def addVariantValSignature(name: String, sig: AnnotationSignature): VariantSampleMatrix[T] = {
    copy(metadata = metadata.copy(variantAnnotationSignatures =
      metadata.variantAnnotationSignatures.addVal(name, sig)))
  }

  def addSampleMapSignatures(mapName: String, map: Map[String, AnnotationSignature]): VariantSampleMatrix[T] = {
    copy(metadata = metadata.copy(sampleAnnotationSignatures =
      metadata.sampleAnnotationSignatures.addMap(mapName, map)))
  }

  def addSampleValSignature(name: String, sig: AnnotationSignature): VariantSampleMatrix[T] = {
    copy(metadata = metadata.copy(sampleAnnotationSignatures =
      metadata.sampleAnnotationSignatures.addVal(name, sig)))
  }

  private def join[S,T2,S2,T3,S3,T4,S4,A1,A2](other:VariantSampleMatrix[S],
                                        sampleJoinFunction:(VariantSampleMatrix[T],VariantSampleMatrix[S]) => (VariantSampleMatrix[T2],VariantSampleMatrix[S2]),
                                        variantJoinFunction:(VariantSampleMatrix[T2],VariantSampleMatrix[S2]) => RDD[(Variant,((A1,T3),(A2,S3)))],
                                        annotationJoinFunction:(A1,A2) => AnnotationData,
                                        genotypeJoinFunction:(T3,S3,Int) => Iterable[(T4,S4)])
                                       (implicit t2ct:ClassTag[T2], t3ct:ClassTag[T3], t4ct:ClassTag[T4],
                                        sct:ClassTag[S], s2ct:ClassTag[S2],s3ct:ClassTag[S3], s4ct:ClassTag[S4],
                                             ttt: TypeTag[T], t2tt: TypeTag[T2], t3tt: TypeTag[T3], t4tt: TypeTag[T4],
                                             stt: TypeTag[S], s2tt: TypeTag[S2], s3tt: TypeTag[S3], s4tt: TypeTag[S4]): VariantSampleMatrix[(T4,S4)] = {

    val (vsm1Prime: VariantSampleMatrix[T2], vsm2Prime: VariantSampleMatrix[S2]) = sampleJoinFunction(this,other)

    require(vsm1Prime.sampleIds.equals(vsm2Prime.sampleIds) && vsm1Prime.localSamples.sameElements(vsm2Prime.localSamples))

    val nSamplesLocal = vsm1Prime.nLocalSamples

    val mergedRdd:RDD[(Variant,((A1,T3),(A2,S3)))] = variantJoinFunction(vsm1Prime,vsm2Prime)

    new VariantSampleMatrix[(T4,S4)](new VariantMetadata(metadata.filters, vsm1Prime.sampleIds, metadata.sampleAnnotations,
      metadata.sampleAnnotationSignatures, metadata.variantAnnotationSignatures),
      vsm1Prime.localSamples,
      mergedRdd.map{case (v, ((va1,gs1),(va2,gs2))) => (v,annotationJoinFunction(va1,va2), genotypeJoinFunction(gs1,gs2,nSamplesLocal))}
    )
  }

  def joinInnerInner[S](other:VariantSampleMatrix[S])(implicit stt: TypeTag[S], sct: ClassTag[S]):VariantSampleMatrix[(T,S)] = {
    join(other, (a: VariantSampleMatrix[T], b: VariantSampleMatrix[S]) => sampleInnerJoin[T,S](a,b),
      variantInnerJoin[T,S],
      annotationInnerJoin[AnnotationData,AnnotationData],
      (a: Iterable[T], b:  Iterable[S], n: Int) => genotypeInnerInnerJoin[T,S](a, b, n))
  }

  def joinInnerLeft[S](other:VariantSampleMatrix[S])(implicit stt: TypeTag[S], sct: ClassTag[S]):VariantSampleMatrix[(T,Option[S])] = {
    val tctLocal = tct
    val tttLocal = ttt
    val sctLocal = sct
    val sttLocal = stt

    join(other, (a: VariantSampleMatrix[T], b: VariantSampleMatrix[S]) => sampleInnerJoin[T,S](a,b)(tttLocal,tctLocal,sttLocal,sctLocal),
      variantLeftJoin[T,S],
      annotationLeftJoin[AnnotationData,Option[AnnotationData]],
      (a: Iterable[T], b: Option[Iterable[S]], n: Int) => genotypeInnerLeftJoin[T,S](a,b,n)(tctLocal,sctLocal))
  }

  def joinInnerRight[S](other:VariantSampleMatrix[S])(implicit stt: TypeTag[S], sct: ClassTag[S]):VariantSampleMatrix[(Option[T],S)] = {
    val tctLocal = tct
    val tttLocal = ttt
    val sctLocal = sct
    val sttLocal = stt

    join(other, (a: VariantSampleMatrix[T], b: VariantSampleMatrix[S]) => sampleInnerJoin[T,S](a,b)(tttLocal,tctLocal,sttLocal,sctLocal),
      variantRightJoin[T,S],
      annotationRightJoin[Option[AnnotationData],AnnotationData],
      (a: Option[Iterable[T]], b:  Iterable[S], n: Int) => genotypeInnerRightJoin[T,S](a,b,n)(tctLocal,sctLocal))
  }

  def joinInnerOuter[S](other:VariantSampleMatrix[S])(implicit stt: TypeTag[S], sct: ClassTag[S]):VariantSampleMatrix[(Option[T],Option[S])] = {
    val tctLocal = tct
    val tttLocal = ttt
    val sctLocal = sct
    val sttLocal = stt

    join(other, (a: VariantSampleMatrix[T], b: VariantSampleMatrix[S]) => sampleInnerJoin[T,S](a,b)(tttLocal,tctLocal,sttLocal,sctLocal),
      variantOuterJoin[T,S],
      annotationOuterJoin[Option[AnnotationData],Option[AnnotationData]],
      (a: Option[Iterable[T]], b:  Option[Iterable[S]], n: Int) => genotypeInnerOuterJoin[T,S](a,b,n)(tctLocal,sctLocal))
  }

  def joinLeftInner[S](other:VariantSampleMatrix[S])(implicit stt: TypeTag[S], sct: ClassTag[S]):VariantSampleMatrix[(T,Option[S])] = {
    val tctLocal = tct
    val tttLocal = ttt
    val sctLocal = sct
    val sttLocal = stt

    join(other, (a: VariantSampleMatrix[T], b: VariantSampleMatrix[S]) => sampleLeftJoin[T,S](a,b)(tttLocal,tctLocal,sttLocal,sctLocal),
      variantInnerJoin[T,Option[S]],
      annotationInnerJoin[AnnotationData,AnnotationData],
      (a: Iterable[T], b:  Iterable[Option[S]], n: Int) => genotypeLeftInnerJoin[T,S](a,b,n)(tctLocal,sctLocal))
  }

  def joinLeftLeft[S](other:VariantSampleMatrix[S])(implicit stt: TypeTag[S], sct: ClassTag[S]):VariantSampleMatrix[(T,Option[S])] = {
    val tctLocal = tct
    val tttLocal = ttt
    val sctLocal = sct
    val sttLocal = stt

    join(other, (a: VariantSampleMatrix[T], b: VariantSampleMatrix[S]) => sampleLeftJoin[T,S](a,b)(tttLocal,tctLocal,sttLocal,sctLocal),
      variantLeftJoin[T,Option[S]],
      annotationLeftJoin[AnnotationData,Option[AnnotationData]],
      (a: Iterable[T], b: Option[Iterable[Option[S]]], n: Int) => genotypeLeftLeftJoin[T,S](a,b,n)(tctLocal,sctLocal))
  }

  def joinLeftRight[S](other:VariantSampleMatrix[S])(implicit stt: TypeTag[S], sct: ClassTag[S]):VariantSampleMatrix[(Option[T],Option[S])] = {
    val tctLocal = tct
    val tttLocal = ttt
    val sctLocal = sct
    val sttLocal = stt

    join(other, (a: VariantSampleMatrix[T], b: VariantSampleMatrix[S]) => sampleLeftJoin[T,S](a,b)(tttLocal,tctLocal,sttLocal,sctLocal),
      variantRightJoin[T,Option[S]],
      annotationRightJoin[Option[AnnotationData],AnnotationData],
      (a: Option[Iterable[T]], b:  Iterable[Option[S]], n: Int) => genotypeLeftRightJoin[T,S](a,b,n)(tctLocal,sctLocal))
  }

  def joinLeftOuter[S](other:VariantSampleMatrix[S])(implicit stt: TypeTag[S], sct: ClassTag[S]):VariantSampleMatrix[(Option[T],Option[S])] = {
    val tctLocal = tct
    val tttLocal = ttt
    val sctLocal = sct
    val sttLocal = stt

    join(other, (a: VariantSampleMatrix[T], b: VariantSampleMatrix[S]) => sampleLeftJoin[T,S](a,b)(tttLocal,tctLocal,sttLocal,sctLocal),
      variantOuterJoin[T,Option[S]],
      annotationOuterJoin[Option[AnnotationData],Option[AnnotationData]],
      (a: Option[Iterable[T]], b: Option[Iterable[Option[S]]], n: Int) => genotypeLeftOuterJoin[T,S](a,b,n)(tctLocal,sctLocal))
  }

  def joinRightInner[S](other:VariantSampleMatrix[S])(implicit stt: TypeTag[S], sct: ClassTag[S]):VariantSampleMatrix[(Option[T],S)] = {
    val tctLocal = tct
    val tttLocal = ttt
    val sctLocal = sct
    val sttLocal = stt

    join(other, (a: VariantSampleMatrix[T], b: VariantSampleMatrix[S]) => sampleRightJoin[T,S](a,b)(tttLocal,tctLocal,sttLocal,sctLocal),
      variantInnerJoin[Option[T],S],
      annotationInnerJoin[AnnotationData,AnnotationData],
      (a: Iterable[Option[T]], b:  Iterable[S], n: Int) => genotypeRightInnerJoin[T,S](a,b,n)(tctLocal,sctLocal))
  }

  def joinRightLeft[S](other:VariantSampleMatrix[S])(implicit stt: TypeTag[S], sct: ClassTag[S]):VariantSampleMatrix[(Option[T],Option[S])] = {
    val tctLocal = tct
    val tttLocal = ttt
    val sctLocal = sct
    val sttLocal = stt

    join(other, (a: VariantSampleMatrix[T], b: VariantSampleMatrix[S]) => sampleRightJoin[T,S](a,b)(tttLocal,tctLocal,sttLocal,sctLocal),
      variantLeftJoin[Option[T],S],
      annotationLeftJoin[AnnotationData,Option[AnnotationData]],
      (a: Iterable[Option[T]], b:  Option[Iterable[S]], n: Int) => genotypeRightLeftJoin[T,S](a,b,n)(tctLocal,sctLocal))
  }

  def joinRightRight[S](other:VariantSampleMatrix[S])(implicit stt: TypeTag[S], sct: ClassTag[S]):VariantSampleMatrix[(Option[T],S)] = {
    val tctLocal = tct
    val tttLocal = ttt
    val sctLocal = sct
    val sttLocal = stt

    join(other, (a: VariantSampleMatrix[T], b: VariantSampleMatrix[S]) => sampleRightJoin[T,S](a,b)(tttLocal,tctLocal,sttLocal,sctLocal),
      variantRightJoin[Option[T],S],
      annotationRightJoin[Option[AnnotationData],AnnotationData],
      (a: Option[Iterable[Option[T]]], b: Iterable[S], n: Int) => genotypeRightRightJoin[T,S](a,b,n)(tctLocal,sctLocal))
  }

  def joinRightOuter[S](other:VariantSampleMatrix[S])(implicit stt: TypeTag[S], sct: ClassTag[S]):VariantSampleMatrix[(Option[T],Option[S])] = {
    val tctLocal = tct
    val tttLocal = ttt
    val sctLocal = sct
    val sttLocal = stt

    join(other, (a: VariantSampleMatrix[T], b: VariantSampleMatrix[S]) => sampleRightJoin[T,S](a,b)(tttLocal,tctLocal,sttLocal,sctLocal),
      variantOuterJoin[Option[T],S],
      annotationOuterJoin[Option[AnnotationData],Option[AnnotationData]],
      (a: Option[Iterable[Option[T]]], b: Option[Iterable[S]], n: Int) => genotypeRightOuterJoin[T,S](a,b,n)(tctLocal,sctLocal))
  }

  def joinOuterInner[S](other:VariantSampleMatrix[S])(implicit stt: TypeTag[S], sct: ClassTag[S]):VariantSampleMatrix[(Option[T],Option[S])] = {
    val tctLocal = tct
    val tttLocal = ttt
    val sctLocal = sct
    val sttLocal = stt

    join(other, (a: VariantSampleMatrix[T], b: VariantSampleMatrix[S]) => sampleOuterJoin[T,S](a,b)(tttLocal,tctLocal,sttLocal,sctLocal),
      variantInnerJoin[Option[T],Option[S]],
      annotationInnerJoin[AnnotationData,AnnotationData],
      (a: Iterable[Option[T]], b: Iterable[Option[S]], n: Int) => genotypeOuterInnerJoin[T,S](a,b,n)(tctLocal,sctLocal))
  }

  def joinOuterLeft[S](other:VariantSampleMatrix[S])(implicit stt: TypeTag[S], sct: ClassTag[S]):VariantSampleMatrix[(Option[T],Option[S])] = {
    val tctLocal = tct
    val tttLocal = ttt
    val sctLocal = sct
    val sttLocal = stt

    join(other, (a: VariantSampleMatrix[T], b: VariantSampleMatrix[S]) => sampleOuterJoin[T,S](a,b)(tttLocal,tctLocal,sttLocal,sctLocal),
      variantLeftJoin[Option[T],Option[S]],
      annotationLeftJoin[AnnotationData,Option[AnnotationData]],
      (a: Iterable[Option[T]], b: Option[Iterable[Option[S]]], n: Int) => genotypeOuterLeftJoin[T,S](a,b,n)(tctLocal,sctLocal))
  }

  def joinOuterRight[S](other:VariantSampleMatrix[S])(implicit stt: TypeTag[S], sct: ClassTag[S]):VariantSampleMatrix[(Option[T],Option[S])] = {
    val tctLocal = tct
    val tttLocal = ttt
    val sctLocal = sct
    val sttLocal = stt

    join(other, (a: VariantSampleMatrix[T], b: VariantSampleMatrix[S]) => sampleOuterJoin[T,S](a,b)(tttLocal,tctLocal,sttLocal,sctLocal),
      variantRightJoin[Option[T],Option[S]],
      annotationRightJoin[Option[AnnotationData],AnnotationData],
      (a: Option[Iterable[Option[T]]], b:  Iterable[Option[S]], n: Int) => genotypeOuterRightJoin[T,S](a,b,n)(tctLocal,sctLocal))
  }

  def joinOuterOuter[S](other:VariantSampleMatrix[S])(implicit stt: TypeTag[S], sct: ClassTag[S]):VariantSampleMatrix[(Option[T],Option[S])] = {
    val tctLocal = tct
    val tttLocal = ttt
    val sctLocal = sct
    val sttLocal = stt

    join(other, (a: VariantSampleMatrix[T], b: VariantSampleMatrix[S]) => sampleOuterJoin[T,S](a,b)(tttLocal,tctLocal,sttLocal,sctLocal),
      variantOuterJoin[Option[T],Option[S]],
      annotationOuterJoin[Option[AnnotationData],Option[AnnotationData]],
      (a: Option[Iterable[Option[T]]], b: Option[Iterable[Option[S]]], n: Int) => genotypeOuterOuterJoin[T,S](a,b,n)(tctLocal,sctLocal))
  }
}

// FIXME AnyVal Scala 2.11
class RichVDS(vds: VariantDataset) {

  def write(sqlContext: SQLContext, dirname: String, compress: Boolean = true) {
    import sqlContext.implicits._

    require(dirname.endsWith(".vds"))

    val hConf = vds.sparkContext.hadoopConfiguration
    hadoopMkdir(dirname, hConf)
    writeObjectFile(dirname + "/metadata.ser", hConf) { s =>
      s.writeObject(vds.localSamples)
      s.writeObject(vds.metadata)
    }

    // rdd.toDF().write.parquet(dirname + "/rdd.parquet")
    // FIXME write annotations: va
    vds.rdd
      .map { case (v, va, gs) => (v, va.toArrays, gs.toGenotypeStream(v, compress)) }
      .toDF()
      .saveAsParquetFile(dirname + "/rdd.parquet")
  }
}
