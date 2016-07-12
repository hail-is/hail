package org.broadinstitute.hail.variant

import java.nio.ByteBuffer

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkContext, SparkEnv}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.check.Gen
import org.broadinstitute.hail.expr.{TStruct, _}
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.vcf.BufferedLineIterator
import org.kududb.spark.kudu.{KuduContext, _}

import scala.io.Source
import scala.language.implicitConversions
import scala.reflect.ClassTag

object VariantSampleMatrix {
  final val fileVersion: Int = 4

  def apply[T](metadata: VariantMetadata,
    rdd: RDD[(Variant, Annotation, Iterable[T])])(implicit tct: ClassTag[T]): VariantSampleMatrix[T] = {
    new VariantSampleMatrix(metadata, rdd)
  }

  private def readMetadata(sqlContext: SQLContext, dirname: String, skipGenotypes: Boolean = false,
    requireParquetSuccess: Boolean = true): VariantMetadata = {
    if (!dirname.endsWith(".vds") && !dirname.endsWith(".vds/"))
      fatal(s"input path ending in `.vds' required, found `$dirname'")

    val hConf = sqlContext.sparkContext.hadoopConfiguration

    if (!hadoopExists(hConf, dirname))
      fatal(s"no VDS found at `$dirname'")

    val metadataFile = dirname + "/metadata.json.gz"
    val pqtSuccess = dirname + "/rdd.parquet/_SUCCESS"

    if (!hadoopExists(hConf, metadataFile))
      fatal("corrupt or old VDS: no `metadata.json.gz' file found.\n  Recreate VDS with current version of Hail.")

    val json = try {
      readFile(metadataFile, hConf)(
        in => parse(Source.fromInputStream(in).mkString)
      )
    } catch {
      case e: Throwable => fatal(s"corrupt VDS: invalid metadata file: ${e.getMessage}.\n  " +
        s"Recreate VDS with current version of Hail")
    }

    val fields = json match {
      case jo: JObject => jo.obj.toMap
      case _ => fatal("corrupt VDS: invalid metadata json.\n  Recreate VDS with current version of Hail.")
    }

    def getAndCastJSON[T <: JValue](fname: String, expected: String)(implicit tct: ClassTag[T]): T =
      fields.get(fname) match {
        case Some(t: T) => t
        case Some(other) =>
          fatal(s"corrupt VDS: invalid json value.  Expected `$expected' in field `$fname', " +
            s"but got `${other.getClass.getName}'\n  " +
          s"Recreate VDS with current version of Hail")
        case None => fatal(s"Invalid metadata file.  Missing field `$fname'")
      }

    val wasSplit = getAndCastJSON[JBool]("was split", "JBool").value
    val version = getAndCastJSON[JInt]("version", "JBool").num

    val saSignature = Parser.parseType(getAndCastJSON[JString]("sample annotation schema", "JString").s)
    val vaSignature = Parser.parseType(getAndCastJSON[JString]("variant annotation schema", "JString").s)
    val globalSignature = Parser.parseType(getAndCastJSON[JString]("global annotation schema", "JString").s)

    val sampleInfoSchema = TStruct(("id", TString), ("annotation", saSignature))
    val sampleInfo = getAndCastJSON[JArray]("sample metadata", "JArray")
      .arr
      .map {
        case JObject(List(("id", JString(id)), ("annotation", jv: JValue))) =>
          (id, Annotation.fromJson(jv, saSignature, "sample-info"))
        case other =>
          fatal(s"invalid sample metadata: expected JObject with fields `id' and `annotation', but got `$other'")
      }
      .toArray

    val globalAnnotation = Annotation.fromJson(getAndCastJSON[JValue]("global annotation", "_"), globalSignature, "global")

    if (!hadoopExists(hConf, pqtSuccess))
      fatal("corrupt VDS: no parquet success indicator, meaning a problem occurred during write.  Recreate VDS.")

    val ids = sampleInfo.map(_._1)
    val annotations = sampleInfo.map(_._2)

    val metadata = VariantMetadata(ids, annotations, globalAnnotation,
      saSignature, vaSignature, globalSignature, wasSplit)
  }

  def read(sqlContext: SQLContext, dirname: String, skipGenotypes: Boolean = false): VariantDataset = {

    val metadata = readMetadata(sqlContext, dirname, skipGenotypes)
    val vaSignature = metadata.vaSignature

    val df = sqlContext.read.parquet(dirname + "/rdd.parquet")

    val vaRequiresConversion = vaSignature.requiresConversion

    if (skipGenotypes)
      new VariantSampleMatrix[Genotype](
        metadata.copy(sampleIds = IndexedSeq.empty[String],
          sampleAnnotations = IndexedSeq.empty[Annotation]),
        df.select("variant", "annotations")
          .map(row => (row.getVariant(0),
            if (vaRequiresConversion) vaSignature.makeSparkReadable(row.get(1)) else row.get(1),
            Iterable.empty[Genotype])))
    else
      new VariantSampleMatrix(
        metadata,
        df.rdd.map { row =>
          val v = row.getVariant(0)

          (v,
            if (vaRequiresConversion) vaSignature.makeSparkReadable(row.get(1)) else row.get(1),
            row.getGenotypeStream(v, 2))
        })
  }

  def readKudu(sqlContext: SQLContext, dirname: String, tableName: String,
    master: String): VariantDataset = {

    val metadata = readMetadata(sqlContext, dirname, skipGenotypes = false,
      requireParquetSuccess = false)

    val df = sqlContext.read.options(
      Map("kudu.table" -> tableName, "kudu.master" -> master)).kudu
    val rdd: RDD[(Variant, Annotation, Iterable[Genotype])] = df.rdd.mapPartitions(iter => {
      val ser = SparkEnv.get.serializer.newInstance()
      iter.map(r => {
        val variant = Variant(r.getAs[String]("contig"), r.getAs[Int]("start"),
          r.getAs[String]("ref"), r.getAs[String]("alt"))
        val annotations = Annotation.empty // TODO: deserialize annotations
        val bb = r.getAs[ByteBuffer]("genotypes")
        val b = new Array[Byte](bb.remaining)
        bb.get(b)
        val genotypeStream = GenotypeStream(variant, Some(r.getAs[Int]("genotypes_byte_len")), b)
        (variant, (annotations, genotypeStream))
      })
    }).spanByKey().map(kv => {
      // combine variant rows with different sample groups (no shuffle)
      val variant = kv._1
      val annotations = kv._2.head._1 // just use first annotation
      val genotypes = kv._2.flatMap(_._2) // combine genotype streams
      (variant, annotations, genotypes)
    })
    new VariantSampleMatrix[Genotype](metadata, rdd)
  }

  def gen[T](sc: SparkContext,
    gen: VSMSubgen[T])(implicit tct: ClassTag[T]): Gen[VariantSampleMatrix[T]] =
    gen.gen(sc)
}

case class VSMSubgen[T](
  sampleIdGen: Gen[IndexedSeq[String]],
  saSigGen: Gen[Type],
  vaSigGen: Gen[Type],
  globalSigGen: Gen[Type],
  saGen: (Int, Type) => Gen[IndexedSeq[Annotation]],
  vaGen: (Type) => Gen[Annotation],
  globalGen: (Type) => Gen[Annotation],
  vGen: Gen[Variant],
  tGen: (Variant) => Gen[T]) {

  def gen(sc: SparkContext)(implicit tct: ClassTag[T]): Gen[VariantSampleMatrix[T]] =
    for (vaSig <- vaSigGen;
      saSig <- saSigGen;
      globalSig <- globalSigGen;
      sampleIds <- sampleIdGen;
      global <- globalGen(globalSig);
      saValues <- saGen(sampleIds.length, saSig);
      rows <- Gen.distinctBuildableOf[Seq[(Variant, Annotation, Iterable[T])], (Variant, Annotation, Iterable[T])](
        for (v <- vGen;
          va <- vaGen(vaSig);
          ts <- Gen.buildableOfN[Iterable[T], T](sampleIds.length, tGen(v)))
          yield (v, va, ts)))
      yield VariantSampleMatrix[T](VariantMetadata(sampleIds, saValues, global, saSig, vaSig, globalSig), sc.parallelize(rows))
}

object VSMSubgen {
  val random = VSMSubgen[Genotype](
    sampleIdGen = Gen.distinctBuildableOf[IndexedSeq[String], String](Gen.identifier),
    saSigGen = Type.genArb,
    vaSigGen = Type.genArb,
    globalSigGen = Type.genArb,
    saGen = (nSamples: Int, t: Type) =>
      Gen.sequence[IndexedSeq[Annotation], Annotation](IndexedSeq.fill[Gen[Annotation]](nSamples)(t.genValue)),
    vaGen = (t: Type) => t.genValue,
    globalGen = (t: Type) => t.genValue,
    vGen = Variant.gen,
    tGen = Genotype.gen)

  val realistic = random.copy(
    tGen = Genotype.genRealistic)
}

class VariantSampleMatrix[T](val metadata: VariantMetadata,
  val rdd: RDD[(Variant, Annotation, Iterable[T])])
  (implicit tct: ClassTag[T]) {

  def sampleIds: IndexedSeq[String] = metadata.sampleIds

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

  def copy[U](rdd: RDD[(Variant, Annotation, Iterable[U])] = rdd,
    sampleIds: IndexedSeq[String] = sampleIds,
    sampleAnnotations: IndexedSeq[Annotation] = sampleAnnotations,
    globalAnnotation: Annotation = globalAnnotation,
    saSignature: Type = saSignature,
    vaSignature: Type = vaSignature,
    globalSignature: Type = globalSignature,
    wasSplit: Boolean = wasSplit)
    (implicit tct: ClassTag[U]): VariantSampleMatrix[U] =
    new VariantSampleMatrix[U](
      VariantMetadata(sampleIds, sampleAnnotations, globalAnnotation,
        saSignature, vaSignature, globalSignature, wasSplit), rdd)

  def sparkContext: SparkContext = rdd.sparkContext

  def cache(): VariantSampleMatrix[T] = copy[T](rdd = rdd.cache())

  def repartition(nPartitions: Int) = copy[T](rdd = rdd.repartition(nPartitions)(null))

  def nPartitions: Int = rdd.partitions.length

  def variants: RDD[Variant] = rdd.map(_._1)

  def variantsAndAnnotations: RDD[(Variant, Annotation)] = rdd.map { case (v, va, gs) => (v, va) }

  def nVariants: Long = variants.count()

  def expand(): RDD[(Variant, String, T)] =
    mapWithKeys[(Variant, String, T)]((v, s, g) => (v, s, g))

  def expandWithAll(): RDD[(Variant, Annotation, String, Annotation, T)] =
    mapWithAll[(Variant, Annotation, String, Annotation, T)]((v, va, s, sa, g) => (v, va, s, sa, g))

  def sampleVariants(fraction: Double): VariantSampleMatrix[T] =
    copy(rdd = rdd.sample(withReplacement = false, fraction, 1))

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
    copy(rdd = rdd.map { case (v, va, gs) =>
      (v, va, localSampleIdsBc.value.lazyMapWith2[Annotation, T, U](localSampleAnnotationsBc.value, gs, {
        case (s, sa, g) => f(v, va, s, sa, g)
      }))
    })
  }

  def mapValuesWithPartialApplication[U](f: (Variant, Annotation) => ((String, Annotation) => U))
    (implicit uct: ClassTag[U]): VariantSampleMatrix[U] = {
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    copy(rdd =
      rdd.mapPartitions[(Variant, Annotation, Iterable[U])] { it: Iterator[(Variant, Annotation, Iterable[T])] =>
        it.map { case (v, va, gs) =>
          val f2 = f(v, va)
          (v, va, localSampleIdsBc.value.lazyMapWith2[Annotation, T, U](localSampleAnnotationsBc.value, gs, {
            case (s, sa, g) => f2(s, sa)
          }))
        }
      })
  }

  def map[U](f: T => U)(implicit uct: ClassTag[U]): RDD[U] =
    mapWithKeys((v, s, g) => f(g))

  def mapWithKeys[U](f: (Variant, String, T) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc

    rdd
      .flatMap { case (v, va, gs) =>
        localSampleIdsBc.value.lazyMapWith[T, U](gs,
          (s, g) => f(v, s, g))
      }
  }

  def mapWithAll[U](f: (Variant, Annotation, String, Annotation, T) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    rdd
      .flatMap { case (v, va, gs) =>
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
      f(it.flatMap { case (v, va, gs) =>
        localSampleIdsBc.value.lazyMapWith2[Annotation, T, (Variant, Annotation, String, Annotation, T)](
          localSampleAnnotationsBc.value, gs, { case (s, sa, g) => (v, va, s, sa, g) })
      })
    }
  }

  def mapAnnotations(f: (Variant, Annotation, Iterable[T]) => Annotation): VariantSampleMatrix[T] =
    copy[T](rdd = rdd.map { case (v, va, gs) => (v, f(v, va, gs), gs) })

  def flatMap[U](f: T => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] =
    flatMapWithKeys((v, s, g) => f(g))

  def flatMapWithKeys[U](f: (Variant, String, T) => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc

    rdd
      .flatMap { case (v, va, gs) => localSampleIdsBc.value.lazyFlatMapWith(gs,
        (s: String, g: T) => f(v, s, g))
      }
  }

  def filterVariants(p: (Variant, Annotation, Iterable[T]) => Boolean): VariantSampleMatrix[T] =
    copy(rdd = rdd.filter { case (v, va, gs) => p(v, va, gs) })

  def filterVariants(ilist: IntervalList): VariantSampleMatrix[T] =
    filterVariants((v, va, gs) => ilist.contains(v.contig, v.start))

  def dropSamples(): VariantSampleMatrix[T] =
    copy(sampleIds = IndexedSeq.empty[String],
      sampleAnnotations = IndexedSeq.empty[Annotation],
      rdd = rdd.map { case (v, va, gs) =>
        (v, va, Iterable.empty[T])
      })

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
      rdd = rdd.map { case (v, va, gs) =>
        (v, va, gs.lazyFilterWith(maskBc.value, (g: T, m: Boolean) => m))
      })
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
      .mapPartitions { (it: Iterator[(Variant, Annotation, Iterable[T])]) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        def copyZeroValue() = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))
        val arrayZeroValue = Array.fill[U](localSampleIdsBc.value.length)(copyZeroValue())

        localSampleIdsBc.value.iterator
          .zip(it.foldLeft(arrayZeroValue) { case (acc, (v, va, gs)) =>
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
      .mapPartitions { (it: Iterator[(Variant, Annotation, Iterable[T])]) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        it.map { case (v, va, gs) =>
          val zeroValue = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))
          (v, gs.iterator.zipWithIndex.map { case (g, i) => (localSampleIdsBc.value(i), localSampleAnnotationsBc.value(i), g) }
            .foldLeft(zeroValue) { case (acc, (s, sa, g)) =>
              seqOp(acc, v, va, s, sa, g)
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

  def foldBySample(zeroValue: T)(combOp: (T, T) => T): RDD[(String, T)] = {

    val localtct = tct

    val serializer = SparkEnv.get.serializer.newInstance()
    val zeroBuffer = serializer.serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    val localSampleIdsBc = sampleIdsBc

    rdd
      .mapPartitions { (it: Iterator[(Variant, Annotation, Iterable[T])]) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        def copyZeroValue() = serializer.deserialize[T](ByteBuffer.wrap(zeroArray))(localtct)
        val arrayZeroValue = Array.fill[T](localSampleIdsBc.value.length)(copyZeroValue())
        localSampleIdsBc.value.iterator
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
    val metadataSame = metadata == that.metadata
    if (!metadataSame) {
      println(
        s"""Discordant Metadata:
            |  Sample IDs:          ${if (sampleIds == that.sampleIds) "PASS" else "FAIL"}
            |  Sample Annotations:  ${if (sampleAnnotations == that.sampleAnnotations) "PASS" else "FAIL"}
            |  VA Signature:        ${if (vaSignature == that.vaSignature) "PASS" else "FAIL"}
            |  SA Signature:        ${if (saSignature == that.saSignature) "PASS" else "FAIL"}
            |  Global Signature:    ${if (globalSignature == that.globalSignature) "PASS" else "FAIL"}
            |  Global Annotation:   ${if (globalAnnotation == that.globalAnnotation) "PASS" else "FAIL"}
            |  Was split:           ${if (wasSplit == that.wasSplit) "PASS" else "FAIL"}""".stripMargin)
      val sb = new StringBuilder
      metadata.vaSignature.compact(sb)
      println(sb.result())
      sb.clear()
      that.metadata.vaSignature.compact(sb)
      println(sb.result())
    }
    metadataSame &&
      rdd.map { case (v, va, gs) => (v, (va, gs)) }
        .fullOuterJoin(that.rdd.map { case (v, va, gs) => (v, (va, gs)) })
        .map {
          case (v, (Some((va1, it1)), Some((va2, it2)))) =>
            val annotationsSame = va1 == va2
            if (!annotationsSame)
              println(s"annotations $va1, $va2 were not the same")
            val genotypesSame = (it1, it2).zipped.forall { case (g1, g2) =>
              if (g1 != g2)
                println(s"genotypes $g1, $g2 were not the same")
              g1 == g2
            }
            annotationsSame && genotypesSame
          case (v, _) =>
            println(s"Found unmatched variant $v")
            false
        }.fold(true)(_ && _)
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
      rdd = rdd.map {
        case (v, va, gs) =>
          val serializer = SparkEnv.get.serializer.newInstance()
          val zeroValue = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))

          (v, mapOp(va, gs.iterator
            .zip(localSampleIdsBc.value.iterator
              .zip(localSampleAnnotationsBc.value.iterator)).foldLeft(zeroValue) {
            case (acc, (g, (s, sa))) =>
              seqOp(acc, v, va, s, sa, g)
          }), gs)
      })
  }

  def annotateInvervals(iList: IntervalList, signature: Type, path: List[String]): VariantSampleMatrix[T] = {
    val (newSignature, inserter) = insertVA(signature, path)
    val booleanSignature = newSignature == TBoolean
    val newRDD = if (booleanSignature)
      rdd.map { case (v, va, gs) => (v, inserter(va, Some(iList.contains(v.contig, v.start))), gs) }
    else
      rdd.map { case (v, va, gs) => (v, inserter(va, iList.query(v.contig, v.start)), gs) }
    copy(rdd = newRDD, vaSignature = newSignature)
  }

  def annotateVariants(otherRDD: RDD[(Variant, Annotation)], signature: Type,
    path: List[String]): VariantSampleMatrix[T] = {
    val (newSignature, inserter) = insertVA(signature, path)
    val newRDD = rdd.map { case (v, va, gs) => (v, (va, gs)) }
      .leftOuterJoinDistinct(otherRDD)
      .map { case (v, ((va, gs), annotation)) => (v, inserter(va, annotation), gs) }
    copy(rdd = newRDD, vaSignature = newSignature)
  }

  def annotateSamples(annotations: Map[String, Annotation], signature: Type,
    path: List[String]): VariantSampleMatrix[T] = {

    val (newSignature, inserter) = insertSA(signature, path)

    val newAnnotations = sampleIds.zipWithIndex.map { case (id, i) =>
      val sa = sampleAnnotations(i)
      inserter(sa, annotations.get(id))
    }

    copy(sampleAnnotations = newAnnotations, saSignature = newSignature)
  }

  def queryVA(code: String): (BaseType, Querier) = {

    val st = Map(Annotation.VARIANT_HEAD ->(0, vaSignature))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parse(code, ec)

    val f2: Annotation => Option[Any] = { annotation =>
      a(0) = annotation
      f()
    }

    (t, f2)
  }

  def querySA(code: String): (BaseType, Querier) = {

    val st = Map(Annotation.SAMPLE_HEAD ->(0, saSignature))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parse(code, ec)

    val f2: Annotation => Option[Any] = { annotation =>
      a(0) = annotation
      f()
    }

    (t, f2)
  }

  def queryGlobal(path: String): (BaseType, Option[Annotation]) = {
    val st = Map(Annotation.GLOBAL_HEAD ->(0, globalSignature))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parse(path, ec)

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
}

// FIXME AnyVal Scala 2.11
class RichVDS(vds: VariantDataset) {
  def makeSchema(): StructType =
    StructType(Array(
      StructField("variant", Variant.schema, nullable = false),
      StructField("annotations", vds.vaSignature.schema, nullable = false),
      StructField("gs", GenotypeStream.schema, nullable = false)
    ))

  private def writeMetadata(sqlContext: SQLContext, dirname: String, compress: Boolean = true) = {
    if (!dirname.endsWith(".vds") && !dirname.endsWith(".vds/"))
      fatal(s"output path ending in `.vds' required, found `$dirname'")

    val hConf = vds.sparkContext.hadoopConfiguration
    hadoopMkdir(dirname, hConf)

    val sb = new StringBuilder

    vds.saSignature.compact(sb, printAttrs = true)
    val saSchemaString = sb.result()

    sb.clear()
    vds.vaSignature.compact(sb, printAttrs = true)
    val vaSchemaString = sb.result()

    sb.clear()
    vds.globalSignature.compact(sb, printAttrs = true)
    val globalSchemaString = sb.result()

    val sampleInfoSchema = TStruct(("id", TString), ("annotation", vds.saSignature))
    val sampleInfoJson = JArray(
      vds.sampleIdsAndAnnotations
        .map { case (id, annotation) =>
          JObject(List(("id", JString(id)), ("annotation", vds.saSignature.makeJSON(annotation))))
        }
        .toList
    )

    val json = JObject(
      ("version", JInt(VariantSampleMatrix.fileVersion)),
      ("was split", JBool(vds.wasSplit)),
      ("sample annotation schema", JString(saSchemaString)),
      ("variant annotation schema", JString(vaSchemaString)),
      ("global annotation schema", JString(globalSchemaString)),
      ("sample metadata", sampleInfoJson),
      ("global annotation", vds.globalSignature.makeJSON(vds.globalAnnotation))
    )

    writeTextFile(dirname + "/metadata.json.gz", hConf) {
      dos => {
        dos.write(pretty(json))
      }
    }
  }

  def write(sqlContext: SQLContext, dirname: String, compress: Boolean = true) {
    writeMetadata(sqlContext, dirname, compress)

    val vaSignature = vds.vaSignature
    val vaRequiresConversion = vaSignature.requiresConversion

    val rowRDD = vds.rdd
      .map {
        case (v, va, gs) =>
          Row.fromSeq(Array(v.toRow,
            if (vaRequiresConversion) vaSignature.makeSparkWritable(va) else va,
            gs.toGenotypeStream(v, compress).toRow))
      }
    sqlContext.createDataFrame(rowRDD, makeSchema())
      .write.parquet(dirname + "/rdd.parquet")
    // .saveAsParquetFile(dirname + "/rdd.parquet")
  }

  def writeKudu(sqlContext: SQLContext, dirname: String, tableName: String,
    master: String, vcfSeqDict: String, rowsPerPartition: Int,
    sampleGroup: String, compress: Boolean = true, drop: Boolean = false) {

    writeMetadata(sqlContext, dirname, compress)

    val hConf = sqlContext.sparkContext.hadoopConfiguration
    val headerLines = readFile(vcfSeqDict, hConf) { s =>
      Source.fromInputStream(s)
        .getLines()
        .takeWhile { line => line(0) == '#' }
        .toArray
    }
    val codec = new htsjdk.variant.vcf.VCFCodec()
    val seqDict = codec.readHeader(new BufferedLineIterator(headerLines.iterator.buffered))
      .getHeaderValue
      .asInstanceOf[htsjdk.variant.vcf.VCFHeader]
      .getSequenceDictionary
    if (drop) {
      KuduUtils.dropTable(master, tableName)
      Thread.sleep(10 * 1000) // wait to avoid overwhelming Kudu service queue
    }
    KuduUtils.createTableIfNecessary(master, tableName, seqDict, rowsPerPartition)

    val rdd: RDD[(Variant, Annotation, Iterable[Genotype])] = vds.rdd
    val rowRdd = rdd.map {
      case (v, va, gs) =>
        val stream = gs.toGenotypeStream(v, compress)
        Row(
          KuduUtils.normalizeContig(v.contig),
          v.start,
          v.ref,
          v.allele(1), // TODO: remove biallelic assumption
          sampleGroup,
          v.contig,
          new Array[Byte](0), // TODO: serialize annotations
          stream.decompLenOption.get,
          stream.a
        )
    }
    val schema = StructType(List(
      StructField("contig_norm", StringType, nullable = false),
      StructField("start", IntegerType, nullable = false),
      StructField("ref", StringType, nullable = false),
      StructField("alt", StringType, nullable = false),
      StructField("sample_group", StringType, nullable = false),
      StructField("contig", StringType, nullable = false),
      StructField("annotations", BinaryType, nullable = true),
      StructField("genotypes_byte_len", IntegerType, nullable = true),
      StructField("genotypes", BinaryType, nullable = true)))
    val df = sqlContext.createDataFrame(rowRdd, schema)

    val kuduContext = new KuduContext(master)
    df.write
      .options(Map("kudu.master" -> master, "kudu.table" -> tableName))
      .mode("append").kudu

    println("Written to Kudu")
  }

  def eraseSplit: VariantDataset = {
    if (vds.wasSplit) {
      val (newSignatures1, f1) = vds.deleteVA("wasSplit")
      val vds1 = vds.copy(vaSignature = newSignatures1)
      val (newSignatures2, f2) = vds1.deleteVA("aIndex")
      vds1.copy(wasSplit = false,
        vaSignature = newSignatures2,
        rdd = vds1.rdd.map {
          case (v, va, gs) =>
            (v, f2(f1(va)), gs.lazyMap(g => g.copy(fakeRef = false)))
        })
    } else
      vds
  }

  def withGenotypeStream(compress: Boolean = false): VariantDataset =
    vds.copy(rdd = vds.rdd.map { case (v, va, gs) =>
      (v, va, gs.toGenotypeStream(v, compress = compress))
    })
}
