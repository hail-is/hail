package org.broadinstitute.hail.variant

import java.nio.ByteBuffer

import org.apache.hadoop
import org.apache.spark.rdd.{OrderedRDD, RDD}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{OrderedPartitioner, SparkContext, SparkEnv}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.check.Gen
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.vcf.BufferedLineIterator
import org.broadinstitute.hail.utils.{Interval, IntervalTree}
import org.json4s._
import org.json4s.jackson.JsonMethods
import org.kududb.spark.kudu.{KuduContext, _}

import scala.io.Source
import scala.language.implicitConversions
import scala.reflect.ClassTag

object VariantSampleMatrix {
  final val fileVersion: Int = 4

  def apply[T](metadata: VariantMetadata,
    rdd: OrderedRDD[Locus, Variant, (Annotation, Iterable[T])])(implicit tct: ClassTag[T]): VariantSampleMatrix[T] = {
    new VariantSampleMatrix(metadata, rdd)
  }

  private def readMetadata(hConf: hadoop.conf.Configuration, dirname: String,
    requireParquetSuccess: Boolean = true): VariantMetadata = {
    if (!dirname.endsWith(".vds") && !dirname.endsWith(".vds/"))
      fatal(s"input path ending in `.vds' required, found `$dirname'")

    if (!hadoopExists(hConf, dirname))
      fatal(s"no VDS found at `$dirname'")

    val metadataFile = dirname + "/metadata.json.gz"
    val pqtSuccess = dirname + "/rdd.parquet/_SUCCESS"

    if (!hadoopExists(hConf, pqtSuccess) && requireParquetSuccess)
      fatal(
        s"""corrupt VDS: no parquet success indicator
            |  Unexpected shutdown occurred during `write'
            |  Recreate VDS.""".stripMargin)

    if (!hadoopExists(hConf, metadataFile))
      fatal(
        s"""corrupt or outdated VDS: invalid metadata
            |  No `metadata.json.gz' file found in VDS directory
            |  Recreate VDS with current version of Hail.""".stripMargin)

    val json = try {
      readFile(metadataFile, hConf)(
        in => JsonMethods.parse(Source.fromInputStream(in).mkString)
      )
    } catch {
      case e: Throwable => fatal(s"corrupt VDS: invalid metadata file: ${ e.getMessage }.\n  " +
        s"Recreate VDS with current version of Hail")
    }

    val fields = json match {
      case jo: JObject => jo.obj.toMap
      case _ =>
        fatal(
          s"""corrupt VDS: invalid metadata value
              |  Recreate VDS with current version of Hail.""".stripMargin)
    }

    def getAndCastJSON[T <: JValue](fname: String)(implicit tct: ClassTag[T]): T =
      fields.get(fname) match {
        case Some(t: T) => t
        case Some(other) =>
          fatal(
            s"""corrupt VDS: invalid metadata
                |  Expected `${ tct.runtimeClass.getName }' in field `$fname', but got `${ other.getClass.getName }'
                |  Recreate VDS with current version of Hail.""".stripMargin)
        case None =>
          fatal(
            s"""corrupt VDS: invalid metadata
                |  Missing field `$fname'
                |  Recreate VDS with current version of Hail.""".stripMargin)
      }

    val version = getAndCastJSON[JInt]("version").num

    if (version != VariantSampleMatrix.fileVersion)
      fatal(
        s"""Invalid VDS: old version [$version]
            |  Recreate VDS with current version of Hail.
         """.stripMargin)

    val wasSplit = getAndCastJSON[JBool]("split").value
    val isDosage = fields.get("isDosage") match {
      case Some(t: JBool) => t.value
      case Some(other) => fatal(
        s"""corrupt VDS: invalid metadata
            |  Expected `JBool' in field `isDosage', but got `${ other.getClass.getName }'
            |  Recreate VDS with current version of Hail.""".stripMargin)
      case _ => false
    }

    val saSignature = Parser.parseType(getAndCastJSON[JString]("sample_annotation_schema").s)
    val vaSignature = Parser.parseType(getAndCastJSON[JString]("variant_annotation_schema").s)
    val globalSignature = Parser.parseType(getAndCastJSON[JString]("global_annotation_schema").s)

    val sampleInfoSchema = TStruct(("id", TString), ("annotation", saSignature))
    val sampleInfo = getAndCastJSON[JArray]("sample_annotations")
      .arr
      .map {
        case JObject(List(("id", JString(id)), ("annotation", jv: JValue))) =>
          (id, JSONAnnotationImpex.importAnnotation(jv, saSignature, "sample_annotations"))
        case other => fatal(
          s"""corrupt VDS: invalid metadata
              |  Invalid sample annotation metadata
              |  Recreate VDS with current version of Hail.""".stripMargin)
      }
      .toArray

    val globalAnnotation = JSONAnnotationImpex.importAnnotation(getAndCastJSON[JValue]("global_annotation"),
      globalSignature, "global")

    val ids = sampleInfo.map(_._1)
    val annotations = sampleInfo.map(_._2)

    VariantMetadata(ids, annotations, globalAnnotation,
      saSignature, vaSignature, globalSignature, wasSplit, isDosage)
  }

  def read(sqlContext: SQLContext, dirname: String, skipGenotypes: Boolean = false): VariantDataset = {

    val hConf = sqlContext.sparkContext.hadoopConfiguration

    val metadata = readMetadata(hConf, dirname, skipGenotypes)
    val vaSignature = metadata.vaSignature

    val df = sqlContext.readPartitioned.parquet(dirname + "/rdd.parquet")

    val vaRequiresConversion = SparkAnnotationImpex.requiresConversion(vaSignature)
    val isDosage = metadata.isDosage

    def readRDD(skipGenotypes: Boolean): RDD[(Variant, (Annotation, Iterable[Genotype]))] = {
      if (skipGenotypes)
        df.select("variant", "annotations")
          .map(row => (row.getVariant(0),
            (if (vaRequiresConversion) SparkAnnotationImpex.importAnnotation(row.get(1), vaSignature) else row.get(1),
              Iterable.empty[Genotype])))
      else
        df.rdd.map { row =>
          val v = row.getVariant(0)

          (v,
            (if (vaRequiresConversion) SparkAnnotationImpex.importAnnotation(row.get(1), vaSignature) else row.get(1),
              row.getGenotypeStream(v, 2, isDosage)))
        }
    }

    val orderedRDD = if (hadoopExists(hConf, dirname + "/partitioner")) {
      val partitioner = readObjectFile(dirname + "/partitioner", sqlContext.sparkContext.hadoopConfiguration) { in =>
        OrderedPartitioner.read[Locus, Variant](in, _.locus)
      }

      new OrderedRDD[Locus, Variant, (Annotation, Iterable[Genotype])](readRDD(skipGenotypes), partitioner)
    } else {
      warn(
        """No partition information found: VDS is old or corrupted and will experience poor performance.
          |  Please `read' and `write' this dataset to update it.""".stripMargin)
      if (skipGenotypes)
        readRDD(skipGenotypes = true).toOrderedRDD(_.locus)
      else
        readRDD(skipGenotypes = false).toOrderedRDD(
          _.locus,
          reducedRepresentation = Some(readRDD(skipGenotypes = true).map(_._1)))

    }

    new VariantSampleMatrix[Genotype](
      if (skipGenotypes) metadata.copy(sampleIds = IndexedSeq.empty[String],
        sampleAnnotations = IndexedSeq.empty[Annotation])
      else metadata,
      orderedRDD)
  }

  def kuduRowType(vaSignature: Type): Type = TStruct("variant" -> Variant.t,
    "annotations" -> vaSignature,
    "gs" -> GenotypeStream.t,
    "sample_group" -> TString)

  def readKudu(sqlContext: SQLContext, dirname: String, tableName: String,
    master: String): VariantDataset = {

    val metadata = readMetadata(sqlContext.sparkContext.hadoopConfiguration, dirname, requireParquetSuccess = false)
    val vaSignature = metadata.vaSignature
    val isDosage = metadata.isDosage

    val df = sqlContext.read.options(
      Map("kudu.table" -> tableName, "kudu.master" -> master)).kudu

    val rowType = kuduRowType(vaSignature)
    val schema: StructType = KuduAnnotationImpex.exportType(rowType).asInstanceOf[StructType]

    // Kudu key fields are always first, so we have to reorder the fields we get back
    // to be in the column order for the flattened schema *before* we unflatten
    val indices: Array[Int] = schema.fields.zipWithIndex.map { case (field, rowIdx) =>
      df.schema.fieldIndex(field.name)
    }

    val rdd: RDD[(Variant, (Annotation, Iterable[Genotype]))] = df.map { row =>
      val importedRow = KuduAnnotationImpex.importAnnotation(
        KuduAnnotationImpex.reorder(row, indices), rowType).asInstanceOf[Row]
      val v = importedRow.getVariant(0)
      (v,
        (importedRow.get(1),
          importedRow.getGenotypeStream(v, 2, metadata.isDosage)))
    }.spanByKey().map(kv => {
      // combine variant rows with different sample groups (no shuffle)
      val variant = kv._1
      val annotations = kv._2.head._1 // just use first annotation
      val genotypes = kv._2.flatMap(_._2) // combine genotype streams
      (variant, (annotations, genotypes))
    }

    )
    new VariantSampleMatrix[Genotype](metadata, rdd.toOrderedRDD(_.locus))
  }

  private def makeSchemaForKudu(vaSignature: Type): StructType =
    StructType(Array(
      StructField("variant", Variant.schema, nullable = false),
      StructField("annotations", vaSignature.schema, nullable = false),
      StructField("gs", GenotypeStream.schema, nullable = false),
      StructField("sample_group", StringType, nullable = false)
    ))

  def gen[T](sc: SparkContext,
    gen: VSMSubgen[T])(implicit tct: ClassTag[T]): Gen[VariantSampleMatrix[T]] =
    gen.gen(sc)
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
  isDosage: Boolean = false) {

  def gen(sc: SparkContext)(implicit tct: ClassTag[T]): Gen[VariantSampleMatrix[T]] =
    for (size <- Gen.size;
      subsizes <- Gen.partition(5).resize(size / 10);
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
        for (subsubsizes <- Gen.partition(3);
          v <- vGen.resize(subsubsizes(0));
          va <- vaGen(vaSig).resize(subsubsizes(1));
          ts <- Gen.buildableOfN[Iterable, T](nSamples, tGen(v.nAlleles)).resize(subsubsizes(2)))
          yield (v, (va, ts))).resize(l))
      yield {
        VariantSampleMatrix[T](VariantMetadata(sampleIds, saValues, global, saSig, vaSig, globalSig, wasSplit = false, isDosage = isDosage),
          sc.parallelize(rows, nPartitions).toOrderedRDD(_.locus))
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

  val realistic = random.copy(
    tGen = Genotype.genRealistic)

  val dosage = random.copy(
    tGen = Genotype.genDosage, isDosage = true)
}

class VariantSampleMatrix[T](val metadata: VariantMetadata,
  val rdd: OrderedRDD[Locus, Variant, (Annotation, Iterable[T])])(implicit tct: ClassTag[T]) {

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
    new VariantSampleMatrix[U](
      VariantMetadata(sampleIds, sampleAnnotations, globalAnnotation,
        saSignature, vaSignature, globalSignature, wasSplit, isDosage), rdd)

  def sparkContext: SparkContext = rdd.sparkContext

  def cache(): VariantSampleMatrix[T] = copy[T](rdd = rdd.cache())

  // FIXME make this faster by maintaining order
  def repartition(nPartitions: Int) = copy[T](rdd = rdd.repartition(nPartitions)(null).toOrderedRDD(_.locus))

  def nPartitions: Int = rdd.partitions.length

  def variants: RDD[Variant] = rdd.keys

  def variantsAndAnnotations: RDD[(Variant, Annotation)] = rdd.mapValuesWithKey { case (v, (va, gs)) => va }

  def nVariants: Long = variants.count()

  def expand(): RDD[(Variant, String, T)] =
    mapWithKeys[(Variant, String, T)]((v, s, g) => (v, s, g))

  def expandWithAll(): RDD[(Variant, Annotation, String, Annotation, T)] =
    mapWithAll[(Variant, Annotation, String, Annotation, T)]((v, va, s, sa, g) => (v, va, s, sa, g))

  def sampleVariants(fraction: Double): VariantSampleMatrix[T] =
    copy(rdd = rdd.sample(withReplacement = false, fraction, 1).toOrderedRDD(_.locus))

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
    }.toOrderedRDD(_.locus))
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
    copy[T](rdd = rdd.mapValuesWithKey { case (v, (va, gs)) => (f(v, va, gs), gs) }.toOrderedRDD(_.locus))

  def flatMap[U](f: T => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] =
    flatMapWithKeys((v, s, g) => f(g))

  def flatMapWithKeys[U](f: (Variant, String, T) => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc

    rdd
      .flatMap { case (v, (va, gs)) => localSampleIdsBc.value.lazyFlatMapWith(gs,
        (s: String, g: T) => f(v, s, g))
      }
  }

  def filterVariants(p: (Variant, Annotation, Iterable[T]) => Boolean): VariantSampleMatrix[T] =
    copy(rdd = rdd.filter { case (v, (va, gs)) => p(v, va, gs) }.toOrderedRDD(_.locus))

  def filterIntervals(iList: IntervalTree[Locus], keep: Boolean = true): VariantSampleMatrix[T] = {
    val iListBc = sparkContext.broadcast(iList)
    filterVariants { (v, va, gs) =>
      val inInterval = iListBc.value.contains(v.locus)
      if (keep)
        inInterval
      else
        !inInterval
    }
  }

  def dropSamples(): VariantSampleMatrix[T] =
    copy(sampleIds = IndexedSeq.empty[String],
      sampleAnnotations = IndexedSeq.empty[Annotation],
      rdd = rdd.mapValues { case (va, gs) => (va, Iterable.empty[T]) }
        .toOrderedRDD(_.locus))

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
      }.toOrderedRDD(_.locus))
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

  def same(that: VariantSampleMatrix[T]): Boolean = {
    val metadataSame = metadata == that.metadata
    if (!metadataSame) {
      println("metadata were not the same")
      if (vaSignature != that.vaSignature)
        println(
          s"""different va signature:
              |  left:  ${ vaSignature.toPrettyString(compact = true) }
              |  right: ${ that.vaSignature.toPrettyString(compact = true) }""".stripMargin)
      if (saSignature != that.saSignature)
        println(
          s"""different sa signature:
              |  left:  ${ saSignature.toPrettyString(compact = true) }
              |  right: ${ that.saSignature.toPrettyString(compact = true) }""".stripMargin)
      if (globalSignature != that.globalSignature)
        println(
          s"""different global signature:
              |  left:  ${ globalSignature.toPrettyString(compact = true) }
              |  right: ${ that.globalSignature.toPrettyString(compact = true) }""".stripMargin)
      if (sampleIds != that.sampleIds)
        println(
          s"""different sample ids:
              |  left:  $sampleIds
              |  right: ${ that.sampleIds }""".stripMargin)
      if (sampleAnnotations != that.sampleAnnotations)
        println(
          s"""different sample annotations:
              |  left:  $sampleAnnotations
              |  right: ${ that.sampleAnnotations }""".stripMargin)
      if (sampleIds != that.sampleIds)
        println(
          s"""different global annotation:
              |  left:  $globalAnnotation
              |  right: ${ that.globalAnnotation }""".stripMargin)
      if (wasSplit != that.wasSplit)
        println(
          s"""different was split:
              |  left:  $wasSplit
              |  right: ${ that.wasSplit }""".stripMargin)
    }
    var printed = false
    metadataSame &&
      rdd
        .fullOuterJoin(that.rdd)
        .forall {
          case (v, (Some((va1, it1)), Some((va2, it2)))) =>
            val annotationsSame = va1 == va2
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
      rdd = rdd.mapValuesWithKey {
        case (v, (va, gs)) =>
          val serializer = SparkEnv.get.serializer.newInstance()
          val zeroValue = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))

          (mapOp(va, gs.iterator
            .zip(localSampleIdsBc.value.iterator
              .zip(localSampleAnnotationsBc.value.iterator)).foldLeft(zeroValue) {
            case (acc, (g, (s, sa))) =>
              seqOp(acc, v, va, s, sa, g)
          }), gs)
      }.toOrderedRDD(_.locus))
  }

  def annotateIntervals(is: IntervalTree[Locus],
    path: List[String]): VariantSampleMatrix[T] = {
    val isBc = sparkContext.broadcast(is)
    val (newSignature, inserter) = insertVA(TBoolean, path)
    copy(rdd = rdd.mapValuesWithKey { case (v, (va, gs)) =>
      (inserter(va, Some(isBc.value.contains(Locus(v.contig, v.start)))), gs)
    }.toOrderedRDD(_.locus),
      vaSignature = newSignature)
  }

  def annotateIntervals(is: IntervalTree[Locus],
    t: Type,
    m: Map[Interval[Locus], Annotation],
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
        Some(queries.map(mBc.value))
      else {
        if (queries.isEmpty)
          None
        else
          Some(mBc.value(queries.head))
      }
      (inserter(va, toIns), gs)
    }.toOrderedRDD(_.locus),
      vaSignature = newSignature)
  }

  def annotateVariants(otherRDD: RDD[(Variant, Annotation)], newSignature: Type,
    inserter: Inserter): VariantSampleMatrix[T] = {
    val newRDD = rdd.orderedLeftJoinDistinct(otherRDD.toOrderedRDD(_.locus))
      .mapValues { case ((va, gs), annotation) =>
        (inserter(va, annotation), gs)
      }.toOrderedRDD(_.locus)
    copy(rdd = newRDD, vaSignature = newSignature)
  }

  def annotateLoci(lociRDD: RDD[(Locus, Annotation)], newSignature: Type, inserter: Inserter): VariantSampleMatrix[T] = {
    val newRDD = rdd
      .mapMonotonic({ case (v, vags) => (v.locus, (v, vags)) },
        identity[Locus])
      .orderedLeftJoinDistinct(lociRDD.toOrderedRDD(identity[Locus]))
      .mapValues { case ((v, (va, gs)), annotation) => (v, (inserter(va, annotation), gs)) }
      .toOrderedRDD[Locus](identity)
      .mapMonotonic({ case (_, vvags) => vvags },
        (v: Variant) => v.locus)
    copy(rdd = newRDD, vaSignature = newSignature)
  }

  def annotateSamples(annotations: Map[String, Annotation], signature: Type, path: List[String]): VariantSampleMatrix[T] = {
    val (t, i) = insertSA(signature, path)
    annotateSamples(annotations, t, i)
  }

  def annotateSamples(annotations: Map[String, Annotation], newSignature: Type,
    inserter: Inserter): VariantSampleMatrix[T] = {
    val newAnnotations = sampleIds.zipWithIndex.map { case (id, i) =>
      val sa = sampleAnnotations(i)
      inserter(sa, annotations.get(id))
    }

    copy(sampleAnnotations = newAnnotations, saSignature = newSignature)
  }

  def queryVA(code: String): (BaseType, Querier) = {

    val st = Map(Annotation.VARIANT_HEAD -> (0, vaSignature))
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

    val st = Map(Annotation.SAMPLE_HEAD -> (0, saSignature))
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
    val st = Map(Annotation.GLOBAL_HEAD -> (0, globalSignature))
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

  def makeSchemaForKudu(): StructType =
    makeSchema().add(StructField("sample_group", StringType, nullable = false))

  private def writeMetadata(sqlContext: SQLContext, dirname: String, compress: Boolean = true) = {
    if (!dirname.endsWith(".vds") && !dirname.endsWith(".vds/"))
      fatal(s"output path ending in `.vds' required, found `$dirname'")

    val hConf = vds.sparkContext.hadoopConfiguration
    hadoopMkdir(dirname, hConf)

    val sb = new StringBuilder

    vds.saSignature.pretty(sb, printAttrs = true, compact = true)
    val saSchemaString = sb.result()

    sb.clear()
    vds.vaSignature.pretty(sb, printAttrs = true, compact = true)
    val vaSchemaString = sb.result()

    sb.clear()
    vds.globalSignature.pretty(sb, printAttrs = true, compact = true)
    val globalSchemaString = sb.result()

    val sampleInfoSchema = TStruct(("id", TString), ("annotation", vds.saSignature))
    val sampleInfoJson = JArray(
      vds.sampleIdsAndAnnotations
        .map { case (id, annotation) =>
          JObject(List(("id", JString(id)), ("annotation", JSONAnnotationImpex.exportAnnotation(annotation, vds.saSignature))))
        }
        .toList
    )

    val json = JObject(
      ("version", JInt(VariantSampleMatrix.fileVersion)),
      ("split", JBool(vds.wasSplit)),
      ("isDosage", JBool(vds.isDosage)),
      ("sample_annotation_schema", JString(saSchemaString)),
      ("variant_annotation_schema", JString(vaSchemaString)),
      ("global_annotation_schema", JString(globalSchemaString)),
      ("sample_annotations", sampleInfoJson),
      ("global_annotation", JSONAnnotationImpex.exportAnnotation(vds.globalAnnotation, vds.globalSignature))
    )

    writeTextFile(dirname + "/metadata.json.gz", hConf)(_.write(JsonMethods.pretty(json)))
  }

  def write(sqlContext: SQLContext, dirname: String, compress: Boolean = true) {
    writeMetadata(sqlContext, dirname, compress)

    val vaSignature = vds.vaSignature
    val vaRequiresConversion = SparkAnnotationImpex.requiresConversion(vaSignature)

    val ordered = vds.rdd.toOrderedRDD(_.locus)

    writeObjectFile(dirname + "/partitioner", vds.sparkContext.hadoopConfiguration) { out =>
      ordered.orderedPartitioner.write(out)
    }

    val isDosage = vds.isDosage
    val rowRDD = ordered.map {
      case (v, (va, gs)) =>
        Row.fromSeq(Array(v.toRow,
          if (vaRequiresConversion) SparkAnnotationImpex.exportAnnotation(va, vaSignature) else va,
          gs.toGenotypeStream(v, isDosage, compress).toRow))
    }
    sqlContext.createDataFrame(rowRDD, makeSchema())
      .write.parquet(dirname + "/rdd.parquet")
    // .saveAsParquetFile(dirname + "/rdd.parquet")
  }

  def writeKudu(sqlContext: SQLContext, dirname: String, tableName: String,
    master: String, vcfSeqDict: String, rowsPerPartition: Int,
    sampleGroup: String, compress: Boolean = true, drop: Boolean = false) {

    writeMetadata(sqlContext, dirname, compress)

    val vaSignature = vds.vaSignature
    val isDosage = vds.isDosage

    val rowType = VariantSampleMatrix.kuduRowType(vaSignature)
    val rowRDD = vds.rdd
      .map { case (v, (va, gs)) =>
        KuduAnnotationImpex.exportAnnotation(Annotation(
          v.toRow,
          va,
          gs.toGenotypeStream(v, isDosage, compress).toRow,
          sampleGroup), rowType).asInstanceOf[Row]
      }

    val schema: StructType = KuduAnnotationImpex.exportType(rowType).asInstanceOf[StructType]
    println(s"schema = $schema")
    val df = sqlContext.createDataFrame(rowRDD, schema)

    val kuduContext = new KuduContext(master)
    if (drop) {
      KuduUtils.dropTable(master, tableName)
      Thread.sleep(10 * 1000) // wait to avoid overwhelming Kudu service queue
    }
    if (!KuduUtils.tableExists(master, tableName)) {
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

      val keys = Seq("variant__contig", "variant__start", "variant__ref",
        "variant__altAlleles_0__alt", "sample_group")
      kuduContext.createTable(tableName, schema, keys,
        KuduUtils.createTableOptions(schema, keys, seqDict, rowsPerPartition))
    }
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
        rdd = vds1.rdd.mapValuesWithKey {
          case (v, (va, gs)) =>
            (f2(f1(va)), gs.lazyMap(g => g.copy(fakeRef = false)))
        }.toOrderedRDD(_.locus))
    } else
      vds
  }

  def withGenotypeStream(compress: Boolean = false): VariantDataset = {
    val isDosage = vds.isDosage
    vds.copy(rdd = vds.rdd.mapValuesWithKey[(Annotation, Iterable[Genotype])] { case (v, (va, gs)) =>
      (va, gs.toGenotypeStream(v, isDosage, compress = compress))
    }.toOrderedRDD(_.locus))
  }
}
