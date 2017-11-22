package is.hail.variant

import java.nio.ByteBuffer

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.expr._
import is.hail.io.vcf.ExportVCF
import is.hail.keytable.KeyTable
import is.hail.methods.Aggregators.SampleFunctions
import is.hail.methods._
import is.hail.sparkextras._
import is.hail.utils._
import is.hail.{HailContext, utils}
import org.apache.hadoop
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Partitioner, SparkContext, SparkEnv}
import org.json4s._
import org.json4s.jackson.JsonMethods.parse
import org.json4s.jackson.Serialization

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.language.{existentials, implicitConversions}
import scala.reflect.ClassTag

case class VDSMetadata(
  version: Int,
  split: Boolean,
  sample_schema: String,
  sample_annotation_schema: String,
  variant_schema: String,
  variant_annotation_schema: String,
  global_schema: String,
  genotype_schema: String,
  sample_annotations: JValue,
  global_annotation: JValue,
  n_partitions: Int)

object VariantSampleMatrix {
  final val fileVersion: Int = 0x101

  def read(hc: HailContext, dirname: String,
    dropSamples: Boolean = false, dropVariants: Boolean = false): VariantSampleMatrix = {
    val (fileMetadata, nPartitions) = readFileMetadata(hc.hadoopConf, dirname)
    new VariantSampleMatrix(hc,
      fileMetadata.metadata,
      MatrixRead(hc, dirname, nPartitions, fileMetadata, dropSamples, dropVariants))
  }

  def apply(hc: HailContext,
    metadata: VSMMetadata,
    localValue: VSMLocalValue,
    rdd: OrderedRDD[Annotation, Annotation, (Annotation, Iterable[Annotation])]): VariantSampleMatrix =
    new VariantSampleMatrix(hc, metadata,
      MatrixLiteral(
        MatrixType(metadata),
        MatrixValue(MatrixType(metadata), localValue, rdd)))

  def apply(hc: HailContext, fileMetadata: VSMFileMetadata,
    rdd: OrderedRDD[Annotation, Annotation, (Annotation, Iterable[Annotation])]): VariantSampleMatrix =
    VariantSampleMatrix(hc, fileMetadata.metadata, fileMetadata.localValue, rdd)

  def fromLegacy[RK, T](hc: HailContext,
    metadata: VSMMetadata,
    localValue: VSMLocalValue,
    rdd: RDD[(RK, (Annotation, Iterable[T]))]): VariantSampleMatrix = {
    implicit val kOk = metadata.vSignature.orderedKey
    VariantSampleMatrix(hc, metadata, localValue,
      rdd.map { case (v, (va, gs)) =>
        (v: Annotation, (va, gs: Iterable[Annotation]))
      }.toOrderedRDD)
  }

  def fromLegacy[RK, T](hc: HailContext,
    metadata: VSMMetadata,
    localValue: VSMLocalValue,
    rdd: RDD[(RK, (Annotation, Iterable[T]))],
    fastKeys: RDD[RK]): VariantSampleMatrix = {
    implicit val kOk = metadata.vSignature.orderedKey
    VariantSampleMatrix(hc, metadata, localValue,
      OrderedRDD(
        rdd.map { case (v, (va, gs)) =>
          (v: Annotation, (va, gs: Iterable[Annotation]))
        },
        Some(fastKeys.map { k => k: Annotation }), None))
  }

  def fromLegacy[RK, T](hc: HailContext,
    fileMetadata: VSMFileMetadata,
    rdd: RDD[(RK, (Annotation, Iterable[T]))]): VariantSampleMatrix =
    fromLegacy(hc, fileMetadata.metadata, fileMetadata.localValue, rdd)

  def readFileMetadata(hConf: hadoop.conf.Configuration, dirname: String,
    requireParquetSuccess: Boolean = true): (VSMFileMetadata, Int) = {
    if (!dirname.endsWith(".vds") && !dirname.endsWith(".vds/"))
      fatal(s"input path ending in `.vds' required, found `$dirname'")

    if (!hConf.exists(dirname))
      fatal(s"no VDS found at `$dirname'")

    val metadataFile = dirname + "/metadata.json.gz"
    if (!hConf.exists(metadataFile))
      fatal(
        s"""corrupt or outdated VDS: invalid metadata
           |  No `metadata.json.gz' file found in VDS directory
           |  Recreate VDS with current version of Hail.""".stripMargin)

    val metadata = hConf.readFile(metadataFile) { in =>
      try {
        val json = parse(in)
        json.extract[VDSMetadata]
      } catch {
        case e: Exception => fatal(
          s"""corrupt or outdated VDS: invalid metadata
             |  Recreate VDS with current version of Hail.
             |  Detailed exception:
             |  ${ e.getMessage }""".stripMargin)
      }
    }

    if (metadata.version != VariantSampleMatrix.fileVersion)
      fatal(
        s"""Invalid VDS: old version [${ metadata.version }]
           |  Recreate VDS with current version of Hail.
         """.stripMargin)

    val sSignature = Parser.parseType(metadata.sample_schema)
    val saSignature = Parser.parseType(metadata.sample_annotation_schema)
    val vSignature = Parser.parseType(metadata.variant_schema)
    val vaSignature = Parser.parseType(metadata.variant_annotation_schema)
    val genotypeSignature = Parser.parseType(metadata.genotype_schema)
    val globalSignature = Parser.parseType(metadata.global_schema)

    val sampleInfoSchema = TStruct(("id", sSignature), ("annotation", saSignature))
    val sampleInfo = metadata.sample_annotations.asInstanceOf[JArray]
      .arr
      .map {
        case JObject(List(("id", id), ("annotation", jv))) =>
          (JSONAnnotationImpex.importAnnotation(id, sSignature, "sample_annotations.id"),
            JSONAnnotationImpex.importAnnotation(jv, saSignature, "sample_annotations.annotation"))
        case other => fatal(
          s"""corrupt VDS: invalid metadata
             |  Invalid sample annotation metadata
             |  Recreate VDS with current version of Hail.""".stripMargin)
      }
      .toArray

    val globalAnnotation = JSONAnnotationImpex.importAnnotation(metadata.global_annotation,
      globalSignature, "global")

    val ids = sampleInfo.map(_._1)
    val annotations = sampleInfo.map(_._2)

    (VSMFileMetadata(VSMMetadata(sSignature, saSignature, vSignature, vaSignature, globalSignature, genotypeSignature, metadata.split),
      VSMLocalValue(globalAnnotation, ids, annotations)),
      metadata.n_partitions)
  }

  def gen(hc: HailContext, gen: VSMSubgen): Gen[VariantSampleMatrix] =
    gen.gen(hc)

  def genGeneric(hc: HailContext): Gen[GenericDataset] =
    VSMSubgen(
      sSigGen = Type.genArb,
      saSigGen = Type.genArb,
      vSigGen = Type.genArb,
      vaSigGen = Type.genArb,
      globalSigGen = Type.genArb,
      tSigGen = Type.genArb,
      sGen = (t: Type) => t.genNonmissingValue,
      saGen = (t: Type) => t.genValue,
      vaGen = (t: Type) => t.genValue,
      globalGen = (t: Type) => t.genValue,
      vGen = (t: Type) => t.genNonmissingValue,
      tGen = (t: Type, v: Annotation) => t.genValue.resize(20))
      .gen(hc)

  def checkDatasetSchemasCompatible(datasets: Array[VariantSampleMatrix]) {
    val first = datasets(0)
    val sampleIds = first.sampleIds
    val vaSchema = first.vaSignature
    val wasSplit = first.wasSplit
    val genotypeSchema = first.genotypeSignature
    val rowKeySchema = first.vSignature
    val colKeySchema = first.sSignature

    datasets.indices.tail.foreach { i =>
      val vds = datasets(i)
      val ids = vds.sampleIds
      val vas = vds.vaSignature
      val gsig = vds.genotypeSignature
      val vsig = vds.vSignature
      val ssig = vds.sSignature

      if (ssig != colKeySchema) {
        fatal(
          s"""cannot combine datasets with different column key schemata
             |  Schema in datasets[0]: @1
             |  Schema in datasets[$i]: @2""".stripMargin,
          colKeySchema.toPrettyString(compact = true, printAttrs = true),
          ssig.toPrettyString(compact = true, printAttrs = true)
        )
      } else if (vsig != rowKeySchema) {
        fatal(
          s"""cannot combine datasets with different row key schemata
             |  Schema in datasets[0]: @1
             |  Schema in datasets[$i]: @2""".stripMargin,
          rowKeySchema.toPrettyString(compact = true, printAttrs = true),
          vsig.toPrettyString(compact = true, printAttrs = true)
        )
      } else if (ids != sampleIds) {
        fatal(
          s"""cannot combine datasets with different column identifiers or ordering
             |  IDs in datasets[0]: @1
             |  IDs in datasets[$i]: @2""".stripMargin, sampleIds, ids)
      } else if (wasSplit != vds.wasSplit) {
        fatal(
          s"""cannot combine split and unsplit datasets
             |  Split status in datasets[0]: $wasSplit
             |  Split status in datasets[$i]: ${ vds.wasSplit }""".stripMargin)
      } else if (vas != vaSchema) {
        fatal(
          s"""cannot combine datasets with different row annotation schemata
             |  Schema in datasets[0]: @1
             |  Schema in datasets[$i]: @2""".stripMargin,
          vaSchema.toPrettyString(compact = true, printAttrs = true),
          vas.toPrettyString(compact = true, printAttrs = true)
        )
      } else if (gsig != genotypeSchema) {
        fatal(
          s"""cannot read datasets with different cell schemata
             |  Schema in datasets[0]: @1
             |  Schema in datasets[$i]: @2""".stripMargin,
          genotypeSchema.toPrettyString(compact = true, printAttrs = true),
          gsig.toPrettyString(compact = true, printAttrs = true)
        )
      }
    }
  }

  def union(datasets: java.util.ArrayList[VariantSampleMatrix]): VariantSampleMatrix =
    union(datasets.asScala.toArray)

  def union(datasets: Array[VariantSampleMatrix]): VariantSampleMatrix = {
    require(datasets.length >= 2)

    checkDatasetSchemasCompatible(datasets)
    val (first, others) = (datasets.head, datasets.tail)
    first.copyLegacy(rdd = first.sparkContext.union(datasets.map(_.rdd)))
  }
}

case class VSMSubgen(
  sSigGen: Gen[Type],
  saSigGen: Gen[Type],
  vSigGen: Gen[Type],
  vaSigGen: Gen[Type],
  globalSigGen: Gen[Type],
  tSigGen: Gen[Type],
  sGen: (Type) => Gen[Annotation],
  saGen: (Type) => Gen[Annotation],
  vaGen: (Type) => Gen[Annotation],
  globalGen: (Type) => Gen[Annotation],
  vGen: (Type) => Gen[Annotation],
  tGen: (Type, Annotation) => Gen[Annotation],
  wasSplit: Boolean = false) {

  def gen(hc: HailContext): Gen[VariantSampleMatrix] =
    for (size <- Gen.size;
      subsizes <- Gen.partitionSize(5).resize(size / 10);
      vSig <- vSigGen.resize(3);
      vaSig <- vaSigGen.resize(subsizes(0));
      sSig <- sSigGen.resize(3);
      saSig <- saSigGen.resize(subsizes(1));
      globalSig <- globalSigGen.resize(subsizes(2));
      tSig <- tSigGen.resize(3);
      global <- globalGen(globalSig).resize(subsizes(3));
      nPartitions <- Gen.choose(1, 10);

      (l, w) <- Gen.squareOfAreaAtMostSize.resize((size / 10) * 9);

      sampleIds <- Gen.distinctBuildableOf[Array, Annotation](sGen(sSig).resize(3)).resize(w)
        .map(a => a.filter(_ != null));
      nSamples = sampleIds.length;
      saValues <- Gen.buildableOfN[Array, Annotation](nSamples, saGen(saSig)).resize(subsizes(4));
      rows <- Gen.distinctBuildableOf[Array, (Annotation, (Annotation, Iterable[Annotation]))](
        for (subsubsizes <- Gen.partitionSize(2);
          v <- vGen(vSig).resize(3);
          va <- vaGen(vaSig).resize(subsubsizes(0));
          ts <- Gen.buildableOfN[Array, Annotation](nSamples, tGen(tSig, v)).resize(subsubsizes(1)))
          yield (v, (va, ts: Iterable[Annotation]))).resize(l)
        .map(a => a.filter(_._1 != null)))
      yield {
        VariantSampleMatrix.fromLegacy(hc,
          VSMMetadata(sSig, saSig, vSig, vaSig, globalSig, tSig, wasSplit = wasSplit),
          VSMLocalValue(global, sampleIds, saValues),
          hc.sc.parallelize(rows, nPartitions))
          .deduplicate()
      }
}

object VSMSubgen {
  val random = VSMSubgen(
    sSigGen = Gen.const(TString()),
    saSigGen = Type.genInsertable,
    vSigGen = Gen.const(TVariant(GenomeReference.defaultReference)),
    vaSigGen = Type.genInsertable,
    globalSigGen = Type.genInsertable,
    tSigGen = Gen.const(TGenotype()),
    sGen = (t: Type) => Gen.identifier.map(s => s: Annotation),
    saGen = (t: Type) => t.genValue,
    vaGen = (t: Type) => t.genValue,
    globalGen = (t: Type) => t.genValue,
    vGen = (t: Type) => Variant.gen,
    tGen = (t: Type, v: Annotation) => Genotype.genExtreme(v.asInstanceOf[Variant]))

  val plinkSafeBiallelic = random.copy(
    sGen = (t: Type) => Gen.plinkSafeIdentifier,
    vGen = (t: Type) => VariantSubgen.plinkCompatible.copy(nAllelesGen = Gen.const(2)).gen,
    wasSplit = true)

  val dosage = VSMSubgen(
    sSigGen = Gen.const(TString()),
    saSigGen = Type.genInsertable,
    vSigGen = Gen.const(TVariant(GenomeReference.defaultReference)),
    vaSigGen = Type.genInsertable,
    globalSigGen = Type.genInsertable,
    tSigGen = Gen.const(TStruct(
      "GT" -> TCall(),
      "GP" -> TArray(TFloat64()))),
    sGen = (t: Type) => Gen.identifier.map(s => s: Annotation),
    saGen = (t: Type) => t.genValue,
    vaGen = (t: Type) => t.genValue,
    globalGen = (t: Type) => t.genValue,
    vGen = (t: Type) => Variant.gen,
    tGen = (t: Type, v: Annotation) => Genotype.genGenericDosageGenotype(v.asInstanceOf[Variant]))

  val realistic = random.copy(
    tGen = (t: Type, v: Annotation) => Genotype.genRealistic(v.asInstanceOf[Variant]))
}

class VariantSampleMatrix(val hc: HailContext, val metadata: VSMMetadata,
  val ast: MatrixIR) extends JoinAnnotator {

  implicit val kOk: OrderedKey[Annotation, Annotation] = ast.typ.vType.orderedKey

  implicit val kOrd: Ordering[Annotation] = kOk.kOrd

  def this(hc: HailContext,
    metadata: VSMMetadata,
    localValue: VSMLocalValue,
    rdd2: OrderedRDD2) =
    this(hc, metadata,
      MatrixLiteral(
        MatrixType(metadata),
        MatrixValue(MatrixType(metadata), localValue, rdd2)))

  def requireRowKeyVariant(method: String) {
    vSignature match {
      case _: TVariant =>
      case _ =>
        fatal(s"in $method: row key (variant) schema must be Variant, found: $vSignature")
    }
  }

  def requireColKeyString(method: String) {
    sSignature match {
      case _: TString =>
      case t =>
        fatal(s"in $method: column key schema must be String, found: $t")
    }
  }

  val VSMMetadata(sSignature, saSignature, vSignature, vaSignature, globalSignature, genotypeSignature, wasSplit) = metadata

  lazy val value: MatrixValue = {
    val opt = MatrixIR.optimize(ast)
    opt.execute(hc)
  }

  lazy val MatrixValue(matrixType, VSMLocalValue(globalAnnotation, sampleIds, sampleAnnotations), rdd2) = value

  lazy val rdd: OrderedRDD[Annotation, Annotation, (Annotation, Iterable[Annotation])] = value.rdd

  def typedRDD[RPK, RK, T](implicit rkct: ClassTag[RK], tct: ClassTag[T]): OrderedRDD[RPK, RK, (Annotation, Iterable[T])] = {
    implicit val kOk = vSignature.typedOrderedKey[RPK, RK]
    rdd.map { case (v, (va, gs)) =>
      (v.asInstanceOf[RK], (va, gs.asInstanceOf[Iterable[T]]))
    }
      .toOrderedRDD
  }

  def stringSampleIds: IndexedSeq[String] = {
    assert(sSignature.isInstanceOf[TString])
    sampleIds.map(_.asInstanceOf[String])
  }

  def stringSampleIdSet: Set[String] = stringSampleIds.toSet

  type RowT = (Annotation, (Annotation, Iterable[Annotation]))

  lazy val sampleIdsBc = sparkContext.broadcast(sampleIds)

  lazy val sampleAnnotationsBc = sparkContext.broadcast(sampleAnnotations)

  def requireUniqueSamples(method: String) {
    val dups = sampleIds.counter().filter(_._2 > 1).toArray
    if (dups.nonEmpty)
      fatal(s"Method '$method' does not support duplicate sample IDs. Duplicates:" +
        s"\n  @1", dups.sortBy(-_._2).map { case (id, count) => s"""($count) "$id"""" }.truncatable("\n  "))
  }

  def unsafeRowRDD(): RDD[UnsafeRow] = {
    val localRowType = rowType
    rdd2.map { rv =>
      new UnsafeRow(localRowType, rv.region.copy(), rv.offset)
    }
  }

  def collect(): Array[UnsafeRow] = unsafeRowRDD().collect()

  def take(n: Int): Array[UnsafeRow] = unsafeRowRDD().take(n)

  def aggregateBySamplePerVariantKey(keyName: String, variantKeysVA: String, aggExpr: String, singleKey: Boolean = false): KeyTable = {

    val (keysType, keysQuerier) = queryVA(variantKeysVA)

    val (keyType, keyedRdd) =
      if (singleKey) {
        (keysType, rdd.flatMap { case (v, (va, gs)) => Option(keysQuerier(va)).map(key => (key, (v, va, gs))) })
      } else {
        val keyType = keysType match {
          case TArray(e, _) => e
          case TSet(e, _) => e
          case _ => fatal(s"With single_key=False, variant keys must be of type Set[T] or Array[T], got $keysType")
        }
        (keyType, rdd.flatMap { case (v, (va, gs)) =>
          Option(keysQuerier(va).asInstanceOf[Iterable[_]]).getOrElse(Iterable.empty).map(key => (key, (v, va, gs)))
        })
      }

    val SampleFunctions(zero, seqOp, combOp, resultOp, resultType) = Aggregators.makeSampleFunctions(this, aggExpr)

    val ktRDD = keyedRdd
      .aggregateByKey(zero)(seqOp, combOp)
      .map { case (key, agg) =>
        val results = resultOp(agg)
        results(0) = key
        Row.fromSeq(results)
      }

    val signature = TStruct((keyName -> keyType) +: stringSampleIds.map(id => id -> resultType): _*)

    KeyTable(hc, ktRDD, signature, key = Array(keyName))
  }

  def aggregateByVariantWithAll[U](zeroValue: U)(
    seqOp: (U, Annotation, Annotation, Annotation, Annotation, Annotation) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Annotation, U)] = {

    // Serialize the zero value to a byte array so that we can apply a new clone of it on each key
    val zeroBuffer = SparkEnv.get.serializer.newInstance().serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    rdd
      .mapPartitions({ (it: Iterator[(Annotation, (Annotation, Iterable[Annotation]))]) =>
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

  def aggregateByVariantWithKeys[U](zeroValue: U)(
    seqOp: (U, Annotation, Annotation, Annotation) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Annotation, U)] = {
    aggregateByVariantWithAll(zeroValue)((e, v, va, s, sa, g) => seqOp(e, v, s, g), combOp)
  }

  def annotateGlobal(a: Annotation, t: Type, code: String): VariantSampleMatrix = {
    val (newT, i) = insertGlobal(t, Parser.parseAnnotationRoot(code, Annotation.GLOBAL_HEAD))
    copy2(globalSignature = newT, globalAnnotation = i(globalAnnotation, a))
  }

  /**
    * Create and destroy global annotations with expression language.
    *
    * @param expr Annotation expression
    */
  def annotateGlobalExpr(expr: String): VariantSampleMatrix = {
    val ec = EvalContext(Map(
      "global" -> (0, globalSignature)))

    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Option(Annotation.GLOBAL_HEAD))

    val inserterBuilder = new ArrayBuilder[Inserter]()

    val finalType = (paths, types).zipped.foldLeft(globalSignature) { case (v, (ids, signature)) =>
      val (s, i) = v.insert(signature, ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    ec.set(0, globalAnnotation)
    val ga = inserters
      .zip(f())
      .foldLeft(globalAnnotation) { case (a, (ins, res)) =>
        ins(a, res)
      }

    copy2(globalAnnotation = ga,
      globalSignature = finalType)
  }

  def insertGlobal(sig: Type, path: List[String]): (Type, Inserter) = {
    globalSignature.insert(sig, path)
  }

  def annotateSamples(signature: Type, path: List[String], annotations: Array[Annotation]): VariantSampleMatrix = {
    val (t, ins) = insertSA(signature, path)

    val newAnnotations = new Array[Annotation](nSamples)

    for (i <- sampleAnnotations.indices) {
      newAnnotations(i) = ins(sampleAnnotations(i), annotations(i))
      t.typeCheck(newAnnotations(i))
    }

    copy(sampleAnnotations = newAnnotations, saSignature = t)
  }

  def annotateSamples(signature: Type, path: List[String], annotation: (Annotation) => Annotation): VariantSampleMatrix = {
    val (t, i) = insertSA(signature, path)
    annotateSamples(annotation, t, i)
  }

  def annotateSamplesExpr(expr: String): VariantSampleMatrix = {
    val ec = sampleEC

    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Some(Annotation.SAMPLE_HEAD))

    val inserterBuilder = new ArrayBuilder[Inserter]()
    val finalType = (paths, types).zipped.foldLeft(saSignature) { case (sas, (ids, signature)) =>
      val (s, i) = sas.insert(signature, ids)
      inserterBuilder += i
      s
    }
    val inserters = inserterBuilder.result()

    val sampleAggregationOption = Aggregators.buildSampleAggregations(hc, value, ec)

    ec.set(0, globalAnnotation)
    val newAnnotations = sampleIdsAndAnnotations.map { case (s, sa) =>
      sampleAggregationOption.foreach(f => f.apply(s))
      ec.set(1, s)
      ec.set(2, sa)
      f().zip(inserters)
        .foldLeft(sa) { case (sa, (v, inserter)) =>
          inserter(sa, v)
        }
    }

    copy2(
      sampleAnnotations = newAnnotations,
      saSignature = finalType
    )
  }

  def annotateSamples(annotations: Map[Annotation, Annotation], signature: Type, code: String): VariantSampleMatrix = {
    val (t, i) = insertSA(signature, Parser.parseAnnotationRoot(code, Annotation.SAMPLE_HEAD))
    annotateSamples(s => annotations.getOrElse(s, null), t, i)
  }

  def annotateSamplesTable(kt: KeyTable, vdsKey: java.util.ArrayList[String],
    root: String, expr: String, product: Boolean): VariantSampleMatrix =
    annotateSamplesTable(kt, if (vdsKey != null) vdsKey.asScala else null, root, expr, product)

  def annotateSamplesTable(kt: KeyTable, vdsKey: Seq[String] = null,
    root: String = null, expr: String = null, product: Boolean = false): VariantSampleMatrix = {

    if (root == null && expr == null || root != null && expr != null)
      fatal("method `annotateSamplesTable' requires one of `root' or 'expr', but not both")

    var (joinSignature, f): (Type, Annotation => Annotation) = kt.valueSignature.size match {
      case 0 => (TBoolean(), _ != null)
      case 1 => (kt.valueSignature.fields.head.typ, x => if (x != null) x.asInstanceOf[Row].get(0) else null)
      case _ => (kt.valueSignature, identity[Annotation])
    }

    if (product) {
      joinSignature = if (joinSignature.isInstanceOf[TBoolean]) TInt32() else TArray(joinSignature)
      f = if (kt.valueSignature.size == 0)
        _.asInstanceOf[IndexedSeq[_]].length
      else {
        val g = f
        _.asInstanceOf[IndexedSeq[_]].map(g)
      }
    }

    val (finalType, inserter): (Type, (Annotation, Annotation) => Annotation) = {
      val (t, ins) = if (expr != null) {
        val ec = EvalContext(Map(
          "sa" -> (0, saSignature),
          "table" -> (1, joinSignature)))
        Annotation.buildInserter(expr, saSignature, ec, Annotation.SAMPLE_HEAD)
      } else insertSA(joinSignature, Parser.parseAnnotationRoot(root, Annotation.SAMPLE_HEAD))

      (t, (a: Annotation, toIns: Annotation) => ins(a, f(toIns)))
    }

    val keyTypes = kt.keyFields.map(_.typ)

    val keyedRDD = kt.keyedRDD()
      .filter { case (k, v) => k.toSeq.forall(_ != null) }

    val nullValue: IndexedSeq[Annotation] = if (product) IndexedSeq() else null

    if (vdsKey != null) {
      val keyEC = EvalContext(Map("s" -> (0, sSignature), "sa" -> (1, saSignature)))
      val (vdsKeyType, vdsKeyFs) = vdsKey.map(Parser.parseExpr(_, keyEC)).unzip

      if (!keyTypes.sameElements(vdsKeyType))
        fatal(
          s"""method `annotateSamplesTable' encountered a mismatch between table keys and computed keys.
             |  Computed keys:  [ ${ vdsKeyType.mkString(", ") } ]
             |  Key table keys: [ ${ keyTypes.mkString(", ") } ]""".stripMargin)

      val keyFuncArray = vdsKeyFs.toArray

      val thisRdd = sparkContext.parallelize(sampleIdsAndAnnotations.map { case (s, sa) =>
        keyEC.setAll(s, sa)
        (Row.fromSeq(keyFuncArray.map(_ ())), s)
      })

      var r = keyedRDD.join(thisRdd).map { case (_, (tableAnnotation, s)) => (s, tableAnnotation: Annotation) }
      if (product)
        r = r.groupByKey().mapValues(is => (is.toArray[Annotation]: IndexedSeq[Annotation]): Annotation)

      val m = r.collectAsMap()

      annotateSamples(m.getOrElse(_, nullValue), finalType, inserter)
    } else {
      keyTypes match {
        case Array(`sSignature`) =>
          var r = keyedRDD.map { case (k, v) => (k.asInstanceOf[Row].get(0), v: Annotation) }

          if (product)
            r = r.groupByKey()
              .map { case (s, rows) => (s, (rows.toArray[Annotation]: IndexedSeq[_]): Annotation) }

          val m = r.collectAsMap()

          annotateSamples(m.getOrElse(_, nullValue), finalType, inserter)
        case other =>
          fatal(
            s"""method 'annotate_samples_table' expects a key table keyed by [ $sSignature ]
               |  Found key [ ${ other.mkString(", ") } ] instead.""".stripMargin)
      }
    }
  }

  def annotateSamples(annotation: (Annotation) => Annotation, newSignature: Type, inserter: Inserter): VariantSampleMatrix = {
    val newAnnotations = sampleIds.zipWithIndex.map { case (id, i) =>
      val sa = sampleAnnotations(i)
      val newAnnotation = inserter(sa, annotation(id))
      newSignature.typeCheck(newAnnotation)
      newAnnotation
    }

    copy2(sampleAnnotations = newAnnotations, saSignature = newSignature)
  }

  def annotateVariants(otherRDD: OrderedRDD[Annotation, Annotation, Annotation], signature: Type,
    code: String): VariantSampleMatrix = {
    val (newSignature, ins) = insertVA(signature, Parser.parseAnnotationRoot(code, Annotation.VARIANT_HEAD))
    annotateVariants(otherRDD, newSignature, ins, product = false)
  }

  def annotateVariantsExpr(expr: String): VariantSampleMatrix = {
    val localGlobalAnnotation = globalAnnotation

    val ec = variantEC
    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Some(Annotation.VARIANT_HEAD))

    var newVASignature = vaSignature
    val inserters = new Array[Inserter](types.length)
    var i = 0
    while (i < types.length) {
      val (newSig, ins) = newVASignature.insert(types(i), paths(i))
      inserters(i) = ins
      newVASignature = newSig
      i += 1
    }

    val aggregateOption = Aggregators.buildVariantAggregations(this, ec)

    val localRowType = rowType
    insertIntoRow(() => new UnsafeRow(localRowType))(
      newVASignature, List("va"), { (ur, rv, rvb) =>
        ur.set(rv)

        val v = ur.getAs[Annotation](1)
        val va = ur.get(2)
        val gs = ur.getAs[Iterable[Annotation]](3)

        ec.setAll(localGlobalAnnotation, v, va)

        aggregateOption.foreach(f => f(rv))

        var newVA = va
        var i = 0
        var newA = f()
        while (i < newA.length) {
          newVA = inserters(i)(newVA, newA(i))
          i += 1
        }

        rvb.addAnnotation(newVASignature, newVA)
      })
  }

  def annotateVariantsTable(kt: KeyTable, vdsKey: java.util.ArrayList[String],
    root: String, expr: String, product: Boolean): VariantSampleMatrix =
    annotateVariantsTable(kt, if (vdsKey != null) vdsKey.asScala else null, root, expr, product)

  def annotateVariantsTable(kt: KeyTable, vdsKey: Seq[String] = null,
    root: String = null, expr: String = null, product: Boolean = false): VariantSampleMatrix = {

    if (root == null && expr == null || root != null && expr != null)
      fatal("method `annotateVariantsTable' requires one of `root' or 'expr', but not both")

    var (joinSignature, f): (Type, Annotation => Annotation) = kt.valueSignature.size match {
      case 0 => (TBoolean(), _ != null)
      case 1 => (kt.valueSignature.fields.head.typ, x => if (x != null) x.asInstanceOf[Row].get(0) else null)
      case _ => (kt.valueSignature, identity[Annotation])
    }

    if (product) {
      joinSignature = if (joinSignature.isInstanceOf[TBoolean]) TInt32(joinSignature.required) else TArray(joinSignature)
      f = if (kt.valueSignature.size == 0)
        _.asInstanceOf[IndexedSeq[_]].length
      else {
        val g = f
        _.asInstanceOf[IndexedSeq[_]].map(g)
      }
    }

    val (finalType, inserter): (Type, (Annotation, Annotation) => Annotation) = {
      val (t, ins) = if (expr != null) {
        val ec = EvalContext(Map(
          "va" -> (0, vaSignature),
          "table" -> (1, joinSignature)))
        Annotation.buildInserter(expr, vaSignature, ec, Annotation.VARIANT_HEAD)
      } else insertVA(joinSignature, Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD))

      (t, (a: Annotation, toIns: Annotation) => ins(a, f(toIns)))
    }

    val keyTypes = kt.keyFields.map(_.typ)

    val keyedRDD = kt.keyedRDD()
      .filter { case (k, v) => k.toSeq.forall(_ != null) }

    if (vdsKey != null) {
      val keyEC = EvalContext(Map("v" -> (0, vSignature), "va" -> (1, vaSignature)))
      val (vdsKeyType, vdsKeyFs) = vdsKey.map(Parser.parseExpr(_, keyEC)).unzip

      if (!keyTypes.sameElements(vdsKeyType))
        fatal(
          s"""method `annotateVariantsTable' encountered a mismatch between table keys and computed keys.
             |  Computed keys:  [ ${ vdsKeyType.mkString(", ") } ]
             |  Key table keys: [ ${ keyTypes.mkString(", ") } ]""".stripMargin)

      val thisRdd = rdd.map { case (v, (va, gs)) =>
        keyEC.setAll(v, va)
        (Row.fromSeq(vdsKeyFs.map(_ ())), v)
      }

      val joinedRDD = keyedRDD
        .join(thisRdd)
        .map { case (_, (table, v)) => (v, table: Annotation) }
        .orderedRepartitionBy(rdd.orderedPartitioner)

      annotateVariants(joinedRDD, finalType, inserter, product = product)

    } else {
      keyTypes match {
        case Array(`vSignature`) =>
          val ord = keyedRDD
            .map { case (k, v) => (k.getAs[Annotation](0), v: Annotation) }
            .toOrderedRDD(rdd.orderedPartitioner)

          annotateVariants(ord, finalType, inserter, product = product)

        case Array(vSignature.partitionKey) =>
          val ord = keyedRDD
            .map { case (k, v) => (k.asInstanceOf[Row].getAs[Annotation](0), v: Annotation) }
            .toOrderedRDD(rdd.orderedPartitioner.projectToPartitionKey())

          annotateLoci(ord, finalType, inserter, product = product)

        case Array(TInterval(_, _)) if vSignature.isInstanceOf[TVariant] =>
          val partBc = sparkContext.broadcast(rdd.orderedPartitioner)
          val partitionKeyedIntervals = keyedRDD
            .flatMap { case (k, v) =>
              val interval = k.getAs[Interval[Locus]](0)
              val start = partBc.value.getPartitionT(interval.start.asInstanceOf[Annotation])
              val end = partBc.value.getPartitionT(interval.end.asInstanceOf[Annotation])
              (start to end).view.map(i => (i, (interval, v)))
            }

          type IntervalT = (Interval[Locus], Annotation)
          val nParts = rdd.partitions.length
          val zipRDD = partitionKeyedIntervals.partitionBy(new Partitioner {
            def getPartition(key: Any): Int = key.asInstanceOf[Int]

            def numPartitions: Int = nParts
          }).values

          val res = rdd.zipPartitions(zipRDD, preservesPartitioning = true) { case (it, intervals) =>
            val iTree = IntervalTree.annotationTree[Locus, Annotation](intervals.toArray)

            it.map { case (v, (va, gs)) =>
              val queries = iTree.queryValues(v.asInstanceOf[Variant].locus)
              val annot = if (product)
                queries: IndexedSeq[Annotation]
              else
                queries.headOption.orNull

              (v, (inserter(va, annot), gs))
            }
          }.asOrderedRDD

          copy(rdd = res, vaSignature = finalType)

        case other =>
          fatal(
            s"""method 'annotate_variants_table' expects a key table keyed by one of the following:
               |  [ $vSignature ]
               |  [ Locus ]
               |  [ Interval ]
               |  Found key [ ${ keyTypes.mkString(", ") } ] instead.""".stripMargin)
      }
    }
  }

  def annotateLoci(lociRDD: OrderedRDD[Annotation, Annotation, Annotation], newSignature: Type,
    inserter: Inserter, product: Boolean): VariantSampleMatrix = {

    def annotate[S](joinedRDD: RDD[(Annotation, ((Annotation, (Annotation, Iterable[Annotation])), S))],
      ins: (Annotation, S) => Annotation): OrderedRDD[Annotation, Annotation, (Annotation, Iterable[Annotation])] = {
      OrderedRDD(joinedRDD.mapPartitions({ it =>
        it.map { case (l, ((v, (va, gs)), annotation)) => (v, (ins(va, annotation), gs)) }
      }),
        rdd.orderedPartitioner)
    }

    val locusKeyedRDD = rdd.mapMonotonic(kOk.orderedProject, { case (v, vags) => (v, vags) })

    val newRDD =
      if (product)
        annotate[Array[Annotation]](locusKeyedRDD.orderedLeftJoin(lociRDD),
          (va, a) => inserter(va, a: IndexedSeq[_]))
      else
        annotate[Option[Annotation]](locusKeyedRDD.orderedLeftJoinDistinct(lociRDD),
          (va, a) => inserter(va, a.orNull))

    copy(rdd = newRDD, vaSignature = newSignature)
  }

  def nPartitions: Int = rdd.partitions.length

  def annotateVariants(otherRDD: OrderedRDD[Annotation, Annotation, Annotation], newSignature: Type,
    inserter: Inserter, product: Boolean): VariantSampleMatrix = {
    val newRDD = if (product)
      rdd.orderedLeftJoin(otherRDD)
        .mapValues { case ((va, gs), annotation) =>
          (inserter(va, annotation: IndexedSeq[_]), gs)
        }
    else
      rdd.orderedLeftJoinDistinct(otherRDD)
        .mapValues { case ((va, gs), annotation) =>
          (inserter(va, annotation.orNull), gs)
        }

    copy(rdd = newRDD, vaSignature = newSignature)
  }

  def annotateVariantsVDS(other: VariantSampleMatrix,
    root: Option[String] = None, code: Option[String] = None): VariantSampleMatrix = {

    val (isCode, annotationExpr) = (root, code) match {
      case (Some(r), None) => (false, r)
      case (None, Some(c)) => (true, c)
      case _ => fatal("this module requires one of `root' or 'code', but not both")
    }

    val (finalType, inserter): (Type, (Annotation, Annotation) => Annotation) =
      if (isCode) {
        val ec = EvalContext(Map(
          "va" -> (0, vaSignature),
          "vds" -> (1, other.vaSignature)))
        Annotation.buildInserter(annotationExpr, vaSignature, ec, Annotation.VARIANT_HEAD)
      } else insertVA(other.vaSignature, Parser.parseAnnotationRoot(annotationExpr, Annotation.VARIANT_HEAD))

    annotateVariants(other.variantsAndAnnotations, finalType, inserter, product = false)
  }

  def count(): (Long, Long) = (nSamples, countVariants())

  def countVariants(): Long = rdd2.count()

  def variants: RDD[Annotation] = rdd.keys

  def deduplicate(): VariantSampleMatrix =
    copy2(rdd2 = rdd2.mapPartitionsPreservesPartitioning(
      SortedDistinctRowIterator.transformer(rdd2.typ)))

  def deleteVA(args: String*): (Type, Deleter) = deleteVA(args.toList)

  def deleteVA(path: List[String]): (Type, Deleter) = vaSignature.delete(path)

  def dropSamples(): VariantSampleMatrix =
    copyAST(ast = FilterSamples(ast, Const(null, false, TBoolean())))

  def dropVariants(): VariantSampleMatrix = copy2(rdd2 = OrderedRDD2.empty(sparkContext, matrixType.orderedRDD2Type))

  def expand(): RDD[(Annotation, Annotation, Annotation)] =
    mapWithKeys[(Annotation, Annotation, Annotation)]((v, s, g) => (v, s, g))

  def expandWithAll(): RDD[(Annotation, Annotation, Annotation, Annotation, Annotation)] =
    mapWithAll[(Annotation, Annotation, Annotation, Annotation, Annotation)]((v, va, s, sa, g) => (v, va, s, sa, g))

  def mapWithAll[U](f: (Annotation, Annotation, Annotation, Annotation, Annotation) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    rdd
      .flatMap { case (v, (va, gs)) =>
        localSampleIdsBc.value.lazyMapWith2[Annotation, Annotation, U](localSampleAnnotationsBc.value, gs, { case (s, sa, g) => f(v, va, s, sa, g)
        })
      }
  }

  def annotateGenotypesExpr(expr: String): VariantSampleMatrix = {
    val symTab = Map(
      "v" -> (0, vSignature),
      "va" -> (1, vaSignature),
      "s" -> (2, sSignature),
      "sa" -> (3, saSignature),
      "g" -> (4, genotypeSignature),
      "global" -> (5, globalSignature))
    val ec = EvalContext(symTab)

    ec.set(5, globalAnnotation)

    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Some(Annotation.GENOTYPE_HEAD))

    val inserterBuilder = new ArrayBuilder[Inserter]()
    val finalType = (paths, types).zipped.foldLeft(genotypeSignature) { case (gsig, (ids, signature)) =>
      val (s, i) = gsig.insert(signature, ids)
      inserterBuilder += i
      s
    }
    val inserters = inserterBuilder.result()

    info(
      s"""Modified the genotype schema with annotateGenotypesExpr.
         |  Original: ${ genotypeSignature.toPrettyString(compact = true) }
         |  New: ${ finalType.toPrettyString(compact = true) }""".stripMargin)

    mapValuesWithAll(finalType, { (v: Annotation, va: Annotation, s: Annotation, sa: Annotation, g: Annotation) =>
      ec.setAll(v, va, s, sa, g)
      f().zip(inserters)
        .foldLeft(g: Annotation) { case (ga, (a, inserter)) =>
          inserter(ga, a)
        }
    })
  }

  def filterVariants(p: (Annotation, Annotation, Iterable[Annotation]) => Boolean): VariantSampleMatrix = {
    val localRowType = matrixType.rowType
    copy2(rdd2 = rdd2.filter { rv =>
      // FIXME ur could be allocate once and set
      val ur = new UnsafeRow(localRowType, rv.region, rv.offset)

      val v = ur.getAs[Annotation](1)
      val va = ur.get(2)
      val gs = ur.getAs[IndexedSeq[Annotation]](3)

      p(v, va, gs)
    })
  }

  def filterSamplesMask(mask: Array[Boolean]): VariantSampleMatrix = {
    require(mask.length == nSamples)
    val maskBc = sparkContext.broadcast(mask)
    copy(sampleIds = sampleIds.zipWithIndex
      .filter { case (s, i) => mask(i) }
      .map(_._1),
      sampleAnnotations = sampleAnnotations.zipWithIndex
        .filter { case (sa, i) => mask(i) }
        .map(_._1),
      rdd = rdd.mapValues { case (va, gs) =>
        (va, gs.lazyFilterWith(maskBc.value, (g: Annotation, m: Boolean) => m))
      })
  }

  // FIXME see if we can remove broadcasts elsewhere in the code
  def filterSamples(p: (Annotation, Annotation) => Boolean): VariantSampleMatrix = {
    val mask = sampleIdsAndAnnotations.map { case (s, sa) => p(s, sa) }.toArray
    filterSamplesMask(mask)
  }

  /**
    * Filter samples using the Hail expression language.
    *
    * @param filterExpr Filter expression involving `s' (sample) and `sa' (sample annotations)
    * @param keep       keep where filterExpr evaluates to true
    */
  def filterSamplesExpr(filterExpr: String, keep: Boolean = true): VariantSampleMatrix = {
    var filterAST = Parser.expr.parse(filterExpr)
    if (!keep)
      filterAST = Apply(filterAST.getPos, "!", Array(filterAST))
    copyAST(ast = FilterSamples(ast, filterAST))
  }

  def filterSamplesList(samples: java.util.ArrayList[Annotation], keep: Boolean): VariantSampleMatrix =
    filterSamplesList(samples.asScala.toSet, keep)

  /**
    * Filter samples using a text file containing sample IDs
    *
    * @param samples Set of samples to keep or remove
    * @param keep    Keep listed samples.
    */
  def filterSamplesList(samples: Set[Annotation], keep: Boolean = true): VariantSampleMatrix = {
    val p = (s: Annotation, sa: Annotation) => Filter.keepThis(samples.contains(s), keep)
    filterSamples(p)
  }

  def filterSamplesTable(table: KeyTable, keep: Boolean): VariantSampleMatrix = {
    table.keyFields.map(_.typ) match {
      case Array(`sSignature`) =>
        val sampleSet = table.keyedRDD()
          .map { case (k, v) => k.get(0) }
          .filter(_ != null)
          .collectAsSet()
        filterSamplesList(sampleSet.toSet, keep)

      case other => fatal(
        s"""method 'filterSamplesTable' requires a table with key [ $sSignature ]
           |  Found key [ ${ other.mkString(", ") } ]""".stripMargin)
    }
  }

  /**
    * Filter variants using the Hail expression language.
    *
    * @param filterExpr filter expression
    * @param keep       keep variants where filterExpr evaluates to true
    * @return
    */
  def filterVariantsExpr(filterExpr: String, keep: Boolean = true): VariantSampleMatrix = {
    var filterAST = Parser.expr.parse(filterExpr)
    if (!keep)
      filterAST = Apply(filterAST.getPos, "!", Array(filterAST))
    copyAST(ast = FilterVariants(ast, filterAST))
  }

  def filterVariantsList(variants: java.util.ArrayList[Annotation], keep: Boolean): VariantSampleMatrix =
    filterVariantsList(variants.asScala.toSet, keep)

  def filterVariantsList(variants: Set[Annotation], keep: Boolean): VariantSampleMatrix = {
    if (keep) {
      val partitionVariants = variants
        .groupBy(v => rdd.orderedPartitioner.getPartition(v))
        .toArray
        .sortBy(_._1)

      val adjRDD = new AdjustedPartitionsRDD[RowT](rdd,
        partitionVariants.map { case (oldPart, variantsSet) =>
          Array(Adjustment[RowT](oldPart,
            _.filter { case (v, _) =>
              variantsSet.contains(v)
            }))
        })

      val adjRangeBounds: Array[Annotation] =
        if (partitionVariants.isEmpty)
          Array.empty
        else
          partitionVariants.init.map { case (oldPart, _) =>
            rdd.orderedPartitioner.rangeBounds(oldPart)
          }

      val adjPart = OrderedPartitioner[Annotation, Annotation](adjRangeBounds, partitionVariants.length)
      copy(rdd = OrderedRDD(adjRDD, adjPart))
    } else {
      val variantsBc = hc.sc.broadcast(variants)
      filterVariants { case (v, _, _) => !variantsBc.value.contains(v) }
    }
  }

  def filterVariantsTable(kt: KeyTable, keep: Boolean = true): VariantSampleMatrix = {
    val keyFields = kt.keyFields.map(_.typ)
    val filt = keyFields match {
      case Array(`vSignature`) =>
        val variantRDD = kt.keyedRDD()
          .map { case (k, v) => (k.getAs[Annotation](0), ()) }
          .filter(_._1 != null)
          .orderedRepartitionBy(rdd.orderedPartitioner)

        rdd.orderedLeftJoinDistinct(variantRDD)
          .filter { case (_, (_, o)) => Filter.keepThis(o.isDefined, keep) }
          .mapValues { case (vags, _) => vags }

      case Array(vSignature.partitionKey) =>
        val locusRDD = kt.keyedRDD()
          .map { case (k, v) => (k.getAs[Annotation](0), ()) }
          .filter(_._1 != null)
          .orderedRepartitionBy(rdd.orderedPartitioner.projectToPartitionKey())

        OrderedRDD[Annotation, Annotation, (Annotation, Iterable[Annotation])](rdd.mapMonotonic(kOk.orderedProject, { case (v, vags) => (v, vags) })
          .orderedLeftJoinDistinct(locusRDD)
          .filter { case (_, (_, o)) => Filter.keepThis(o.isDefined, keep) }
          .map { case (_, ((v, vags), _)) => (v, vags) },
          rdd.orderedPartitioner)

      case Array(TInterval(_, _)) if vSignature.isInstanceOf[TVariant] =>
        val partBc = sparkContext.broadcast(rdd.orderedPartitioner)
        val intRDD = kt.keyedRDD()
          .map { case (k, _) => k.getAs[Interval[Locus]](0) }
          .filter(_ != null)
          .flatMap { interval =>
            val start = partBc.value.getPartitionT(interval.start.asInstanceOf[Annotation])
            val end = partBc.value.getPartitionT(interval.end.asInstanceOf[Annotation])
            (start to end).view.map(i => (i, interval))
          }

        val overlapPartitions = intRDD.keys.collectAsSet().toArray.sorted
        val partitionMap = overlapPartitions.zipWithIndex.toMap
        val leftTotalPartitions = rdd.partitions.length

        if (keep) {
          if (overlapPartitions.length < rdd.partitions.length)
            info(s"filtered to ${ overlapPartitions.length } of ${ leftTotalPartitions } partitions")


          val zipRDD = intRDD.partitionBy(new Partitioner {
            def getPartition(key: Any): Int = partitionMap(key.asInstanceOf[Int])

            def numPartitions: Int = overlapPartitions.length
          }).values

          rdd.subsetPartitions(overlapPartitions)
            .zipPartitions(zipRDD, preservesPartitioning = true) { case (it, intervals) =>
              val itree = IntervalTree.apply[Locus](intervals.toArray)
              it.filter { case (v, _) => itree.contains(v.asInstanceOf[Variant].locus) }
            }
        } else {
          val zipRDD = intRDD.partitionBy(new Partitioner {
            def getPartition(key: Any): Int = key.asInstanceOf[Int]

            def numPartitions: Int = leftTotalPartitions
          }).values

          rdd.zipPartitions(zipRDD, preservesPartitioning = true) { case (it, intervals) =>
            val itree = IntervalTree.apply[Locus](intervals.toArray)
            it.filter { case (v, _) => !itree.contains(v.asInstanceOf[Variant].locus) }
          }
        }

      case _ => fatal(
        s"""method 'filterVariantsTable' requires a table with one of the following keys:
           |  [ $vSignature ]
           |  [ Locus ]
           |  [ Interval ]
           |  Found [ ${ keyFields.mkString(", ") } ]""".stripMargin)
    }

    copy(rdd = filt.asOrderedRDD)
  }

  def sparkContext: SparkContext = hc.sc

  def hadoopConf: hadoop.conf.Configuration = hc.hadoopConf

  def head(n: Long): VariantSampleMatrix = {
    if (n < 0)
      fatal(s"n must be non-negative! Found `$n'.")
    copy(rdd = rdd.head(n))
  }

  /**
    *
    * @param computeMafExpr An expression for the minor allele frequency of the current variant, `v', given
    *                       the variant annotations `va'. If unspecified, MAF will be estimated from the dataset
    * @param bounded        Allows the estimations for Z0, Z1, Z2, and PI_HAT to take on biologically-nonsense values
    *                       (e.g. outside of [0,1]).
    * @param minimum        Sample pairs with a PI_HAT below this value will not be included in the output. Must be in [0,1]
    * @param maximum        Sample pairs with a PI_HAT above this value will not be included in the output. Must be in [0,1]
    */
  def ibd(computeMafExpr: Option[String] = None, bounded: Boolean = true,
    minimum: Option[Double] = None, maximum: Option[Double] = None): KeyTable = {
    require(wasSplit)
    IBD(this, computeMafExpr, bounded, minimum, maximum)
  }

  def insertSA(sig: Type, args: String*): (Type, Inserter) = insertSA(sig, args.toList)

  def insertSA(sig: Type, path: List[String]): (Type, Inserter) = saSignature.insert(sig, path)

  def insertVA(sig: Type, args: String*): (Type, Inserter) = insertVA(sig, args.toList)

  def insertVA(sig: Type, path: List[String]): (Type, Inserter) = {
    vaSignature.insert(sig, path)
  }

  def insertIntoRow[PC](makePartitionContext: () => PC)(typeToInsert: Type, path: List[String],
    inserter: (PC, RegionValue, RegionValueBuilder) => Unit): VariantSampleMatrix = {
    val newRDD2 = rdd2.insert(makePartitionContext)(typeToInsert, path, inserter)
    copy2(rdd2 = newRDD2,
      // don't need to update vSignature, insert can't change the keys
      vaSignature = newRDD2.typ.rowType.fieldType(2),
      genotypeSignature = newRDD2.typ.rowType.fieldType(3).asInstanceOf[TArray].elementType)
  }

  /**
    *
    * @param right right-hand dataset with which to join
    */
  def join(right: VariantSampleMatrix): VariantSampleMatrix = {
    if (wasSplit != right.wasSplit) {
      warn(
        s"""cannot join split and unsplit datasets
           |  left was split: ${ wasSplit }
           |  light was split: ${ right.wasSplit }""".stripMargin)
    }

    if (genotypeSignature != right.genotypeSignature) {
      fatal(
        s"""cannot join datasets with different genotype schemata
           |  left genotype schema: @1
           |  right genotype schema: @2""".stripMargin,
        genotypeSignature.toPrettyString(compact = true, printAttrs = true),
        right.genotypeSignature.toPrettyString(compact = true, printAttrs = true))
    }

    if (saSignature != right.saSignature) {
      fatal(
        s"""cannot join datasets with different sample schemata
           |  left sample schema: @1
           |  right sample schema: @2""".stripMargin,
        saSignature.toPrettyString(compact = true, printAttrs = true),
        right.saSignature.toPrettyString(compact = true, printAttrs = true))
    }

    if (vSignature != right.vSignature) {
      fatal(
        s"""cannot join datasets with different variant schemata
           |  left variant schema: @1
           |  right variant schema: @2""".stripMargin,
        vSignature.toPrettyString(compact = true, printAttrs = true),
        right.vSignature.toPrettyString(compact = true, printAttrs = true))
    }

    val newSampleIds = sampleIds ++ right.sampleIds
    val duplicates = newSampleIds.duplicates()
    if (duplicates.nonEmpty)
      fatal("duplicate sample IDs: @1", duplicates)

    val localRowType = rowType
    val localLeftSamples = nSamples
    val localRightSamples = right.nSamples
    val tgs = rowType.fieldType(3).asInstanceOf[TArray]

    val joined = rdd2.orderedJoinDistinct(right.rdd2, "inner").mapPartitions({ it =>
      val rvb = new RegionValueBuilder()
      val rv2 = RegionValue()

      it.map { jrv =>
        val lrv = jrv.rvLeft
        val rrv = jrv.rvRight

        rvb.set(lrv.region)
        rvb.start(localRowType)
        rvb.startStruct()
        rvb.addField(localRowType, lrv, 0) // l
        rvb.addField(localRowType, lrv, 1) // v
        rvb.addField(localRowType, lrv, 2) // va
        rvb.startArray(localLeftSamples + localRightSamples)

        val gsLeftOffset = localRowType.loadField(lrv.region, lrv.offset, 3) // left gs
        val gsLeftLength = tgs.loadLength(lrv.region, gsLeftOffset)
        assert(gsLeftLength == localLeftSamples)

        val gsRightOffset = localRowType.loadField(rrv.region, rrv.offset, 3) // right gs
        val gsRightLength = tgs.loadLength(rrv.region, gsRightOffset)
        assert(gsRightLength == localRightSamples)

        var i = 0
        while (i < localLeftSamples) {
          rvb.addElement(tgs, lrv.region, gsLeftOffset, i)
          i += 1
        }

        i = 0
        while (i < localRightSamples) {
          rvb.addElement(tgs, rrv.region, gsRightOffset, i)
          i += 1
        }

        rvb.endArray()
        rvb.endStruct()
        rv2.set(lrv.region, rvb.end())
        rv2
      }
    }, preservesPartitioning = true)

    copy2(sampleIds = newSampleIds,
      sampleAnnotations = sampleAnnotations ++ right.sampleAnnotations,
      rdd2 = OrderedRDD2(rdd2.typ, rdd2.orderedPartitioner, joined))
  }

  def makeKT(variantCondition: String, genotypeCondition: String, keyNames: Array[String] = Array.empty, seperator: String = "."): KeyTable = {
    requireColKeyString("make table")

    val vSymTab = Map(
      "v" -> (0, vSignature),
      "va" -> (1, vaSignature))
    val vEC = EvalContext(vSymTab)
    val vA = vEC.a

    val (vNames, vTypes, vf) = Parser.parseNamedExprs(variantCondition, vEC)

    val gSymTab = Map(
      "v" -> (0, vSignature),
      "va" -> (1, vaSignature),
      "s" -> (2, sSignature),
      "sa" -> (3, saSignature),
      "g" -> (4, genotypeSignature))
    val gEC = EvalContext(gSymTab)
    val gA = gEC.a

    val (gNames, gTypes, gf) = Parser.parseNamedExprs(genotypeCondition, gEC)

    val sig = TStruct(((vNames, vTypes).zipped ++
      stringSampleIds.flatMap { s =>
        (gNames, gTypes).zipped.map { case (n, t) =>
          (if (n.isEmpty)
            s
          else
            s + seperator + n, t)
        }
      }).toSeq: _*)

    val localNSamples = nSamples
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    KeyTable(hc,
      rdd.mapPartitions { it =>
        val n = vNames.length + gNames.length * localNSamples

        it.map { case (v, (va, gs)) =>
          val a = new Array[Any](n)

          var j = 0
          vEC.setAll(v, va)
          vf().foreach { x =>
            a(j) = x
            j += 1
          }

          gs.iterator.zipWithIndex.foreach { case (g, i) =>
            val s = localSampleIdsBc.value(i)
            val sa = localSampleAnnotationsBc.value(i)
            gEC.setAll(v, va, s, sa, g)
            gf().foreach { x =>
              a(j) = x
              j += 1
            }
          }

          assert(j == n)
          Row.fromSeq(a)
        }
      },
      sig,
      keyNames)
  }

  def mapWithKeys[U](f: (Annotation, Annotation, Annotation) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc

    rdd
      .flatMap { case (v, (va, gs)) =>
        localSampleIdsBc.value.lazyMapWith[Annotation, U](gs,
          (s, g) => f(v, s, g))
      }
  }

  def mapAnnotations(newVASignature: Type, f: (Annotation, Annotation, Iterable[Annotation]) => Annotation): VariantSampleMatrix = {
    val localRowType = rowType
    insertIntoRow(() => new UnsafeRow(localRowType))(newVASignature, List("va"), { case (ur, rv, rvb) =>
      ur.set(rv)
      val v = ur.getAs[Annotation](1)
      val va = ur.get(2)
      val gs = ur.getAs[Iterable[Annotation]](3)
      rvb.addAnnotation(newVASignature, f(v, va, gs))
    })
  }

  def mapPartitionsWithAll[U](f: Iterator[(Annotation, Annotation, Annotation, Annotation, Annotation)] => Iterator[U])
    (implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    rdd.mapPartitions { it =>
      f(it.flatMap { case (v, (va, gs)) =>
        localSampleIdsBc.value.lazyMapWith2[Annotation, Annotation, (Annotation, Annotation, Annotation, Annotation, Annotation)](
          localSampleAnnotationsBc.value, gs, { case (s, sa, g) => (v, va, s, sa, g) })
      })
    }
  }

  def mapValues[U >: Null](newGSignature: Type, f: (Annotation) => U)(implicit uct: ClassTag[U]): VariantSampleMatrix = {
    mapValuesWithAll(newGSignature, (v, va, s, sa, g) => f(g))
  }

  def mapValuesWithAll[U >: Null](newGSignature: Type, f: (Annotation, Annotation, Annotation, Annotation, Annotation) => U)
    (implicit uct: ClassTag[U]): VariantSampleMatrix = {
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc
    copy(genotypeSignature = newGSignature,
      rdd = rdd.mapValuesWithKey { case (v, (va, gs)) =>
        (va, localSampleIdsBc.value.lazyMapWith2[Annotation, Annotation, U](
          localSampleAnnotationsBc.value, gs, { case (s, sa, g) => f(v, va, s, sa, g) }))
      })
  }

  def queryGenotypes(expr: String): (Annotation, Type) = {
    val qv = queryGenotypes(Array(expr))
    assert(qv.length == 1)
    qv.head
  }

  def queryGenotypes(exprs: Array[String]): Array[(Annotation, Type)] = {
    val aggregationST = Map(
      "global" -> (0, globalSignature),
      "g" -> (1, genotypeSignature),
      "v" -> (2, vSignature),
      "va" -> (3, vaSignature),
      "s" -> (4, sSignature),
      "sa" -> (5, saSignature))
    val ec = EvalContext(Map(
      "global" -> (0, globalSignature),
      "gs" -> (1, TAggregable(genotypeSignature, aggregationST))))

    val ts = exprs.map(e => Parser.parseExpr(e, ec))

    val localGlobalAnnotation = globalAnnotation
    val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[Annotation](ec, { case (ec, g) =>
      ec.set(1, g)
    })

    val globalBc = sparkContext.broadcast(globalAnnotation)
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    val result = rdd.mapPartitions { it =>
      val zv = zVal.map(_.copy())
      ec.set(0, globalBc.value)
      it.foreach { case (v, (va, gs)) =>
        var i = 0
        ec.set(2, v)
        ec.set(3, va)
        gs.foreach { g =>
          ec.set(4, localSampleIdsBc.value(i))
          ec.set(5, localSampleAnnotationsBc.value(i))
          seqOp(zv, g)
          i += 1
        }
      }
      Iterator(zv)
    }.fold(zVal.map(_.copy()))(combOp)
    resOp(result)

    ec.set(0, localGlobalAnnotation)
    ts.map { case (t, f) => (f(), t) }
  }

  def queryGlobal(path: String): (Type, Annotation) = {
    val st = Map(Annotation.GLOBAL_HEAD -> (0, globalSignature))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parseExpr(path, ec)

    val f2: Annotation => Any = { annotation =>
      a(0) = annotation
      f()
    }

    (t, f2(globalAnnotation))
  }

  def querySA(code: String): (Type, Querier) = {

    val st = Map(Annotation.SAMPLE_HEAD -> (0, saSignature))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parseExpr(code, ec)

    val f2: Annotation => Any = { annotation =>
      a(0) = annotation
      f()
    }

    (t, f2)
  }

  def querySamples(expr: String): (Annotation, Type) = {
    val qs = querySamples(Array(expr))
    assert(qs.length == 1)
    qs.head
  }

  def querySamples(exprs: Array[String]): Array[(Annotation, Type)] = {
    val aggregationST = Map(
      "global" -> (0, globalSignature),
      "s" -> (1, sSignature),
      "sa" -> (2, saSignature))
    val ec = EvalContext(Map(
      "global" -> (0, globalSignature),
      "samples" -> (1, TAggregable(sSignature, aggregationST))))

    val ts = exprs.map(e => Parser.parseExpr(e, ec))

    val localGlobalAnnotation = globalAnnotation
    val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[(Annotation, Annotation)](ec, { case (ec, (s, sa)) =>
      ec.setAll(localGlobalAnnotation, s, sa)
    })

    val results = sampleIdsAndAnnotations
      .aggregate(zVal)(seqOp, combOp)
    resOp(results)
    ec.set(0, localGlobalAnnotation)

    ts.map { case (t, f) => (f(), t) }
  }

  def queryVA(code: String): (Type, Querier) = {

    val st = Map(Annotation.VARIANT_HEAD -> (0, vaSignature))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parseExpr(code, ec)

    val f2: Annotation => Any = { annotation =>
      a(0) = annotation
      f()
    }

    (t, f2)
  }

  def queryVariants(expr: String): (Annotation, Type) = {
    val qv = queryVariants(Array(expr))
    assert(qv.length == 1)
    qv.head
  }

  def queryVariants(exprs: Array[String]): Array[(Annotation, Type)] = {

    val aggregationST = Map(
      "global" -> (0, globalSignature),
      "v" -> (1, vSignature),
      "va" -> (2, vaSignature))
    val ec = EvalContext(Map(
      "global" -> (0, globalSignature),
      "variants" -> (1, TAggregable(vSignature, aggregationST))))

    val ts = exprs.map(e => Parser.parseExpr(e, ec))

    val localGlobalAnnotation = globalAnnotation
    val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[(Annotation, Annotation)](ec, { case (ec, (v, va)) =>
      ec.setAll(localGlobalAnnotation, v, va)
    })

    val result = variantsAndAnnotations
      .treeAggregate(zVal)(seqOp, combOp, depth = treeAggDepth(hc, nPartitions))
    resOp(result)

    ec.setAll(localGlobalAnnotation)
    ts.map { case (t, f) => (f(), t) }
  }


  def queryGA(code: String): (Type, Querier) = {
    val st = Map(Annotation.GENOTYPE_HEAD -> (0, genotypeSignature))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parseExpr(code, ec)

    val f2: Annotation => Any = { annotation =>
      a(0) = annotation
      f()
    }

    (t, f2)
  }

  def reorderSamples(newIds: java.util.ArrayList[Annotation]): VariantSampleMatrix =
    reorderSamples(newIds.asScala.toArray)

  def reorderSamples(newIds: Array[Annotation]): VariantSampleMatrix = {
    requireUniqueSamples("reorder_samples")

    val oldOrder = sampleIds.zipWithIndex.toMap
    val newOrder = newIds.zipWithIndex.toMap

    val newIndices = new Array[Int](nSamples)
    val missingSamples = mutable.Set[Annotation]()
    val notInDataset = mutable.Set[Annotation]()

    oldOrder.outerJoin(newOrder).foreach { case (s, (oldIdx, newIdx)) =>
      ((oldIdx, newIdx): @unchecked) match {
        case (Some(i), Some(j)) => newIndices(i) = j
        case (Some(i), None) => missingSamples += s
        case (None, Some(j)) => notInDataset += s
      }
    }

    if (missingSamples.nonEmpty)
      fatal(s"Found ${ missingSamples.size } ${ plural(missingSamples.size, "sample ID") } in dataset that are not in new ordering:\n  " +
        s"@1", missingSamples.truncatable("\n  "))

    if (notInDataset.nonEmpty)
      fatal(s"Found ${ notInDataset.size } ${ plural(notInDataset.size, "sample ID") } in new ordering that are not in dataset:\n  " +
        s"@1", notInDataset.truncatable("\n  "))

    val newAnnotations = new Array[Annotation](nSamples)
    sampleAnnotations.zipWithIndex.foreach { case (sa, idx) =>
      newAnnotations(newIndices(idx)) = sa
    }

    val nSamplesLocal = nSamples

    val reorderedRdd = rdd.mapPartitions({ it =>
      it.map { case (v, (va, gs)) =>
        val reorderedGs = new Array[Annotation](nSamplesLocal)
        val gsIt = gs.iterator

        var i = 0
        while (gsIt.hasNext) {
          reorderedGs(newIndices(i)) = gsIt.next
          i += 1
        }

        (v, (va, reorderedGs.toIterable))
      }
    }, preservesPartitioning = true).asOrderedRDD

    copy(rdd = reorderedRdd, sampleIds = newIds, sampleAnnotations = newAnnotations)
  }

  def renameSamples(newIds: java.util.ArrayList[Annotation]): VariantSampleMatrix =
    renameSamples(newIds.asScala.toArray)

  def renameSamples(newIds: Array[Annotation]): VariantSampleMatrix = {
    if (newIds.length != sampleIds.length)
      fatal(s"dataset contains $nSamples samples, but new ID list contains ${ newIds.length }")
    copy2(sampleIds = newIds)
  }

  def renameSamples(mapping: java.util.Map[Annotation, Annotation]): VariantSampleMatrix =
    renameSamples(mapping.asScala.toMap)

  def renameSamples(mapping: Map[Annotation, Annotation]): VariantSampleMatrix = {
    val newSampleIds = sampleIds.map(s => mapping.getOrElse(s, s))
    copy2(sampleIds = newSampleIds)
  }

  def renameDuplicates(): VariantSampleMatrix = {
    requireColKeyString("rename duplicates")
    val (newIds, duplicates) = mangle(stringSampleIds.toArray)
    if (duplicates.nonEmpty)
      info(s"Renamed ${ duplicates.length } duplicate ${ plural(duplicates.length, "sample ID") }. " +
        s"Mangled IDs as follows:\n  @1", duplicates.map { case (pre, post) => s""""$pre" => "$post"""" }.truncatable("\n  "))
    else
      info(s"No duplicate sample IDs found.")
    val (newSchema, ins) = insertSA(TString(), "originalID")
    val newAnnotations = sampleIdsAndAnnotations.map { case (s, sa) => ins(sa, s) }
    copy2(sampleIds = newIds, saSignature = newSchema, sampleAnnotations = newAnnotations)
  }

  def same(that: VariantSampleMatrix, tolerance: Double = utils.defaultTolerance): Boolean = {
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

    val localSampleIds = sampleIds
    val vaSignatureBc = sparkContext.broadcast(vaSignature)
    val gSignatureBc = sparkContext.broadcast(genotypeSignature)

    metadataSame &&
      rdd.zipPartitions(that.rdd.orderedRepartitionBy(rdd.orderedPartitioner)) { (it1, it2) =>
        var partSame = true
        while (it1.hasNext && it2.hasNext) {
          val (v1, (va1, gs1)) = it1.next()
          val (v2, (va2, gs2)) = it2.next()

          if (v1 != v2 && partSame) {
            println(
              s"""variants were not the same:
                 |  $v1
                 |  $v2
               """.stripMargin)
            partSame = false
          }
          val annotationsSame = vaSignatureBc.value.valuesSimilar(va1, va2, tolerance)
          if (!annotationsSame && partSame) {
            println(
              s"""at variant `$v1', annotations were not the same:
                 |  $va1
                 |  $va2""".stripMargin)
            partSame = false
          }
          val genotypesSame = (localSampleIds, gs1, gs2).zipped.forall { case (s, g1, g2) =>
            val gSame = gSignatureBc.value.valuesSimilar(g1, g2, tolerance)
            if (!gSame && !partSame) {
              println(
                s"""at $v1, $s, genotypes were not the same:
                   |  $g1
                   |  $g2
                   """.stripMargin)
            }
            gSame
          }
        }

        if ((it1.hasNext || it2.hasNext) && partSame) {
          println("partition has different number of variants")
          partSame = false
        }

        Iterator(partSame)
      }.forall(t => t)
  }

  def sampleEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalSignature),
      "s" -> (1, sSignature),
      "sa" -> (2, saSignature),
      "g" -> (3, genotypeSignature),
      "v" -> (4, vSignature),
      "va" -> (5, vaSignature))
    EvalContext(Map(
      "global" -> (0, globalSignature),
      "s" -> (1, sSignature),
      "sa" -> (2, saSignature),
      "gs" -> (3, TAggregable(genotypeSignature, aggregationST))))
  }

  def sampleAnnotationsSimilar(that: VariantSampleMatrix, tolerance: Double = utils.defaultTolerance): Boolean = {
    require(saSignature == that.saSignature)
    sampleAnnotations.zip(that.sampleAnnotations)
      .forall { case (s1, s2) => saSignature.valuesSimilar(s1, s2, tolerance)
      }
  }

  def sampleVariants(fraction: Double, seed: Int = 1): VariantSampleMatrix = {
    require(fraction > 0 && fraction < 1, s"the 'fraction' parameter must fall between 0 and 1, found $fraction")
    copy2(rdd2 = rdd2.sample(withReplacement = false, fraction, seed))
  }

  def copy(rdd: OrderedRDD[Annotation, Annotation, (Annotation, Iterable[Annotation])] = rdd,
    sampleIds: IndexedSeq[Annotation] = sampleIds,
    sampleAnnotations: IndexedSeq[Annotation] = sampleAnnotations,
    globalAnnotation: Annotation = globalAnnotation,
    sSignature: Type = sSignature,
    saSignature: Type = saSignature,
    vSignature: Type = vSignature,
    vaSignature: Type = vaSignature,
    globalSignature: Type = globalSignature,
    genotypeSignature: Type = genotypeSignature,
    wasSplit: Boolean = wasSplit): VariantSampleMatrix =
    VariantSampleMatrix(hc,
      VSMMetadata(sSignature, saSignature, vSignature, vaSignature, globalSignature, genotypeSignature, wasSplit),
      VSMLocalValue(globalAnnotation, sampleIds, sampleAnnotations), rdd)

  def copyLegacy[RK, T](rdd: RDD[(RK, (Annotation, Iterable[T]))],
    sampleIds: IndexedSeq[Annotation] = sampleIds,
    sampleAnnotations: IndexedSeq[Annotation] = sampleAnnotations,
    globalAnnotation: Annotation = globalAnnotation,
    sSignature: Type = sSignature,
    saSignature: Type = saSignature,
    vSignature: Type = vSignature,
    vaSignature: Type = vaSignature,
    globalSignature: Type = globalSignature,
    genotypeSignature: Type = genotypeSignature,
    wasSplit: Boolean = wasSplit): VariantSampleMatrix =
    VariantSampleMatrix.fromLegacy(hc,
      VSMMetadata(sSignature, saSignature, vSignature, vaSignature, globalSignature, genotypeSignature, wasSplit),
      VSMLocalValue(globalAnnotation, sampleIds, sampleAnnotations),
      rdd)

  def copy2(rdd2: OrderedRDD2 = rdd2,
    sampleIds: IndexedSeq[Annotation] = sampleIds,
    sampleAnnotations: IndexedSeq[Annotation] = sampleAnnotations,
    globalAnnotation: Annotation = globalAnnotation,
    sSignature: Type = sSignature,
    saSignature: Type = saSignature,
    vSignature: Type = vSignature,
    vaSignature: Type = vaSignature,
    globalSignature: Type = globalSignature,
    genotypeSignature: Type = genotypeSignature,
    wasSplit: Boolean = wasSplit): VariantSampleMatrix =
    new VariantSampleMatrix(hc,
      VSMMetadata(sSignature, saSignature, vSignature, vaSignature, globalSignature, genotypeSignature, wasSplit),
      VSMLocalValue(globalAnnotation, sampleIds, sampleAnnotations), rdd2)

  def copyAST(ast: MatrixIR = ast,
    sSignature: Type = sSignature,
    saSignature: Type = saSignature,
    vSignature: Type = vSignature,
    vaSignature: Type = vaSignature,
    globalSignature: Type = globalSignature,
    genotypeSignature: Type = genotypeSignature,
    wasSplit: Boolean = wasSplit): VariantSampleMatrix =
    new VariantSampleMatrix(hc,
      VSMMetadata(sSignature, saSignature, vSignature, vaSignature, globalSignature, genotypeSignature, wasSplit),
      ast)

  def samplesKT(): KeyTable = {
    KeyTable(hc, sparkContext.parallelize(sampleIdsAndAnnotations)
      .map { case (s, sa) =>
        Row(s, sa)
      },
      TStruct(
        "s" -> sSignature,
        "sa" -> saSignature),
      Array("s"))
  }

  def storageLevel: String = rdd2.getStorageLevel2.toReadableString()

  def setVaAttributes(path: String, kv: Map[String, String]): VariantSampleMatrix = {
    setVaAttributes(Parser.parseAnnotationRoot(path, Annotation.VARIANT_HEAD), kv)
  }

  def setVaAttributes(path: List[String], kv: Map[String, String]): VariantSampleMatrix = {
    vaSignature match {
      case t: TStruct => copy2(vaSignature = t.setFieldAttributes(path, kv))
      case t => fatal(s"Cannot set va attributes to ${ path.mkString(".") } since va is not a Struct.")
    }
  }

  def deleteVaAttribute(path: String, attribute: String): VariantSampleMatrix = {
    deleteVaAttribute(Parser.parseAnnotationRoot(path, Annotation.VARIANT_HEAD), attribute)
  }

  def deleteVaAttribute(path: List[String], attribute: String): VariantSampleMatrix = {
    vaSignature match {
      case t: TStruct => copy2(vaSignature = t.deleteFieldAttribute(path, attribute))
      case t => fatal(s"Cannot delete va attributes from ${ path.mkString(".") } since va is not a Struct.")
    }
  }

  override def toString =
    s"VariantSampleMatrix(metadata=$metadata, rdd=$rdd, sampleIds=$sampleIds, nSamples=$nSamples, vaSignature=$vaSignature, saSignature=$saSignature, globalSignature=$globalSignature, sampleAnnotations=$sampleAnnotations, sampleIdsAndAnnotations=$sampleIdsAndAnnotations, globalAnnotation=$globalAnnotation, wasSplit=$wasSplit)"

  def nSamples: Int = sampleIds.length

  def typecheck() {
    var foundError = false
    if (!globalSignature.typeCheck(globalAnnotation)) {
      warn(
        s"""found violation in global annotation
           |Schema: ${ globalSignature.toPrettyString() }
           |Annotation: ${ Annotation.printAnnotation(globalAnnotation) }""".stripMargin)
    }

    sampleIdsAndAnnotations.find { case (_, sa) => !saSignature.typeCheck(sa) }
      .foreach { case (s, sa) =>
        foundError = true
        warn(
          s"""found violation in sample annotations for sample $s
             |Schema: ${ saSignature.toPrettyString() }
             |Annotation: ${ Annotation.printAnnotation(sa) }""".stripMargin)
      }

    val localVaSignature = vaSignature

    variantsAndAnnotations.find { case (_, va) => !localVaSignature.typeCheck(va) }
      .foreach { case (v, va) =>
        foundError = true
        warn(
          s"""found violation in variant annotations for variant $v
             |Schema: ${ localVaSignature.toPrettyString() }
             |Annotation: ${ Annotation.printAnnotation(va) }""".stripMargin)
      }

    if (foundError)
      fatal("found one or more type check errors")
  }

  def sampleIdsAndAnnotations: IndexedSeq[(Annotation, Annotation)] = sampleIds.zip(sampleAnnotations)

  def stringSampleIdsAndAnnotations: IndexedSeq[(Annotation, Annotation)] = stringSampleIds.zip(sampleAnnotations)

  def variantsAndAnnotations: OrderedRDD[Annotation, Annotation, Annotation] =
    rdd.mapValuesWithKey { case (v, (va, gs)) => va }

  def variantEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalSignature),
      "v" -> (1, vSignature),
      "va" -> (2, vaSignature),
      "g" -> (3, genotypeSignature),
      "s" -> (4, sSignature),
      "sa" -> (5, saSignature))
    EvalContext(Map(
      "global" -> (0, globalSignature),
      "v" -> (1, vSignature),
      "va" -> (2, vaSignature),
      "gs" -> (3, TAggregable(genotypeSignature, aggregationST))))
  }

  def variantsKT(): KeyTable = {
    val localRowType = rowType
    val typ = TStruct(
      "v" -> vSignature,
      "va" -> vaSignature)
    new KeyTable(hc, rdd2.mapPartitions { it =>
      val rv2b = new RegionValueBuilder()
      val rv2 = RegionValue()
      it.map { rv =>
        rv2b.set(rv.region)
        rv2b.start(typ)
        rv2b.startStruct()
        rv2b.addField(localRowType, rv, 1) // v
        rv2b.addField(localRowType, rv, 2) // va
        rv2b.endStruct()
        rv2.set(rv.region, rv2b.end())
        rv2
      }
    },
      typ,
      Array("v"))
  }

  def genotypeKT(): KeyTable = {
    val localNSamples = nSamples
    val localSType = sSignature
    val localSAType = saSignature
    val localRowType = rowType
    val typ = TStruct(
      "v" -> vSignature,
      "va" -> vaSignature,
      "s" -> sSignature,
      "sa" -> saSignature,
      "g" -> genotypeSignature)
    val gsType = localRowType.fieldType(3).asInstanceOf[TArray]
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc
    new KeyTable(hc, rdd2.mapPartitions { it =>
      val rv2b = new RegionValueBuilder()
      val rv2 = RegionValue()
      it.flatMap { rv =>
        val rvEnd = rv.region.end
        rv2b.set(rv.region)
        val gsOffset = localRowType.loadField(rv, 3)
        (0 until localNSamples).iterator.map { i =>
          rv.region.clear(rvEnd)
          rv2b.start(typ)
          rv2b.startStruct()
          rv2b.addField(localRowType, rv, 1) // v
          rv2b.addField(localRowType, rv, 2) // va
          rv2b.addAnnotation(localSType, localSampleIdsBc.value(i))
          rv2b.addAnnotation(localSAType, localSampleAnnotationsBc.value(i))
          rv2b.addElement(gsType, rv.region, gsOffset, i)
          rv2b.endStruct()
          rv2.set(rv.region, rv2b.end())
          rv2
        }
      }
    },
      typ,
      Array("v", "s"))
  }

  def writeMetadata(dirname: String, nPartitions: Int) {
    if (!dirname.endsWith(".vds") && !dirname.endsWith(".vds/"))
      fatal(s"output path ending in `.vds' required, found `$dirname'")

    val sqlContext = hc.sqlContext
    val hConf = hc.hadoopConf
    hConf.mkDir(dirname)

    val sampleAnnotationsJ = JArray(
      sampleIdsAndAnnotations
        .map { case (id, annotation) =>
          JObject(List(("id", JSONAnnotationImpex.exportAnnotation(id, sSignature)),
            ("annotation", JSONAnnotationImpex.exportAnnotation(annotation, saSignature))))
        }
        .toList)
    val globalJ = JSONAnnotationImpex.exportAnnotation(globalAnnotation, globalSignature)

    val metadata = VDSMetadata(
      version = VariantSampleMatrix.fileVersion,
      split = wasSplit,
      sample_schema = sSignature.toPrettyString(printAttrs = true, compact = true),
      sample_annotation_schema = saSignature.toPrettyString(printAttrs = true, compact = true),
      variant_schema = vSignature.toPrettyString(printAttrs = true, compact = true),
      variant_annotation_schema = vaSignature.toPrettyString(printAttrs = true, compact = true),
      genotype_schema = genotypeSignature.toPrettyString(printAttrs = true, compact = true),
      global_schema = globalSignature.toPrettyString(printAttrs = true, compact = true),
      sample_annotations = sampleAnnotationsJ,
      global_annotation = globalJ,
      n_partitions = nPartitions)

    hConf.writeTextFile(dirname + "/metadata.json.gz")(out =>
      Serialization.write(metadata, out))
  }

  def coalesce(k: Int, shuffle: Boolean = true): VariantSampleMatrix =
    copyLegacy(rdd = rdd.coalesce(k, shuffle = shuffle)(null))

  def persist(storageLevel: String = "MEMORY_AND_DISK"): VariantSampleMatrix = {
    val level = try {
      StorageLevel.fromString(storageLevel)
    } catch {
      case e: IllegalArgumentException =>
        fatal(s"unknown StorageLevel `$storageLevel'")
    }

    copy2(rdd2 = rdd2.persist2(level))
  }

  def cache(): VariantSampleMatrix = persist("MEMORY_ONLY")

  def unpersist(): VariantSampleMatrix = copy2(rdd2 = rdd2.unpersist2())

  def naiveCoalesce(maxPartitions: Int): VariantSampleMatrix =
    copy2(rdd2 = rdd2.naiveCoalesce(maxPartitions))

  /**
    * @param filterExpr filter expression involving v (Variant), va (variant annotations), s (sample),
    *                   sa (sample annotations), and g (genotype annotation), which returns a boolean value
    * @param keep       keep genotypes where filterExpr evaluates to true
    */
  def filterGenotypes(filterExpr: String, keep: Boolean = true): VariantSampleMatrix = {

    val symTab = Map(
      "v" -> (0, vSignature),
      "va" -> (1, vaSignature),
      "s" -> (2, sSignature),
      "sa" -> (3, saSignature),
      "g" -> (4, genotypeSignature),
      "global" -> (5, globalSignature))


    val ec = EvalContext(symTab)
    ec.set(5, globalAnnotation)
    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](filterExpr, ec)

    val localKeep = keep
    mapValuesWithAll(genotypeSignature, { (v: Annotation, va: Annotation, s: Annotation, sa: Annotation, g: Annotation) =>
      ec.setAll(v, va, s, sa, g)
      if (Filter.boxedKeepThis(f(), localKeep))
        g
      else
        null
    })
  }

  def rowType: TStruct = matrixType.rowType

  def write(dirname: String, overwrite: Boolean = false): Unit = {
    require(dirname.endsWith(".vds"), "generic dataset write paths must end in '.vds'")

    if (overwrite)
      hadoopConf.delete(dirname, recursive = true)
    else if (hadoopConf.exists(dirname))
      fatal(s"file already exists at `$dirname'")

    writeMetadata(dirname, rdd2.partitions.length)

    hadoopConf.writeTextFile(dirname + "/partitioner.json.gz") { out =>
      Serialization.write(rdd2.orderedPartitioner.toJSON, out)
    }

    rdd2.writeRows(dirname, rowType)
  }

  def linreg(ysExpr: Array[String], xExpr: String, covExpr: Array[String] = Array.empty[String], root: String = "va.linreg", variantBlockSize: Int = 16): VariantSampleMatrix = {
    LinearRegression(this, ysExpr, xExpr, covExpr, root, variantBlockSize)
  }

  def logreg(test: String,
    y: String, x: String, covariates: Array[String] = Array.empty[String],
    root: String = "va.logreg"): VariantSampleMatrix = {
    LogisticRegression(this, test, y, x, covariates, root)
  }

  def lmmreg(kinshipMatrix: KinshipMatrix,
    y: String,
    x: String,
    covariates: Array[String] = Array.empty[String],
    useML: Boolean = false,
    rootGA: String = "global.lmmreg",
    rootVA: String = "va.lmmreg",
    runAssoc: Boolean = true,
    delta: Option[Double] = None,
    sparsityThreshold: Double = 1.0,
    nEigs: Option[Int] = None,
    optDroppedVarianceFraction: Option[Double] = None): VariantSampleMatrix = {
    LinearMixedRegression(this, kinshipMatrix, y, x, covariates, useML, rootGA, rootVA,
      runAssoc, delta, sparsityThreshold, nEigs, optDroppedVarianceFraction)
  }

  def skat(variantKeys: String,
    singleKey: Boolean,
    weightExpr: String,
    y: String,
    x: String,
    covariates: Array[String] = Array.empty[String],
    logistic: Boolean = false,
    maxSize: Int = 46340, // floor(sqrt(Int.MaxValue))
    accuracy: Double = 1e-6,
    iterations: Int = 10000): KeyTable = {
    Skat(this, variantKeys, singleKey, weightExpr, y, x, covariates, logistic, maxSize, accuracy, iterations)
  }

  /**
    *
    * @param path     output path
    * @param append   append file to header
    * @param parallel export VCF in parallel using the path argument as a directory
    */
  def exportVCF(path: String, append: Option[String] = None, parallel: Boolean = false) {
    ExportVCF(this, path, append, parallel)
  }

  def minRep(leftAligned: Boolean = false): VariantSampleMatrix = {
    requireRowKeyVariant("min_rep")

    val localRowType = rowType

    def minRep1(removeLeftAligned: Boolean, removeMoving: Boolean, verifyLeftAligned: Boolean): RDD[RegionValue] = {
      rdd2.mapPartitions { it =>
        var prevLocus: Locus = null
        val rvb = new RegionValueBuilder()
        val rv2 = RegionValue()

        it.flatMap { rv =>
          val ur = new UnsafeRow(localRowType, rv.region, rv.offset)
          val v = ur.getAs[Variant](1)
          val minv = v.minRep

          var isLeftAligned = (prevLocus == null || prevLocus != v.locus) &&
            (v.locus == minv.locus)

          if (isLeftAligned && removeLeftAligned)
            None
          else if (!isLeftAligned && removeMoving)
            None
          else if (!isLeftAligned && verifyLeftAligned)
            fatal(s"found non-left aligned variant $v")
          else {
            rvb.set(rv.region)
            rvb.start(localRowType)
            rvb.startStruct()
            rvb.addAnnotation(localRowType.fieldType(0), minv.locus)
            rvb.addAnnotation(localRowType.fieldType(1), minv)
            rvb.addField(localRowType, rv, 2)
            rvb.addField(localRowType, rv, 3)
            rvb.endStruct()
            rv2.set(rv.region, rvb.end())
            Some(rv2)
          }
        }
      }
    }

    val newRDD2 =
      if (leftAligned)
        OrderedRDD2(rdd2.typ,
          rdd2.orderedPartitioner,
          minRep1(removeLeftAligned = false, removeMoving = false, verifyLeftAligned = true))
      else
        SplitMulti.unionMovedVariants(
          OrderedRDD2(rdd2.typ,
            rdd2.orderedPartitioner,
            minRep1(removeLeftAligned = false, removeMoving = true, verifyLeftAligned = false)),
          minRep1(removeLeftAligned = true, removeMoving = false, verifyLeftAligned = false))

    copy2(rdd2 = newRDD2)
  }

  def sampleQC(root: String = "sa.qc"): VariantSampleMatrix = {
    requireRowKeyVariant("sample_qc")
    SampleQC(this, root)
  }

  def variantQC(root: String = "va.qc"): VariantSampleMatrix = {
    require(wasSplit)
    requireRowKeyVariant("variant_qc")
    VariantQC(this, root)
  }

  def trioMatrix(pedigree: Pedigree, completeTrios: Boolean): VariantSampleMatrix = {
    if (!sSignature.isInstanceOf[TString])
      fatal("trio_matrix requires column keys of type String")
    requireUniqueSamples("trio_matrix")

    val filteredPedigree = pedigree.filterTo(stringSampleIds.toSet)
    val trios = if (completeTrios) filteredPedigree.completeTrios else filteredPedigree.trios
    val nTrios = trios.length

    val sampleIndices = sampleIds.zipWithIndex.toMap

    val kidIndices = Array.fill[Int](nTrios)(-1)
    val dadIndices = Array.fill[Int](nTrios)(-1)
    val momIndices = Array.fill[Int](nTrios)(-1)
    val kidIds = new Array[String](nTrios)

    val memberAnnotationType = TStruct(
      "id" -> TString(required = true),
      "annotations" -> saSignature
    )
    val newSaSignature = TStruct(
      "proband" -> memberAnnotationType,
      "father" -> memberAnnotationType,
      "mother" -> memberAnnotationType,
      "isFemale" -> TBooleanOptional,
      "famID" -> TStringOptional
    )

    val newSampleAnnotations = new Array[Annotation](nTrios)

    var i = 0
    while (i < nTrios) {
      val t = trios(i)
      val kidIndex = sampleIndices(t.kid)
      kidIndices(i) = kidIndex
      kidIds(i) = t.kid
      val kidAnnotation = Row(t.kid, sampleAnnotations(kidIndex))

      var dadAnnotation: Annotation = null
      t.dad.foreach { dad =>
        val index = sampleIndices(dad)
        dadIndices(i) = index
        dadAnnotation = Row(dad, sampleAnnotations(index))
      }

      var momAnnotation: Annotation = null
      t.mom.foreach { mom =>
        val index = sampleIndices(mom)
        momIndices(i) = index
        momAnnotation = Row(mom, sampleAnnotations(index))
      }

      val isFemale: java.lang.Boolean = (t.sex: @unchecked) match {
        case Some(Sex.Female) => true
        case Some(Sex.Male) => false
        case None => null
      }

      val famID = t.fam.orNull

      newSampleAnnotations(i) = Row(kidAnnotation, dadAnnotation, momAnnotation, isFemale, famID)
      i += 1
    }

    val gSig = genotypeSignature

    val newEntryType = TStruct(
      "proband" -> gSig,
      "father" -> gSig,
      "mother" -> gSig
    )

    val oldRowType = rowType
    val newRowType = TStruct(Array(
      rowType.fields(0),
      rowType.fields(1),
      rowType.fields(2),
      Field(rowType.fields(3).name, TArray(newEntryType), 3)
    ))

    val oldGsType = rowType.fieldType(3).asInstanceOf[TArray]

    val newRDD = rdd2.mapPartitionsPreservesPartitioning { it =>
      it.map { r =>
        val rvb = new RegionValueBuilder(MemoryBuffer())
        rvb.start(newRowType)
        rvb.startStruct()

        rvb.addField(oldRowType, r, 0)
        rvb.addField(oldRowType, r, 1)
        rvb.addField(oldRowType, r, 2)

        rvb.startArray(nTrios)
        val gsOffset = oldRowType.loadField(r, 3)

        var i = 0
        while (i < nTrios) {
          rvb.startStruct()

          // append kid element
          rvb.addElement(oldGsType, r.region, gsOffset, kidIndices(i))

          // append dad element if the dad is defined
          val dadIndex = dadIndices(i)
          if (dadIndex >= 0)
            rvb.addElement(oldGsType, r.region, gsOffset, dadIndex)
          else
            rvb.setMissing()

          // append mom element if the mom is defined
          val momIndex = momIndices(i)
          if (momIndex >= 0)
            rvb.addElement(oldGsType, r.region, gsOffset, momIndex)
          else
            rvb.setMissing()

          rvb.endStruct()
          i += 1
        }
        rvb.endArray()
        rvb.endStruct()
        rvb.end()
        rvb.result()
      }
    }.copy(typ = new OrderedRDD2Type(rdd2.typ.partitionKey, rdd2.typ.key, newRowType))

    copy2(rdd2 = newRDD,
      sampleIds = kidIds,
      sampleAnnotations = newSampleAnnotations,
      saSignature = newSaSignature,
      genotypeSignature = newEntryType)
  }

  def pcaResults(k: Int = 10, computeLoadings: Boolean = false, computeEigenvalues: Boolean = false, asArrays: Boolean = false): (KeyTable, Option[KeyTable], Option[IndexedSeq[Double]]) = {
    require(wasSplit)

    if (k < 1)
      fatal(
        s"""requested invalid number of components: $k
           |  Expect componenents >= 1""".stripMargin)

    info(s"Running PCA with $k components...")

    val (scoresmatrix, optionLoadings, optionEigenvalues) = SamplePCA(this, k, computeLoadings, computeEigenvalues, asArrays)

    val rowType = TStruct("s" -> sSignature, "pcaScores" -> SamplePCA.pcSchema(k, asArrays))
    val rowTypeBc = sparkContext.broadcast(rowType)

    val scoresrdd = sparkContext.parallelize(sampleIds.zip(scoresmatrix.rowIter.toSeq)).mapPartitions[RegionValue] { it =>
      val region = MemoryBuffer()
      val rv = RegionValue(region)
      val rvb = new RegionValueBuilder(region)
      val localRowType = rowTypeBc.value

      it.map{ case (s,v) =>
        rvb.start(localRowType)
        rvb.startStruct()
        rvb.addAnnotation(rowType.fieldType(0), s)
        if (asArrays) rvb.startArray(k) else rvb.startStruct()
        var j = 0
        while (j < k) {
          rvb.addDouble(v(j))
          j += 1
        }
        if (asArrays) rvb.endArray() else rvb.endStruct()
        rvb.endStruct()
        rv.setOffset(rvb.end())
        rv
      }
    }
    val scores = new KeyTable(hc,
      scoresrdd,
      rowType,
      Array("s"))

    (scores, optionLoadings, optionEigenvalues)
  }

  /**
    *
    * @param scoresRoot   Sample annotation path for scores (period-delimited path starting in 'sa')
    * @param k            Number of principal components
    * @param loadingsRoot Variant annotation path for site loadings (period-delimited path starting in 'va')
    * @param eigenRoot    Global annotation path for eigenvalues (period-delimited path starting in 'global'
    * @param asArrays     Store score and loading results as arrays, rather than structs
    */

  def pca(scoresRoot: String, k: Int = 10, loadingsRoot: Option[String] = None, eigenRoot: Option[String] = None,
    asArrays: Boolean = false): VariantSampleMatrix = {

    val pcSchema = SamplePCA.pcSchema(k, asArrays)

    val (scores, loadings, eigenvalues) = pcaResults(k, loadingsRoot.isDefined, eigenRoot.isDefined)
    var ret = annotateSamplesTable(scores, root = scoresRoot)

    loadings.foreach { kt =>
      ret = ret.annotateVariantsTable(kt, root = loadingsRoot.get)
    }

    eigenvalues.foreach { eig =>
      ret = ret.annotateGlobal(if (asArrays) eig else Annotation.fromSeq(eig), pcSchema, eigenRoot.get)
    }
    ret
  }
}
