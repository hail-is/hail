package is.hail.variant

import java.io.FileNotFoundException
import java.nio.ByteBuffer

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.expr._
import is.hail.io._
import is.hail.io.vcf.LoadVCF
import is.hail.keytable.{KeyTable, KeyTableMetadata}
import is.hail.methods.Aggregators.SampleFunctions
import is.hail.methods.{Aggregators, Filter, VEP}
import is.hail.sparkextras._
import is.hail.utils._
import is.hail.variant.Variant.orderedKey
import is.hail.{HailContext, utils}
import org.apache.hadoop
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{ArrayType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Partitioner, SparkContext, SparkEnv}
import org.json4s._
import org.json4s.jackson.JsonMethods.parse
import org.json4s.jackson.{JsonMethods, Serialization}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.language.{existentials, implicitConversions}
import scala.reflect.{ClassTag, classTag}

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
    dropSamples: Boolean = false, dropVariants: Boolean = false): VariantSampleMatrix[_, _, _] = {
    val (fileMetadata, nPartitions) = readFileMetadata(hc.hadoopConf, dirname)
    new VariantSampleMatrix(hc,
      fileMetadata.metadata,
      MatrixRead(hc, dirname, nPartitions, fileMetadata, dropSamples, dropVariants))
  }

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
             |  ${e.getMessage}""".stripMargin)
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

  def gen[RPK, RK, T >: Null](hc: HailContext,
    gen: VSMSubgen[RPK, RK, T])(implicit tct: ClassTag[T], kOk: OrderedKey[RPK, RK]): Gen[VariantSampleMatrix[RPK, RK, T]] =
    gen.gen(hc)

  def genGeneric(hc: HailContext): Gen[GenericDataset] =
    VSMSubgen[Annotation, Annotation, Annotation](
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
      tGen = (t: Type, v: Annotation) => t.genValue.resize(20),
      makeKOk = (vSig: Type) => vSig.orderedKey)
      .gen(hc)
}

case class VSMSubgen[RPK, RK, T >: Null](
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
  vGen: (Type) => Gen[RK],
  tGen: (Type, RK) => Gen[T],
  wasSplit: Boolean = false,
  makeKOk: (Type) => OrderedKey[RPK, RK]) {

  def gen(hc: HailContext)(implicit tct: ClassTag[T]): Gen[VariantSampleMatrix[RPK, RK, T]] =
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
      rows <- Gen.distinctBuildableOf[Array, (RK, (Annotation, Iterable[T]))](
        for (subsubsizes <- Gen.partitionSize(2);
          v <- vGen(vSig).resize(3);
          va <- vaGen(vaSig).resize(subsubsizes(0));
          ts <- Gen.buildableOfN[Array, T](nSamples, tGen(tSig, v)).resize(subsubsizes(1)))
          yield (v, (va, ts: Iterable[T]))).resize(l)
        .map(a => a.filter(_._1 != null)))
      yield {
        implicit val kOk = makeKOk(vSig)
        import kOk._

        new VariantSampleMatrix[RPK, RK, T](hc,
          VSMMetadata(sSig, saSig, vSig, vaSig, globalSig, tSig, wasSplit = wasSplit),
          VSMLocalValue(global, sampleIds, saValues),
          hc.sc.parallelize(rows, nPartitions).toOrderedRDD)
          .deduplicate()
      }
}

object VSMSubgen {
  val random = VSMSubgen[Locus, Variant, Genotype](
    sSigGen = Gen.const(TString),
    saSigGen = Type.genArb,
    vSigGen = Gen.const(TVariant(GenomeReference.GRCh37)),
    vaSigGen = Type.genArb,
    globalSigGen = Type.genArb,
    tSigGen = Gen.const(TGenotype),
    sGen = (t: Type) => Gen.identifier.map(s => s: Annotation),
    saGen = (t: Type) => t.genValue,
    vaGen = (t: Type) => t.genValue,
    globalGen = (t: Type) => t.genValue,
    vGen = (t: Type) => Variant.gen,
    tGen = (t: Type, v: Variant) => Genotype.genExtreme(v),
    makeKOk = _ => Variant.orderedKey)

  val plinkSafeBiallelic = random.copy(
    sGen = (t: Type) => Gen.plinkSafeIdentifier,
    vGen = (t: Type) => VariantSubgen.plinkCompatible.copy(nAllelesGen = Gen.const(2)).gen,
    wasSplit = true)

  val realistic = random.copy(
    tGen = (t: Type, v: Variant) => Genotype.genRealistic(v))

  val dosageGenotype = random.copy(
    tGen = (t: Type, v: Variant) => Genotype.genDosageGenotype(v))
}

class VariantSampleMatrix[RPK, RK, T >: Null](val hc: HailContext, val metadata: VSMMetadata,
  val ast: MatrixAST[RPK, RK, T])(implicit val tct: ClassTag[T]) extends JoinAnnotator {
  implicit val kOk: OrderedKey[RPK, RK] = ast.kOk

  import kOk._

  def this(hc: HailContext,
    metadata: VSMMetadata,
    localValue: VSMLocalValue,
    rdd: OrderedRDD[RPK, RK, (Annotation, Iterable[T])])(implicit tct: ClassTag[T]) =
    this(hc, metadata,
      MatrixLiteral(
        MatrixType(metadata),
        MatrixValue(localValue, rdd)))

  def this(hc: HailContext, fileMetadata: VSMFileMetadata,
    rdd: OrderedRDD[RPK, RK, (Annotation, Iterable[T])])(implicit tct: ClassTag[T]) =
    this(hc, fileMetadata.metadata, fileMetadata.localValue, rdd)

  def requireSampleTString(method: String) {
    if (sSignature != TString)
      fatal(s"in $method: column key (sample) schema must be String, but found: $sSignature")
  }

  val VSMMetadata(sSignature, saSignature, vSignature, vaSignature, globalSignature, genotypeSignature, wasSplit) = metadata

  lazy val value: MatrixValue[RPK, RK, T] = {
    val opt = MatrixAST.optimize(ast)
    opt.execute(hc)
  }

  lazy val MatrixValue(VSMLocalValue(globalAnnotation, sampleIds, sampleAnnotations), rdd) = value

  def stringSampleIds: IndexedSeq[String] = {
    assert(sSignature == TString)
    sampleIds.map(_.asInstanceOf[String])
  }

  def stringSampleIdSet: Set[String] = stringSampleIds.toSet

  type RowT = (RK, (Annotation, Iterable[T]))

  lazy val sampleIdsBc = sparkContext.broadcast(sampleIds)

  lazy val sampleAnnotationsBc = sparkContext.broadcast(sampleAnnotations)

  def unsafeRowRDD: RDD[UnsafeRow] = {
    val ttBc = BroadcastTypeTree(hc.sc, rowSignature)

    val localNSamples = nSamples
    rdd.mapPartitions { it =>
      val region = MemoryBuffer(8 * 1024)
      val rvb = new RegionValueBuilder(region)

      val a = new Array[Any](localNSamples)
      val t = ttBc.value.typ.asInstanceOf[TStruct]

      it.map { case (v, (va, gs)) =>

        val gis: IndexedSeq[_] = gs match {
          case gis: IndexedSeq[_] => gis
          case gs: Iterable[_] =>
            var i = 0
            val git = gs.iterator
            while (i < localNSamples) {
              val g = git.next()
              a(i) = g
              i += 1
            }
            assert(!git.hasNext)
            a
        }
        region.clear()
        rvb.start(t)
        rvb.startStruct()
        rvb.addAnnotation(t.fields(0).typ, v)
        rvb.addAnnotation(t.fields(1).typ, va)
        rvb.addAnnotation(t.fields(2).typ, gis)
        rvb.endStruct()
        val offset = rvb.end()
        new UnsafeRow(ttBc, region.copy(), offset)
      }
    }
  }

  def requireUniqueSamples(method: String) {
    val dups = sampleIds.counter().filter(_._2 > 1).toArray
    if (dups.nonEmpty)
      fatal(s"Method '$method' does not support duplicate sample IDs. Duplicates:" +
        s"\n  @1", dups.sortBy(-_._2).map { case (id, count) => s"""($count) "$id"""" }.truncatable("\n  "))
  }

  def collect(): Array[UnsafeRow] = unsafeRowRDD.collect()

  def take(n: Int): Array[UnsafeRow] = unsafeRowRDD.take(n)

  def aggregateBySamplePerVariantKey(keyName: String, variantKeysVA: String, aggExpr: String, singleKey: Boolean = false): KeyTable = {

    val (keysType, keysQuerier) = queryVA(variantKeysVA)

    val (keyType, keyedRdd) =
      if (singleKey) {
        (keysType, rdd.flatMap { case (v, (va, gs)) => Option(keysQuerier(va)).map(key => (key, (v, va, gs))) })
      } else {
        val keyType = keysType match {
          case TArray(e) => e
          case TSet(e) => e
          case _ => fatal(s"With single_key=False, variant keys must be of type Set[T] or Array[T], got $keysType")
        }
        (keyType, rdd.flatMap { case (v, (va, gs)) =>
          Option(keysQuerier(va).asInstanceOf[Iterable[_]]).getOrElse(Iterable.empty).map(key => (key, (v, va, gs)))
        })
      }

    val SampleFunctions(zero, seqOp, combOp, resultOp, resultType) = Aggregators.makeSampleFunctions[RPK, RK, T](this, aggExpr)

    val ktRDD = keyedRdd
      .aggregateByKey(zero)(seqOp, combOp)
      .map { case (key, agg) =>
        val results = resultOp(agg)
        results(0) = key
        Row.fromSeq(results)
      }

    val signature = TStruct((keyName -> keyType) +: stringSampleIds.map(id => id -> resultType): _*)

    new KeyTable(hc, ktRDD, signature, key = Array(keyName))
  }

  def aggregateBySample[U](zeroValue: U)(
    seqOp: (U, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Annotation, U)] =
    aggregateBySampleWithKeys(zeroValue)((e, v, s, g) => seqOp(e, g), combOp)

  def aggregateBySampleWithKeys[U](zeroValue: U)(
    seqOp: (U, RK, Annotation, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Annotation, U)] = {
    aggregateBySampleWithAll(zeroValue)((e, v, va, s, sa, g) => seqOp(e, v, s, g), combOp)
  }

  def aggregateBySampleWithAll[U](zeroValue: U)(
    seqOp: (U, RK, Annotation, Annotation, Annotation, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Annotation, U)] = {

    val serializer = SparkEnv.get.serializer.newInstance()
    val zeroBuffer = serializer.serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    rdd
      .mapPartitions { (it: Iterator[(RK, (Annotation, Iterable[T]))]) =>
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
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(RK, U)] =
    aggregateByVariantWithAll(zeroValue)((e, v, va, s, sa, g) => seqOp(e, g), combOp)

  def aggregateByVariantWithAll[U](zeroValue: U)(
    seqOp: (U, RK, Annotation, Annotation, Annotation, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(RK, U)] = {

    // Serialize the zero value to a byte array so that we can apply a new clone of it on each key
    val zeroBuffer = SparkEnv.get.serializer.newInstance().serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    rdd
      .mapPartitions({ (it: Iterator[(RK, (Annotation, Iterable[T]))]) =>
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
    seqOp: (U, RK, Annotation, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(RK, U)] = {
    aggregateByVariantWithAll(zeroValue)((e, v, va, s, sa, g) => seqOp(e, v, s, g), combOp)
  }

  def annotateGlobal(a: Annotation, t: Type, code: String): VariantSampleMatrix[RPK, RK, T] = {
    val (newT, i) = insertGlobal(t, Parser.parseAnnotationRoot(code, Annotation.GLOBAL_HEAD))
    copy(globalSignature = newT, globalAnnotation = i(globalAnnotation, a))
  }

  /**
    * Create and destroy global annotations with expression language.
    *
    * @param expr Annotation expression
    */
  def annotateGlobalExpr(expr: String): VariantSampleMatrix[RPK, RK, T] = {
    val ec = EvalContext(Map(
      "global" -> (0, globalSignature)))

    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Option(Annotation.GLOBAL_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]

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

    copy(globalAnnotation = ga,
      globalSignature = finalType)
  }

  def insertGlobal(sig: Type, path: List[String]): (Type, Inserter) = {
    globalSignature.insert(sig, path)
  }

  def annotateSamples(signature: Type, path: List[String], annotations: Array[Annotation]): VariantSampleMatrix[RPK, RK, T] = {
    val (t, ins) = insertSA(signature, path)

    val newAnnotations = new Array[Annotation](nSamples)

    for (i <- sampleAnnotations.indices) {
      newAnnotations(i) = ins(sampleAnnotations(i), annotations(i))
      t.typeCheck(newAnnotations(i))
    }

    copy(sampleAnnotations = newAnnotations, saSignature = t)
  }

  def annotateSamples(signature: Type, path: List[String], annotation: (Annotation) => Annotation): VariantSampleMatrix[RPK, RK, T] = {
    val (t, i) = insertSA(signature, path)
    annotateSamples(annotation, t, i)
  }

  def annotateSamplesExpr(expr: String): VariantSampleMatrix[RPK, RK, T] = {
    val ec = sampleEC

    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Some(Annotation.SAMPLE_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
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

    copy(
      sampleAnnotations = newAnnotations,
      saSignature = finalType
    )
  }

  def annotateSamples(annotations: Map[Annotation, Annotation], signature: Type, code: String): VariantSampleMatrix[RPK, RK, T] = {
    val (t, i) = insertSA(signature, Parser.parseAnnotationRoot(code, Annotation.SAMPLE_HEAD))
    annotateSamples(s => annotations.getOrElse(s, null), t, i)
  }

  def annotateSamplesTable(kt: KeyTable, vdsKey: java.util.ArrayList[String],
    root: String, expr: String, product: Boolean): VariantSampleMatrix[RPK, RK, T] =
    annotateSamplesTable(kt, if (vdsKey != null) vdsKey.asScala else null, root, expr, product)

  def annotateSamplesTable(kt: KeyTable, vdsKey: Seq[String] = null,
    root: String = null, expr: String = null, product: Boolean = false): VariantSampleMatrix[RPK, RK, T] = {

    if (root == null && expr == null || root != null && expr != null)
      fatal("method `annotateSamplesTable' requires one of `root' or 'expr', but not both")

    var (joinSignature, f): (Type, Annotation => Annotation) = kt.valueSignature.size match {
      case 0 => (TBoolean, _ != null)
      case 1 => (kt.valueSignature.fields.head.typ, x => if (x != null) x.asInstanceOf[Row].get(0) else null)
      case _ => (kt.valueSignature, identity[Annotation])
    }

    if (product) {
      joinSignature = if (joinSignature == TBoolean) TInt32 else TArray(joinSignature)
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

  def annotateSamples(annotation: (Annotation) => Annotation, newSignature: Type, inserter: Inserter): VariantSampleMatrix[RPK, RK, T] = {
    val newAnnotations = sampleIds.zipWithIndex.map { case (id, i) =>
      val sa = sampleAnnotations(i)
      val newAnnotation = inserter(sa, annotation(id))
      newSignature.typeCheck(newAnnotation)
      newAnnotation
    }

    copy(sampleAnnotations = newAnnotations, saSignature = newSignature)
  }

  def annotateVariants(otherRDD: OrderedRDD[RPK, RK, Annotation], signature: Type,
    code: String): VariantSampleMatrix[RPK, RK, T] = {
    val (newSignature, ins) = insertVA(signature, Parser.parseAnnotationRoot(code, Annotation.VARIANT_HEAD))
    annotateVariants(otherRDD, newSignature, ins, product = false)
  }

  def annotateVariantsExpr(expr: String): VariantSampleMatrix[RPK, RK, T] = {
    val localGlobalAnnotation = globalAnnotation

    val ec = variantEC
    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Some(Annotation.VARIANT_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val finalType = (paths, types).zipped.foldLeft(vaSignature) { case (vas, (ids, signature)) =>
      val (s, i) = vas.insert(signature, ids)
      inserterBuilder += i
      s
    }
    val inserters = inserterBuilder.result()

    val aggregateOption = Aggregators.buildVariantAggregations(this, ec)

    mapAnnotations(finalType, { case (v, va, gs) =>
      ec.setAll(localGlobalAnnotation, v, va)

      aggregateOption.foreach(f => f(v, va, gs))
      f().zip(inserters)
        .foldLeft(va) { case (va, (v, inserter)) =>
          inserter(va, v)
        }
    })
  }

  def annotateVariantsTable(kt: KeyTable, vdsKey: java.util.ArrayList[String],
    root: String, expr: String, product: Boolean): VariantSampleMatrix[RPK, RK, T] =
    annotateVariantsTable(kt, if (vdsKey != null) vdsKey.asScala else null, root, expr, product)

  def annotateVariantsTable(kt: KeyTable, vdsKey: Seq[String] = null,
    root: String = null, expr: String = null, product: Boolean = false): VariantSampleMatrix[RPK, RK, T] = {

    if (root == null && expr == null || root != null && expr != null)
      fatal("method `annotateVariantsTable' requires one of `root' or 'expr', but not both")

    var (joinSignature, f): (Type, Annotation => Annotation) = kt.valueSignature.size match {
      case 0 => (TBoolean, _ != null)
      case 1 => (kt.valueSignature.fields.head.typ, x => if (x != null) x.asInstanceOf[Row].get(0) else null)
      case _ => (kt.valueSignature, identity[Annotation])
    }

    if (product) {
      joinSignature = if (joinSignature == TBoolean) TInt32 else TArray(joinSignature)
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
            .map { case (k, v) => (k.getAs[RK](0), v: Annotation) }
            .toOrderedRDD(rdd.orderedPartitioner)

          annotateVariants(ord, finalType, inserter, product = product)

        case Array(vSignature.partitionKey) =>
          val ord = keyedRDD
            .map { case (k, v) => (k.asInstanceOf[Row].getAs[RPK](0), v: Annotation) }
            .toOrderedRDD(rdd.orderedPartitioner.projectToPartitionKey())

          annotateLoci(ord, finalType, inserter, product = product)

        case Array(TInterval(_)) if vSignature == TVariant(GenomeReference.GRCh37) =>
          val partBc = sparkContext.broadcast(rdd.orderedPartitioner)
          val partitionKeyedIntervals = keyedRDD
            .flatMap { case (k, v) =>
              val interval = k.getAs[Interval[Locus]](0)
              val start = partBc.value.getPartitionT(interval.start.asInstanceOf[RPK])
              val end = partBc.value.getPartitionT(interval.end.asInstanceOf[RPK])
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

  def annotateLoci(lociRDD: OrderedRDD[RPK, RPK, Annotation], newSignature: Type,
    inserter: Inserter, product: Boolean): VariantSampleMatrix[RPK, RK, T] = {

    def annotate[S](joinedRDD: RDD[(RPK, ((RK, (Annotation, Iterable[T])), S))],
      ins: (Annotation, S) => Annotation): OrderedRDD[RPK, RK, (Annotation, Iterable[T])] = {
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

  def annotateVariants(otherRDD: OrderedRDD[RPK, RK, Annotation], newSignature: Type,
    inserter: Inserter, product: Boolean): VariantSampleMatrix[RPK, RK, T] = {
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

  def annotateVariantsVDS(other: VariantSampleMatrix[RPK, RK, _],
    root: Option[String] = None, code: Option[String] = None): VariantSampleMatrix[RPK, RK, T] = {

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

  def count(): (Long, Long) = (nSamples, variants.count())

  def countVariants(): Long = variants.count()

  def variants: RDD[RK] = rdd.keys

  def deduplicate(): VariantSampleMatrix[RPK, RK, T] = {
    copy(rdd = rdd.mapPartitionsPreservingPartitioning { it =>
      new SortedDistinctPairIterator(it)
    })
  }

  def deleteGlobal(args: String*): (Type, Deleter) = deleteGlobal(args.toList)

  def deleteGlobal(path: List[String]): (Type, Deleter) = globalSignature.delete(path)

  def deleteSA(args: String*): (Type, Deleter) = deleteSA(args.toList)

  def deleteSA(path: List[String]): (Type, Deleter) = saSignature.delete(path)

  def deleteVA(args: String*): (Type, Deleter) = deleteVA(args.toList)

  def deleteVA(path: List[String]): (Type, Deleter) = vaSignature.delete(path)

  def dropSamples(): VariantSampleMatrix[RPK, RK, T] =
    copy(sampleIds = IndexedSeq.empty[Annotation],
      sampleAnnotations = IndexedSeq.empty[Annotation],
      rdd = rdd.mapValues { case (va, gs) => (va, Iterable.empty[T]) })

  def dropVariants(): VariantSampleMatrix[RPK, RK, T] = copy(rdd = OrderedRDD.empty(sparkContext))

  def expand(): RDD[(RK, Annotation, T)] =
    mapWithKeys[(RK, Annotation, T)]((v, s, g) => (v, s, g))

  def expandWithAll(): RDD[(RK, Annotation, Annotation, Annotation, T)] =
    mapWithAll[(RK, Annotation, Annotation, Annotation, T)]((v, va, s, sa, g) => (v, va, s, sa, g))

  def mapWithAll[U](f: (RK, Annotation, Annotation, Annotation, T) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    rdd
      .flatMap { case (v, (va, gs)) =>
        localSampleIdsBc.value.lazyMapWith2[Annotation, T, U](localSampleAnnotationsBc.value, gs, { case (s, sa, g) => f(v, va, s, sa, g)
        })
      }
  }

  def annotateGenotypesExpr(expr: String): VariantSampleMatrix[RPK, RK, Annotation] = {
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

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
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

    mapValuesWithAll(finalType, { (v: RK, va: Annotation, s: Annotation, sa: Annotation, g: T) =>
      ec.setAll(v, va, s, sa, g)
      f().zip(inserters)
        .foldLeft(g: Annotation) { case (ga, (a, inserter)) =>
          inserter(ga, a)
        }
    })
  }

  def filterVariants(p: (RK, Annotation, Iterable[T]) => Boolean): VariantSampleMatrix[RPK, RK, T] =
    copy(rdd = rdd.filter { case (v, (va, gs)) => p(v, va, gs) })

  def filterSamplesMask(mask: Array[Boolean]): VariantSampleMatrix[RPK, RK, T] = {
    require(mask.length == nSamples)
    val maskBc = sparkContext.broadcast(mask)
    val localtct = tct
    copy(sampleIds = sampleIds.zipWithIndex
      .filter { case (s, i) => mask(i) }
      .map(_._1),
      sampleAnnotations = sampleAnnotations.zipWithIndex
        .filter { case (sa, i) => mask(i) }
        .map(_._1),
      rdd = rdd.mapValues { case (va, gs) =>
        (va, gs.lazyFilterWith(maskBc.value, (g: T, m: Boolean) => m))
      })
  }

  // FIXME see if we can remove broadcasts elsewhere in the code
  def filterSamples(p: (Annotation, Annotation) => Boolean): VariantSampleMatrix[RPK, RK, T] = {
    val mask = sampleIdsAndAnnotations.map { case (s, sa) => p(s, sa) }.toArray
    filterSamplesMask(mask)
  }
  
  /**
    * Filter samples using the Hail expression language.
    *
    * @param filterExpr Filter expression involving `s' (sample) and `sa' (sample annotations)
    * @param keep       keep where filterExpr evaluates to true
    */
  def filterSamplesExpr(filterExpr: String, keep: Boolean = true): VariantSampleMatrix[RPK, RK, T] = {
    var filterAST = Parser.expr.parse(filterExpr)
    if (!keep)
      filterAST = Apply(filterAST.getPos, "!", Array(filterAST))
    copyAST(ast = FilterSamples(ast, filterAST))
  }

  def filterSamplesList(samples: java.util.ArrayList[Annotation], keep: Boolean): VariantSampleMatrix[RPK, RK, T] =
    filterSamplesList(samples.asScala.toSet, keep)

  /**
    * Filter samples using a text file containing sample IDs
    *
    * @param samples Set of samples to keep or remove
    * @param keep    Keep listed samples.
    */
  def filterSamplesList(samples: Set[Annotation], keep: Boolean = true): VariantSampleMatrix[RPK, RK, T] = {
    val p = (s: Annotation, sa: Annotation) => Filter.keepThis(samples.contains(s), keep)
    filterSamples(p)
  }

  def filterSamplesTable(table: KeyTable, keep: Boolean): VariantSampleMatrix[RPK, RK, T] = {
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
  def filterVariantsExpr(filterExpr: String, keep: Boolean = true): VariantSampleMatrix[RPK, RK, T] = {
    var filterAST = Parser.expr.parse(filterExpr)
    if (!keep)
      filterAST = Apply(filterAST.getPos, "!", Array(filterAST))
    copyAST(ast = FilterVariants(ast, filterAST))
  }

  def filterVariantsList(variants: java.util.ArrayList[RK], keep: Boolean): VariantSampleMatrix[RPK, RK, T] =
    filterVariantsList(variants.asScala.toSet, keep)

  def filterVariantsList(variants: Set[RK], keep: Boolean): VariantSampleMatrix[RPK, RK, T] = {
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

      val adjRangeBounds: Array[RPK] =
        if (partitionVariants.isEmpty)
          Array.empty
        else
          partitionVariants.init.map { case (oldPart, _) =>
            rdd.orderedPartitioner.rangeBounds(oldPart)
          }

      val adjPart = OrderedPartitioner[RPK, RK](adjRangeBounds, partitionVariants.length)
      copy(rdd = OrderedRDD(adjRDD, adjPart))
    } else {
      val variantsBc = hc.sc.broadcast(variants)
      filterVariants { case (v, _, _) => !variantsBc.value.contains(v) }
    }
  }

  def filterVariantsTable(kt: KeyTable, keep: Boolean = true): VariantSampleMatrix[RPK, RK, T] = {
    val keyFields = kt.keyFields.map(_.typ)
    val filt = keyFields match {
      case Array(`vSignature`) =>
        val variantRDD = kt.keyedRDD()
          .map { case (k, v) => (k.getAs[RK](0), ()) }
          .filter(_._1 != null)
          .orderedRepartitionBy(rdd.orderedPartitioner)

        rdd.orderedLeftJoinDistinct(variantRDD)
          .filter { case (_, (_, o)) => Filter.keepThis(o.isDefined, keep) }
          .mapValues { case (vags, _) => vags }

      case Array(vSignature.partitionKey) =>
        val locusRDD = kt.keyedRDD()
          .map { case (k, v) => (k.getAs[RPK](0), ()) }
          .filter(_._1 != null)
          .orderedRepartitionBy(rdd.orderedPartitioner.projectToPartitionKey())

        OrderedRDD[RPK, RK, (Annotation, Iterable[T])](rdd.mapMonotonic(kOk.orderedProject, { case (v, vags) => (v, vags) })
          .orderedLeftJoinDistinct(locusRDD)
          .filter { case (_, (_, o)) => Filter.keepThis(o.isDefined, keep) }
          .map { case (_, ((v, vags), _)) => (v, vags) },
          rdd.orderedPartitioner)

      case Array(TInterval(_)) if vSignature == TVariant(GenomeReference.GRCh37) =>
        val partBc = sparkContext.broadcast(rdd.orderedPartitioner)
        val intRDD = kt.keyedRDD()
          .map { case (k, _) => k.getAs[Interval[Locus]](0) }
          .filter(_ != null)
          .flatMap { interval =>
            val start = partBc.value.getPartitionT(interval.start.asInstanceOf[RPK])
            val end = partBc.value.getPartitionT(interval.end.asInstanceOf[RPK])
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

  def flatMap[U](f: T => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] =
    flatMapWithKeys((v, s, g) => f(g))

  def flatMapWithKeys[U](f: (RK, Annotation, T) => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc

    rdd
      .flatMap { case (v, (va, gs)) => localSampleIdsBc.value.lazyFlatMapWith(gs,
        (s: Annotation, g: T) => f(v, s, g))
      }
  }

  /**
    * The function {@code f} must be monotonic with respect to the ordering on {@code Locus}
    */
  def flatMapVariants(f: (RK, Annotation, Iterable[T]) => TraversableOnce[(RK, (Annotation, Iterable[T]))]): VariantSampleMatrix[RPK, RK, T] =
  copy(rdd = rdd.flatMapMonotonic[(Annotation, Iterable[T])] { case (v, (va, gs)) => f(v, va, gs) })

  def hadoopConf: hadoop.conf.Configuration = hc.hadoopConf

  def head(n: Long): VariantSampleMatrix[RPK, RK, T] = {
    if (n < 0)
      fatal(s"n must be non-negative! Found `$n'.")
    copy(rdd = rdd.head(n))
  }

  def insertGlobal(sig: Type, args: String*): (Type, Inserter) = insertGlobal(sig, args.toList)

  def insertSA(sig: Type, args: String*): (Type, Inserter) = insertSA(sig, args.toList)

  def insertSA(sig: Type, path: List[String]): (Type, Inserter) = saSignature.insert(sig, path)

  def insertVA(sig: Type, args: String*): (Type, Inserter) = insertVA(sig, args.toList)

  def insertVA(sig: Type, path: List[String]): (Type, Inserter) = {
    vaSignature.insert(sig, path)
  }

  /**
    *
    * @param right right-hand dataset with which to join
    */
  def join(right: VariantSampleMatrix[RPK, RK, T]): VariantSampleMatrix[RPK, RK, T] = {
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

    val newSampleIds = sampleIds ++ right.sampleIds
    val duplicates = newSampleIds.duplicates()
    if (duplicates.nonEmpty)
      fatal("duplicate sample IDs: @1", duplicates)

    val joined = rdd.orderedInnerJoinDistinct(right.rdd)
      .mapValues { case ((lva, lgs), (rva, rgs)) =>
        val lrgs: Iterable[T] = new Iterable[T] with Serializable {
          def iterator: Iterator[T] = lgs.iterator ++ rgs.iterator
        }
        (lva, lrgs)
      }.asOrderedRDD

    copy(
      sampleIds = newSampleIds,
      sampleAnnotations = sampleAnnotations ++ right.sampleAnnotations,
      rdd = joined)
  }

  def makeKT(variantCondition: String, genotypeCondition: String, keyNames: Array[String] = Array.empty, seperator: String = "."): KeyTable = {
    requireSampleTString("make table")

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

  def map[U](f: T => U)(implicit uct: ClassTag[U]): RDD[U] =
    mapWithKeys((v, s, g) => f(g))

  def mapWithKeys[U](f: (RK, Annotation, T) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc

    rdd
      .flatMap { case (v, (va, gs)) =>
        localSampleIdsBc.value.lazyMapWith[T, U](gs,
          (s, g) => f(v, s, g))
      }
  }

  def mapAnnotations(newVASignature: Type, f: (RK, Annotation, Iterable[T]) => Annotation): VariantSampleMatrix[RPK, RK, T] =
    copy(vaSignature = newVASignature,
      rdd = rdd.mapValuesWithKey { case (v, (va, gs)) => (f(v, va, gs), gs) })

  def mapPartitionsWithAll[U](f: Iterator[(RK, Annotation, Annotation, Annotation, T)] => Iterator[U])
    (implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    rdd.mapPartitions { it =>
      f(it.flatMap { case (v, (va, gs)) =>
        localSampleIdsBc.value.lazyMapWith2[Annotation, T, (RK, Annotation, Annotation, Annotation, T)](
          localSampleAnnotationsBc.value, gs, { case (s, sa, g) => (v, va, s, sa, g) })
      })
    }
  }

  def mapValues[U >: Null](newGSignature: Type, f: (T) => U)(implicit uct: ClassTag[U]): VariantSampleMatrix[RPK, RK, U] = {
    mapValuesWithAll(newGSignature, (v, va, s, sa, g) => f(g))
  }

  def mapValuesWithAll[U >: Null](newGSignature: Type, f: (RK, Annotation, Annotation, Annotation, T) => U)
    (implicit uct: ClassTag[U]): VariantSampleMatrix[RPK, RK, U] = {
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc
    copy(genotypeSignature = newGSignature,
      rdd = rdd.mapValuesWithKey { case (v, (va, gs)) =>
        (va, localSampleIdsBc.value.lazyMapWith2[Annotation, T, U](
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
    val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[T](ec, { case (ec, g) =>
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
    val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[(RK, Annotation)](ec, { case (ec, (v, va)) =>
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

  def reorderSamples(newIds: java.util.ArrayList[Annotation]): VariantSampleMatrix[RPK, RK, T] =
    reorderSamples(newIds.asScala.toArray)

  def reorderSamples(newIds: Array[Annotation]): VariantSampleMatrix[RPK, RK, T] = {
    requireUniqueSamples("reorder_samples")

    val oldOrder = sampleIds.zipWithIndex.toMap
    val newOrder = newIds.zipWithIndex.toMap

    val newIndices = new Array[Int](nSamples)
    val missingSamples = mutable.Set[Annotation]()
    val notInDataset = mutable.Set[Annotation]()

    oldOrder.outerJoin(newOrder).foreach { case (s, (oldIdx, newIdx)) =>
      (oldIdx: @unchecked, newIdx: @unchecked) match {
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
    val localtct = tct

    val reorderedRdd = rdd.mapPartitions({ it =>
      it.map { case (v, (va, gs)) =>
        val reorderedGs = localtct.newArray(nSamplesLocal)
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

  def renameSamples(newIds: java.util.ArrayList[Annotation]): VariantSampleMatrix[RPK, RK, T] =
    renameSamples(newIds.asScala.toArray)

  def renameSamples(newIds: Array[Annotation]): VariantSampleMatrix[RPK, RK, T] = {
    if (newIds.length != sampleIds.length)
      fatal(s"dataset contains $nSamples samples, but new ID list contains ${ newIds.length }")
    copy(sampleIds = newIds)
  }

  def renameSamples(mapping: java.util.Map[Annotation, Annotation]): VariantSampleMatrix[RPK, RK, T] =
    renameSamples(mapping.asScala.toMap)

  def renameSamples(mapping: Map[Annotation, Annotation]): VariantSampleMatrix[RPK, RK, T] = {
    val newSampleIds = sampleIds.map(s => mapping.getOrElse(s, s))
    copy(sampleIds = newSampleIds)
  }

  def renameDuplicates(): VariantSampleMatrix[RPK, RK, T] = {
    requireSampleTString("rename duplicates")
    val (newIds, duplicates) = mangle(stringSampleIds.toArray)
    if (duplicates.nonEmpty)
      info(s"Renamed ${ duplicates.length } duplicate ${ plural(duplicates.length, "sample ID") }. " +
        s"Mangled IDs as follows:\n  @1", duplicates.map { case (pre, post) => s""""$pre" => "$post"""" }.truncatable("\n  "))
    else
      info(s"No duplicate sample IDs found.")
    val (newSchema, ins) = insertSA(TString, "originalID")
    val newAnnotations = sampleIdsAndAnnotations.map { case (s, sa) => ins(sa, s) }
    copy(sampleIds = newIds, saSignature = newSchema, sampleAnnotations = newAnnotations)
  }

  def same(that: VariantSampleMatrix[RPK, RK, T], tolerance: Double = utils.defaultTolerance): Boolean = {
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

  def sampleAnnotationsSimilar(that: VariantSampleMatrix[RPK, RK, T], tolerance: Double = utils.defaultTolerance): Boolean = {
    require(saSignature == that.saSignature)
    sampleAnnotations.zip(that.sampleAnnotations)
      .forall { case (s1, s2) => saSignature.valuesSimilar(s1, s2, tolerance)
      }
  }

  def sampleVariants(fraction: Double, seed: Int = 1): VariantSampleMatrix[RPK, RK, T] = {
    require(fraction > 0 && fraction < 1, s"the 'fraction' parameter must fall between 0 and 1, found $fraction")
    copy(rdd = rdd.sample(withReplacement = false, fraction, seed).asOrderedRDD)
  }

  def copy[RPK2, RK2, T2 >: Null](rdd: OrderedRDD[RPK2, RK2, (Annotation, Iterable[T2])] = rdd,
    sampleIds: IndexedSeq[Annotation] = sampleIds,
    sampleAnnotations: IndexedSeq[Annotation] = sampleAnnotations,
    globalAnnotation: Annotation = globalAnnotation,
    sSignature: Type = sSignature,
    saSignature: Type = saSignature,
    vSignature: Type = vSignature,
    vaSignature: Type = vaSignature,
    globalSignature: Type = globalSignature,
    genotypeSignature: Type = genotypeSignature,
    wasSplit: Boolean = wasSplit)
    (implicit tct: ClassTag[T2]): VariantSampleMatrix[RPK2, RK2, T2] =
    new VariantSampleMatrix[RPK2, RK2, T2](hc,
      VSMMetadata(sSignature, saSignature, vSignature, vaSignature, globalSignature, genotypeSignature, wasSplit),
      VSMLocalValue(globalAnnotation, sampleIds, sampleAnnotations), rdd)

  def copyAST[RPK2, RK2, T2 >: Null](ast: MatrixAST[RPK2, RK2, T2] = ast,
    sSignature: Type = sSignature,
    saSignature: Type = saSignature,
    vSignature: Type = vSignature,
    vaSignature: Type = vaSignature,
    globalSignature: Type = globalSignature,
    genotypeSignature: Type = genotypeSignature,
    wasSplit: Boolean = wasSplit)(implicit tct: ClassTag[T2]): VariantSampleMatrix[RPK2, RK2, T2] =
    new VariantSampleMatrix[RPK2, RK2, T2](hc,
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

  def storageLevel: String = rdd.getStorageLevel.toReadableString()

  def setVaAttributes(path: String, kv: Map[String, String]): VariantSampleMatrix[RPK, RK, T] = {
    setVaAttributes(Parser.parseAnnotationRoot(path, Annotation.VARIANT_HEAD), kv)
  }

  def setVaAttributes(path: List[String], kv: Map[String, String]): VariantSampleMatrix[RPK, RK, T] = {
    vaSignature match {
      case t: TStruct => copy(vaSignature = t.setFieldAttributes(path, kv))
      case t => fatal(s"Cannot set va attributes to ${ path.mkString(".") } since va is not a Struct.")
    }
  }

  def deleteVaAttribute(path: String, attribute: String): VariantSampleMatrix[RPK, RK, T] = {
    deleteVaAttribute(Parser.parseAnnotationRoot(path, Annotation.VARIANT_HEAD), attribute)
  }

  def deleteVaAttribute(path: List[String], attribute: String): VariantSampleMatrix[RPK, RK, T] = {
    vaSignature match {
      case t: TStruct => copy(vaSignature = t.deleteFieldAttribute(path, attribute))
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

  def variantsAndAnnotations: OrderedRDD[RPK, RK, Annotation] =
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
    KeyTable(hc, rdd.map { case (v, (va, gs)) =>
      Row(v, va)
    },
      TStruct(
        "v" -> vSignature,
        "va" -> vaSignature),
      Array("v"))
  }

  def genotypeKT(): KeyTable = {
    KeyTable(hc,
      expandWithAll().map { case (v, va, s, sa, g) => Row(v, va, s, sa, g) },
      TStruct(
        "v" -> vSignature,
        "va" -> vaSignature,
        "s" -> sSignature,
        "sa" -> saSignature,
        "g" -> genotypeSignature),
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

  def coalesce(k: Int, shuffle: Boolean = true): VariantSampleMatrix[RPK, RK, T] =
    copy(rdd = rdd.coalesce(k, shuffle = shuffle)(null).toOrderedRDD)

  def persist(storageLevel: String = "MEMORY_AND_DISK"): VariantSampleMatrix[RPK, RK, T] = {
    val level = try {
      StorageLevel.fromString(storageLevel)
    } catch {
      case e: IllegalArgumentException =>
        fatal(s"unknown StorageLevel `$storageLevel'")
    }

    copy(rdd = rdd.persist(level))
  }

  def cache(): VariantSampleMatrix[RPK, RK, T] = persist("MEMORY_ONLY")

  def unpersist() {
    rdd.unpersist()
  }

  def naiveCoalesce(maxPartitions: Int): VariantSampleMatrix[RPK, RK, T] =
    copy(rdd = rdd.naiveCoalesce(maxPartitions))

  /**
    * @param filterExpr filter expression involving v (Variant), va (variant annotations), s (sample),
    *                   sa (sample annotations), and g (genotype annotation), which returns a boolean value
    * @param keep       keep genotypes where filterExpr evaluates to true
    */
  def filterGenotypes(filterExpr: String, keep: Boolean = true): VariantSampleMatrix[RPK, RK, T] = {

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
    mapValuesWithAll(genotypeSignature, { (v: RK, va: Annotation, s: Annotation, sa: Annotation, g: T) =>
      ec.setAll(v, va, s, sa, g)
      if (Filter.boxedKeepThis(f(), localKeep))
        g
      else
        null
    })
  }

  def makeVariantConcrete(): VariantSampleMatrix[Locus, Variant, T] = {
    vSignature match {
      case TVariant(_) =>
      case _ => fatal(s"variant signature `Variant' required, found: ${ vSignature.toPrettyString() }")
    }

    if (kOk == Variant.orderedKey)
      this.asInstanceOf[VariantSampleMatrix[Locus, Variant, T]]
    else {
      copy(
        rdd = rdd.mapKeysMonotonic[Locus, Variant]((k: Annotation) => k.asInstanceOf[Variant],
          (pk: Annotation) => pk.asInstanceOf[Locus])(Variant.orderedKey))
    }
  }

  def makeGenotypeConcrete(): VariantSampleMatrix[RPK, RK, Genotype] = {
    if (genotypeSignature != TGenotype)
      fatal(s"genotype signature `Genotype' required, found: `${ genotypeSignature.toPrettyString() }'")

    if (tct == classTag[Genotype])
      this.asInstanceOf[VariantSampleMatrix[RPK, RK, Genotype]]
    else {
      copy(
        rdd = rdd.mapValues { case (va, gs) =>
          (va, gs.asInstanceOf[Iterable[Genotype]])
        })
    }
  }

  def toVKDS: VariantSampleMatrix[Locus, Variant, T] = makeVariantConcrete()

  def toVDS: VariantDataset = makeVariantConcrete().makeGenotypeConcrete()

  def toGDS: GenericDataset = {
    if (kOk.kct == classTag[Annotation] &&
      kOk.pkct == classTag[Annotation] &&
      tct == classTag[Annotation])
      this.asInstanceOf[VariantSampleMatrix[Annotation, Annotation, Annotation]]
    else {
      copy(
        rdd = rdd
          .mapValues { case (va, gs) => (va, gs: Iterable[Annotation]) }
          .mapKeysMonotonic[Annotation, Annotation]((k: RK) => k: Annotation, (pk: Annotation) => pk: Annotation)(vSignature.orderedKey))
    }
  }

  def rowSignature: TStruct = TStruct(
    "v" -> vSignature,
    "va" -> vaSignature,
    "gs" -> TArray(genotypeSignature))

  def write(dirname: String, overwrite: Boolean = false): Unit = {
    require(dirname.endsWith(".vds"), "generic dataset write paths must end in '.vds'")

    if (overwrite)
      hadoopConf.delete(dirname, recursive = true)
    else if (hadoopConf.exists(dirname))
      fatal(s"file already exists at `$dirname'")

    writeMetadata(dirname, rdd.partitions.length)

    hadoopConf.writeTextFile(dirname + "/partitioner.json.gz") { out =>
      implicit val pkjw = vSignature.partitionKey.jsonWriter
      Serialization.write(
        rdd.orderedPartitioner.mapMonotonic[Annotation, Annotation]((pk: RPK) => pk: Annotation)(
          vSignature.orderedKey).toJSON, out)
    }

    val localNSamples = nSamples
    val localGenotypeSignature = genotypeSignature
    unsafeRowRDD.writeRows(dirname, rowSignature)
  }

  def makeSchema(): StructType = {
    StructType(Array(
      StructField("variant", vSignature.schema, nullable = false),
      StructField("annotations", vaSignature.schema),
      StructField("gs",
        ArrayType(genotypeSignature.schema),
        nullable = false)
    ))
  }
}
