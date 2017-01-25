package is.hail.variant

import java.nio.ByteBuffer

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.expr.{EvalContext, TAggregable, _}
import is.hail.io.annotators.{BedAnnotator, IntervalListAnnotator}
import is.hail.io.plink.{FamFileConfig, PlinkLoader}
import is.hail.keytable.KeyTable
import is.hail.methods.{Aggregators, Filter}
import is.hail.sparkextras._
import is.hail.utils._
import is.hail.variant.Variant.orderedKey
import is.hail.{HailContext, utils}
import org.apache.hadoop
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkContext, SparkEnv}
import org.json4s.jackson.Serialization

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.io.Source
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

  lazy val sampleIdsBc = sparkContext.broadcast(sampleIds)

  lazy val sampleAnnotationsBc = sparkContext.broadcast(sampleAnnotations)

  /**
    * Aggregate by user-defined key and aggregation expressions.
    *
    * Equivalent of a group-by operation in SQL.
    *
    * @param keyExpr Named expression(s) for which fields are keys
    * @param aggExpr Named aggregation expression(s)
    */
  def aggregateByKey(keyExpr: String, aggExpr: String): KeyTable = {
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

    val (keyNames, keyTypes, keyF) = Parser.parseNamedExprs(keyExpr, keyEC)
    val (aggNames, aggTypes, aggF) = Parser.parseNamedExprs(aggExpr, ec)

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

  def aggregateByVariantWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, String, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Variant, U)] = {
    aggregateByVariantWithAll(zeroValue)((e, v, va, s, sa, g) => seqOp(e, v, s, g), combOp)
  }

  /**
    * Aggregate over intervals and export.
    *
    * @param intervalList Input interval list file
    * @param expr Export expression
    * @param out Output file path
    */
  def aggregateIntervals(intervalList: String, expr: String, out: String) {

    val vas = vaSignature
    val sas = saSignature
    val localGlobalAnnotation = globalAnnotation

    val aggregationST = Map(
      "global" -> (0, globalSignature),
      "interval" -> (1, TInterval),
      "v" -> (2, TVariant),
      "va" -> (3, vas))
    val symTab = Map(
      "global" -> (0, globalSignature),
      "interval" -> (1, TInterval),
      "variants" -> (2, TAggregable(TVariant, aggregationST)))

    val ec = EvalContext(symTab)
    ec.set(1, globalAnnotation)

    val (names, _, f) = Parser.parseExportExprs(expr, ec)

    if (names.isEmpty)
      fatal("this module requires one or more named expr arguments")

    val (zVals, seqOp, combOp, resultOp) =
      Aggregators.makeFunctions[(Interval[Locus], Variant, Annotation)](ec, { case (ec, (i, v, va)) =>
        ec.setAll(localGlobalAnnotation, i, v, va)
      })

    val iList = IntervalListAnnotator.read(intervalList, hc.hadoopConf)
    val iListBc = sparkContext.broadcast(iList)

    val results = variantsAndAnnotations.flatMap { case (v, va) =>
      iListBc.value.query(v.locus).map { i => (i, (i, v, va)) }
    }
      .aggregateByKey(zVals)(seqOp, combOp)
      .collectAsMap()

    hc.hadoopConf.writeTextFile(out) { out =>
      val sb = new StringBuilder
      sb.append("Contig")
      sb += '\t'
      sb.append("Start")
      sb += '\t'
      sb.append("End")
      names.foreach { col =>
        sb += '\t'
        sb.append(col)
      }
      sb += '\n'

      iList.toIterator
        .foreachBetween { interval =>

          sb.append(interval.start.contig)
          sb += '\t'
          sb.append(interval.start.position)
          sb += '\t'
          sb.append(interval.end.position)
          val res = results.getOrElse(interval, zVals)
          resultOp(res)

          ec.setAll(localGlobalAnnotation, interval)
          f().foreach { field =>
            sb += '\t'
            sb.append(field)
          }
        }(sb += '\n')

      out.write(sb.result())
    }
  }

  def annotateGlobal(a: Annotation, t: Type, code: String): VariantSampleMatrix[T] = {
    val (newT, i) = insertGlobal(t, Parser.parseAnnotationRoot(code, Annotation.GLOBAL_HEAD))
    copy(globalSignature = newT, globalAnnotation = i(globalAnnotation, Option(a)))
  }

  /**
    * Create and destroy global annotations with expression language.
    *
    * @param expr Annotation expression
    */
  def annotateGlobalExpr(expr: String): VariantSampleMatrix[T] = {
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

  /**
    * Load text file into global annotations as Array[String] or
    *   Set[String].
    *
    * @param path Input text file
    * @param root Global annotation path to store text file
    * @param asSet If true, load text file as Set[String],
    *   otherwise, load as Array[String]
    */
  def annotateGlobalList(path: String, root: String, asSet: Boolean = false): VariantSampleMatrix[T] = {
    val textList = hc.hadoopConf.readFile(path) { in =>
      Source.fromInputStream(in)
        .getLines()
        .toArray
    }

    val (sig, toInsert) =
      if (asSet)
        (TSet(TString), textList.toSet)
      else
        (TArray(TString), textList: IndexedSeq[String])

    val rootPath = Parser.parseAnnotationRoot(root, "global")

    val (newGlobalSig, inserter) = insertGlobal(sig, rootPath)

    copy(
      globalAnnotation = inserter(globalAnnotation, Some(toInsert)),
      globalSignature = newGlobalSig)
  }

  def globalAnnotation: Annotation = metadata.globalAnnotation

  def insertGlobal(sig: Type, path: List[String]): (Type, Inserter) = {
    globalSignature.insert(sig, path)
  }

  def globalSignature: Type = metadata.globalSignature

  /**
    * Load delimited text file (text table) into global annotations as
    *   Array[Struct].
    *
    * @param path Input text file
    * @param root Global annotation path to store text table
    * @param config Configuration options for importing text files
    */
  def annotateGlobalTable(path: String, root: String,
    config: TextTableConfiguration = TextTableConfiguration()): VariantSampleMatrix[T] = {
    val annotationPath = Parser.parseAnnotationRoot(root, Annotation.GLOBAL_HEAD)

    val (struct, rdd) = TextTableReader.read(sparkContext)(Array(path), config)
    val arrayType = TArray(struct)

    val (finalType, inserter) = insertGlobal(arrayType, annotationPath)

    val table = rdd
      .map(_.value)
      .collect(): IndexedSeq[Annotation]

    copy(
      globalAnnotation = inserter(globalAnnotation, Some(table)),
      globalSignature = finalType)
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

  def annotateSamples(signature: Type, path: List[String], annotation: (String) => Option[Annotation]): VariantSampleMatrix[T] = {
    val (t, i) = insertSA(signature, path)
    annotateSamples(annotation, t, i)
  }

  /**
    * Import PLINK .fam file into sample annotations.
    *
    * @param path Path to .fam file
    * @param root Sample annotation path at which to store .fam file
    * @param config .fam file configuration options
    */
  def annotateSamplesFam(path: String, root: String = "sa.fam",
    config: FamFileConfig = FamFileConfig()): VariantSampleMatrix[T] = {
    if (!path.endsWith(".fam"))
      fatal("input file must end in .fam")

    val (info, signature) = PlinkLoader.parseFam(path, config, hc.hadoopConf)

    val duplicateIds = info.map(_._1).duplicates().toArray
    if (duplicateIds.nonEmpty) {
      val n = duplicateIds.length
      fatal(
        s"""found $n duplicate sample ${ plural(n, "id") }:
           |  @1""".stripMargin, duplicateIds)
    }

    annotateSamples(info.toMap, signature, root)
  }

  def annotateSamplesList(path: String, root: String): VariantSampleMatrix[T] = {

    val samplesInList = hc.hadoopConf.readLines(path) { lines =>
      if (lines.isEmpty)
        warn(s"Empty annotation file given: $path")

      lines.map(_.value).toSet
    }

    val sampleAnnotations = sampleIds.map { s => (s, samplesInList.contains(s)) }.toMap
    annotateSamples(sampleAnnotations, TBoolean, root)
  }

  def annotateSamples(annotations: Map[String, Annotation], signature: Type, code: String): VariantSampleMatrix[T] = {
    val (t, i) = insertSA(signature, Parser.parseAnnotationRoot(code, Annotation.SAMPLE_HEAD))
    annotateSamples(annotations.get _, t, i)
  }

  def annotateSamplesTable(path: String, sampleExpr: String,
    root: Option[String] = None, code: Option[String] = None,
    config: TextTableConfiguration = TextTableConfiguration()): VariantSampleMatrix[T] = {

    val (isCode, annotationExpr) = (root, code) match {
      case (Some(r), None) => (false, r)
      case (None, Some(c)) => (true, c)
      case _ => fatal("this module requires one of `root' or 'code', but not both")
    }

    val (struct, rdd) = TextTableReader.read(sparkContext)(Array(path), config)

    val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) =
      if (isCode) {
        val ec = EvalContext(Map(
          "sa" -> (0, saSignature),
          "table" -> (1, struct)))
        Annotation.buildInserter(annotationExpr, saSignature, ec, Annotation.SAMPLE_HEAD)
      } else
        insertSA(struct, Parser.parseAnnotationRoot(annotationExpr, Annotation.SAMPLE_HEAD))

    val sampleQuery = struct.parseInStructScope[String](sampleExpr)

    val map = rdd
      .flatMap {
        _.map { a =>
          sampleQuery(a).map(s => (s, a))
        }.value
      }
      .collect()
      .toMap

    val vdsKeys = sampleIds.toSet
    val tableKeys = map.keySet
    val onlyVds = vdsKeys -- tableKeys
    val onlyTable = tableKeys -- vdsKeys
    if (onlyVds.nonEmpty) {
      warn(s"There were ${ onlyVds.size } samples present in the VDS but not in the table.")
    }
    if (onlyTable.nonEmpty) {
      warn(s"There were ${ onlyTable.size } samples present in the table but not in the VDS.")
    }

    annotateSamples(map.get _, finalType, inserter)
  }

  def annotateSamplesVDS(other: VariantSampleMatrix[_],
    root: Option[String] = None,
    code: Option[String] = None): VariantSampleMatrix[T] = {

    val (isCode, annotationExpr) = (root, code) match {
      case (Some(r), None) => (false, r)
      case (None, Some(c)) => (true, c)
      case _ => fatal("this module requires one of `root' or 'code', but not both")
    }

    val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) =
      if (isCode) {
        val ec = EvalContext(Map(
          "sa" -> (0, saSignature),
          "vds" -> (1, other.saSignature)))
        Annotation.buildInserter(annotationExpr, saSignature, ec, Annotation.SAMPLE_HEAD)
      } else
        insertSA(other.saSignature, Parser.parseAnnotationRoot(annotationExpr, Annotation.SAMPLE_HEAD))

    val m = other.sampleIdsAndAnnotations.toMap
    annotateSamples(m.get _, finalType, inserter)
  }

  def annotateSamples(annotation: (String) => Option[Annotation], newSignature: Type, inserter: Inserter): VariantSampleMatrix[T] = {
    val newAnnotations = sampleIds.zipWithIndex.map { case (id, i) =>
      val sa = sampleAnnotations(i)
      val newAnnotation = inserter(sa, annotation(id))
      newSignature.typeCheck(newAnnotation)
      newAnnotation
    }

    copy(sampleAnnotations = newAnnotations, saSignature = newSignature)
  }

  def annotateVariants(otherRDD: OrderedRDD[Locus, Variant, Annotation], signature: Type,
    code: String): VariantSampleMatrix[T] = {
    val (newSignature, ins) = insertVA(signature, Parser.parseAnnotationRoot(code, Annotation.VARIANT_HEAD))
    annotateVariants(otherRDD, newSignature, ins)
  }

  def annotateVariantsBED(path: String, root: String, all: Boolean = false): VariantSampleMatrix[T] = {
    val annotationPath = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)
    BedAnnotator(path, hc.hadoopConf) match {
      case (is, None) =>
        annotateIntervals(is, annotationPath)

      case (is, Some((t, m))) =>
        annotateIntervals(is, t, m, all = all, annotationPath)
    }
  }

  def annotateVariantsIntervals(path: String, root: String, all: Boolean = false): VariantSampleMatrix[T] = {
    val annotationPath = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)

    IntervalListAnnotator(path, hc.hadoopConf) match {
      case (is, Some((m, t))) =>
        annotateIntervals(is, m, t, all = all, annotationPath)

      case (is, None) =>
        annotateIntervals(is, annotationPath)
    }
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

  def annotateVariantsLoci(path: String, locusExpr: String,
    root: Option[String] = None, code: Option[String] = None,
    config: TextTableConfiguration = TextTableConfiguration()): VariantSampleMatrix[T] = {
    annotateVariantsLociAll(List(path), locusExpr, root, code, config)
  }

  def annotateVariantsLociAll(paths: Seq[String], locusExpr: String,
    root: Option[String] = None, code: Option[String] = None,
    config: TextTableConfiguration = TextTableConfiguration()): VariantSampleMatrix[T] = {
    val files = hc.hadoopConf.globAll(paths)
    if (files.isEmpty)
      fatal("Arguments referred to no files")

    val (struct, locusRDD) = TextTableReader.read(sparkContext)(files, config, nPartitions)

    val (isCode, annotationExpr) = (root, code) match {
      case (Some(r), None) => (false, r)
      case (None, Some(c)) => (true, c)
      case _ => fatal("this module requires one of `root' or 'code', but not both")
    }

    val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) =
      if (isCode) {
        val ec = EvalContext(Map(
          "va" -> (0, vaSignature),
          "table" -> (1, struct)))
        Annotation.buildInserter(annotationExpr, vaSignature, ec, Annotation.VARIANT_HEAD)
      } else insertVA(struct, Parser.parseAnnotationRoot(annotationExpr, Annotation.VARIANT_HEAD))

    val locusQuery = struct.parseInStructScope[Locus](locusExpr)


    import is.hail.variant.LocusImplicits.orderedKey
    val lociRDD = locusRDD.flatMap {
      _.map { a =>
        locusQuery(a).map(l => (l, a))
      }.value
    }.toOrderedRDD(rdd.orderedPartitioner.mapMonotonic)

    annotateLoci(lociRDD, finalType, inserter)
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

  def nPartitions: Int = rdd.partitions.length

  def annotateVariantsTable(path: String, variantExpr: String,
    root: Option[String] = None, code: Option[String] = None,
    config: TextTableConfiguration = TextTableConfiguration()): VariantSampleMatrix[T] = {
    annotateVariantsTables(List(path), variantExpr, root, code, config)
  }

  def annotateVariantsTables(paths: Seq[String], variantExpr: String,
    root: Option[String] = None, code: Option[String] = None,
    config: TextTableConfiguration = TextTableConfiguration()): VariantSampleMatrix[T] = {
    val files = hc.hadoopConf.globAll(paths)
    if (files.isEmpty)
      fatal("Arguments referred to no files")

    val (struct, variantRDD) = TextTableReader.read(sparkContext)(files, config, nPartitions)

    val (isCode, annotationExpr) = (root, code) match {
      case (Some(r), None) => (false, r)
      case (None, Some(c)) => (true, c)
      case _ => fatal("this module requires one of `root' or 'code', but not both")
    }

    val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) =
      if (isCode) {
        val ec = EvalContext(Map(
          "va" -> (0, vaSignature),
          "table" -> (1, struct)))
        Annotation.buildInserter(annotationExpr, vaSignature, ec, Annotation.VARIANT_HEAD)
      } else insertVA(struct, Parser.parseAnnotationRoot(annotationExpr, Annotation.VARIANT_HEAD))

    val variantQuery = struct.parseInStructScope[Variant](variantExpr)

    val keyedRDD = variantRDD.flatMap {
      _.map { a =>
        variantQuery(a).map(v => (v, a))
      }.value
    }.toOrderedRDD(rdd.orderedPartitioner)

    annotateVariants(keyedRDD, finalType, inserter)
  }

  def annotateVariants(otherRDD: OrderedRDD[Locus, Variant, Annotation], newSignature: Type,
    inserter: Inserter): VariantSampleMatrix[T] = {
    val newRDD = rdd.orderedLeftJoinDistinct(otherRDD)
      .mapValues { case ((va, gs), annotation) =>
        (inserter(va, annotation), gs)
      }.asOrderedRDD
    copy(rdd = newRDD, vaSignature = newSignature)
  }

  def annotateVariantsVDS(other: VariantSampleMatrix[_],
    root: Option[String] = None, code: Option[String] = None): VariantSampleMatrix[T] = {

    val (isCode, annotationExpr) = (root, code) match {
      case (Some(r), None) => (false, r)
      case (None, Some(c)) => (true, c)
      case _ => fatal("this module requires one of `root' or 'code', but not both")
    }

    val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) =
      if (isCode) {
        val ec = EvalContext(Map(
          "va" -> (0, vaSignature),
          "vds" -> (1, other.vaSignature)))
        Annotation.buildInserter(annotationExpr, vaSignature, ec, Annotation.VARIANT_HEAD)
      } else insertVA(other.vaSignature, Parser.parseAnnotationRoot(annotationExpr, Annotation.VARIANT_HEAD))

    annotateVariants(other.variantsAndAnnotations, finalType, inserter)
  }

  def countVariants(): Long = variants.count()

  def variants: RDD[Variant] = rdd.keys

  def deleteGlobal(args: String*): (Type, Deleter) = deleteGlobal(args.toList)

  def deleteGlobal(path: List[String]): (Type, Deleter) = globalSignature.delete(path)

  def deleteSA(args: String*): (Type, Deleter) = deleteSA(args.toList)

  def deleteSA(path: List[String]): (Type, Deleter) = saSignature.delete(path)

  def deleteVA(args: String*): (Type, Deleter) = deleteVA(args.toList)

  def deleteVA(path: List[String]): (Type, Deleter) = vaSignature.delete(path)

  def downsampleVariants(keep: Long): VariantSampleMatrix[T] = {
    sampleVariants(keep.toDouble / countVariants())
  }

  def dropSamples(): VariantSampleMatrix[T] =
    copy(sampleIds = IndexedSeq.empty[String],
      sampleAnnotations = IndexedSeq.empty[Annotation],
      rdd = rdd.mapValues { case (va, gs) => (va, Iterable.empty[T]) }
        .asOrderedRDD)

  def dropVariants(): VariantSampleMatrix[T] = copy(rdd = OrderedRDD.empty(sparkContext))

  def expand(): RDD[(Variant, String, T)] =
    mapWithKeys[(Variant, String, T)]((v, s, g) => (v, s, g))

  def expandWithAll(): RDD[(Variant, Annotation, String, Annotation, T)] =
    mapWithAll[(Variant, Annotation, String, Annotation, T)]((v, va, s, sa, g) => (v, va, s, sa, g))

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

  def filterIntervals(path: String, keep: Boolean): VariantSampleMatrix[T] = {
    filterIntervals(IntervalListAnnotator.read(path, sparkContext.hadoopConfiguration, prune = true), keep)
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

  def filterVariants(p: (Variant, Annotation, Iterable[T]) => Boolean): VariantSampleMatrix[T] =
    copy(rdd = rdd.filter { case (v, (va, gs)) => p(v, va, gs) }.asOrderedRDD)

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

  /**
    * Filter samples using a text file containing sample IDs
    * @param path path to sample list file
    * @param keep keep listed samples
    */
  def filterSamplesList(path: String, keep: Boolean = true): VariantSampleMatrix[T] = {
    val samples = hc.hadoopConf.readFile(path) { reader =>
      Source.fromInputStream(reader)
        .getLines()
        .filter(line => !line.isEmpty)
        .toSet
    }
    val p = (s: String, sa: Annotation) => Filter.keepThis(samples.contains(s), keep)

    filterSamples(p)
  }

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

  def sparkContext: SparkContext = hc.sc

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

  def hadoopConf: hadoop.conf.Configuration = hc.hadoopConf

  def insertGlobal(sig: Type, args: String*): (Type, Inserter) = insertGlobal(sig, args.toList)

  def insertSA(sig: Type, args: String*): (Type, Inserter) = insertSA(sig, args.toList)

  def insertSA(sig: Type, path: List[String]): (Type, Inserter) = saSignature.insert(sig, path)

  def insertVA(sig: Type, args: String*): (Type, Inserter) = insertVA(sig, args.toList)

  def insertVA(sig: Type, path: List[String]): (Type, Inserter) = {
    vaSignature.insert(sig, path)
  }

  def isDosage: Boolean = metadata.isDosage

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

  def mapAnnotations(f: (Variant, Annotation, Iterable[T]) => Annotation): VariantSampleMatrix[T] =
    copy[T](rdd = rdd.mapValuesWithKey { case (v, (va, gs)) => (f(v, va, gs), gs) }.asOrderedRDD)

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

  def minrep(maxShift: Int = 100): VariantSampleMatrix[T] = {
    require(maxShift > 0, s"invalid value for maxShift: $maxShift. Parameter must be a positive integer.")
    val minrepped = rdd.map {
      case (v, (va, gs)) =>
        (v.minrep, (va, gs))
    }
    copy(rdd = minrepped.smartShuffleAndSort(rdd.orderedPartitioner, maxShift))
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

  def vaSignature: Type = metadata.vaSignature

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

  /**
    *
    * @param path ID mapping file
    */
  def renameSamples(path: String): VariantSampleMatrix[T] = {
    val m = hc.hadoopConf.readFile(path) { s =>
      Source.fromInputStream(s)
        .getLines()
        .map {
          _.split("\t") match {
            case Array(old, news) => (old, news)
            case _ =>
              fatal("Invalid input. Use two tab-separated columns.")
          }
        }.toMap
    }

    val newSamples = mutable.Set.empty[String]
    val newSampleIds = sampleIds
      .map { s =>
        val news = m.getOrElse(s, s)
        if (newSamples.contains(news))
          fatal(s"duplicate sample ID `$news' after rename")
        newSamples += news
        news
      }
    copy(sampleIds = newSampleIds)
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

  def sampleIds: IndexedSeq[String] = metadata.sampleIds

  def saSignature: Type = metadata.saSignature

  def sampleAnnotations: IndexedSeq[Annotation] = metadata.sampleAnnotations

  def wasSplit: Boolean = metadata.wasSplit

  def sampleAnnotationsSimilar(that: VariantSampleMatrix[T], tolerance: Double = utils.defaultTolerance): Boolean = {
    require(saSignature == that.saSignature)
    sampleAnnotations.zip(that.sampleAnnotations)
      .forall { case (s1, s2) => saSignature.valuesSimilar(s1, s2, tolerance) }
  }

  def sampleVariants(fraction: Double): VariantSampleMatrix[T] =
    copy(rdd = rdd.sample(withReplacement = false, fraction, 1).asOrderedRDD)

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

  def storageLevel: String = rdd.getStorageLevel.toReadableString()

  override def toString = s"VariantSampleMatrix(metadata=$metadata, rdd=$rdd, sampleIds=$sampleIds, nSamples=$nSamples, vaSignature=$vaSignature, saSignature=$saSignature, globalSignature=$globalSignature, sampleAnnotations=$sampleAnnotations, sampleIdsAndAnnotations=$sampleIdsAndAnnotations, globalAnnotation=$globalAnnotation, wasSplit=$wasSplit)"

  def nSamples: Int = metadata.sampleIds.length

  def typecheck() {
    var foundError = false
    if (!globalSignature.typeCheck(globalAnnotation)) {
      warn(
        s"""found violation in global annotation
           |Schema: ${ globalSignature.toPrettyString() }
           |
            |Annotation: ${ Annotation.printAnnotation(globalAnnotation) }""".stripMargin)
    }

    sampleIdsAndAnnotations.find { case (_, sa) => !saSignature.typeCheck(sa) }
      .foreach { case (s, sa) =>
        foundError = true
        warn(
          s"""found violation in sample annotations for sample $s
             |Schema: ${ saSignature.toPrettyString() }
             |
              |Annotation: ${ Annotation.printAnnotation(sa) }""".stripMargin)
      }

    val localVaSignature = vaSignature
    variantsAndAnnotations.find { case (_, va) => !localVaSignature.typeCheck(va) }
      .foreach { case (v, va) =>
        foundError = true
        warn(
          s"""found violation in variant annotations for variant $v
             |Schema: ${ localVaSignature.toPrettyString() }
             |
              |Annotation: ${ Annotation.printAnnotation(va) }""".stripMargin)
      }

    if (foundError)
      fatal("found one or more type check errors")
  }

  def sampleIdsAndAnnotations: IndexedSeq[(String, Annotation)] = sampleIds.zip(sampleAnnotations)

  def variantsAndAnnotations: OrderedRDD[Locus, Variant, Annotation] = rdd.mapValuesWithKey { case (v, (va, gs)) => va }.asOrderedRDD

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

}
