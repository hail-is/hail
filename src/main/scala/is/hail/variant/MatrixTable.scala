package is.hail.variant

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.distributedmatrix._
import is.hail.expr._
import is.hail.io.VCFMetadata
import is.hail.table.Table
import is.hail.methods.Aggregators.SampleFunctions
import is.hail.methods._
import is.hail.sparkextras._
import is.hail.rvd.{OrderedRVD, OrderedRVType}
import is.hail.stats.RegressionUtils
import is.hail.utils._
import is.hail.{HailContext, utils}
import breeze.linalg.DenseMatrix
import is.hail.expr.types._
import org.apache.hadoop
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Partitioner, SparkContext}
import org.json4s._
import org.json4s.jackson.JsonMethods.parse
import org.json4s.jackson.Serialization

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.language.{existentials, implicitConversions}
import scala.reflect.ClassTag

case class VDSMetadata(
  version: Int,
  // FIXME remove on next reimport
  split: Option[Boolean],
  sample_schema: String,
  sample_annotation_schema: String,
  variant_schema: String,
  variant_annotation_schema: String,
  global_schema: String,
  genotype_schema: String,
  sample_annotations: JValue,
  global_annotation: JValue,
  // FIXME make partition_counts non-optional, remove n_partitions at next reimport
  n_partitions: Int,
  partition_counts: Option[Array[Long]])

object MatrixTable {
  final val fileVersion: Int = 0x101

  def read(hc: HailContext, dirname: String,
    dropSamples: Boolean = false, dropVariants: Boolean = false): MatrixTable = {
    // FIXME check _SUCCESS next time we force reimport
    val (fileMetadata, nPartitions) = readFileMetadata(hc.hadoopConf, dirname)
    new MatrixTable(hc,
      fileMetadata.metadata,
      MatrixRead(dirname, nPartitions, fileMetadata, dropSamples, dropVariants))
  }

  def apply(hc: HailContext,
    metadata: VSMMetadata,
    localValue: VSMLocalValue,
    rdd: OrderedRDD[Annotation, Annotation, (Annotation, Iterable[Annotation])]): MatrixTable =
    new MatrixTable(hc, metadata,
      MatrixLiteral(
        MatrixType(metadata),
        MatrixValue(MatrixType(metadata), localValue, rdd)))

  def apply(hc: HailContext, fileMetadata: VSMFileMetadata,
    rdd: OrderedRDD[Annotation, Annotation, (Annotation, Iterable[Annotation])]): MatrixTable =
    MatrixTable(hc, fileMetadata.metadata, fileMetadata.localValue, rdd)

  def fromLegacy[RK, T](hc: HailContext,
    metadata: VSMMetadata,
    localValue: VSMLocalValue,
    rdd: RDD[(RK, (Annotation, Iterable[T]))]): MatrixTable = {
    implicit val kOk = metadata.vSignature.orderedKey
    MatrixTable(hc, metadata, localValue,
      rdd.mapPartitions({ it =>
        it.map { case (v, (va, gs)) =>
          (v: Annotation, (va, gs: Iterable[Annotation]))
        }
      }, preservesPartitioning = true).toOrderedRDD)
  }

  def fromLegacy[RK, T](hc: HailContext,
    metadata: VSMMetadata,
    localValue: VSMLocalValue,
    rdd: RDD[(RK, (Annotation, Iterable[T]))],
    fastKeys: RDD[RK]): MatrixTable = {
    implicit val kOk = metadata.vSignature.orderedKey
    MatrixTable(hc, metadata, localValue,
      OrderedRDD(
        rdd.mapPartitions({ it =>
          it.map { case (v, (va, gs)) =>
            (v: Annotation, (va, gs: Iterable[Annotation]))
          }
        }, preservesPartitioning = true),
        Some(fastKeys.map { k => k: Annotation }), None))
  }

  def fromLegacy[RK, T](hc: HailContext,
    fileMetadata: VSMFileMetadata,
    rdd: RDD[(RK, (Annotation, Iterable[T]))]): MatrixTable =
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

    if (metadata.version != MatrixTable.fileVersion)
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

    (VSMFileMetadata(VSMMetadata(sSignature, saSignature, vSignature, vaSignature, globalSignature, genotypeSignature),
      VSMLocalValue(globalAnnotation, ids, annotations),
      metadata.partition_counts),
      metadata.n_partitions)
  }

  def gen(hc: HailContext, gen: VSMSubgen): Gen[MatrixTable] =
    gen.gen(hc)

  def genGeneric(hc: HailContext): Gen[MatrixTable] =
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

  def checkDatasetSchemasCompatible(datasets: Array[MatrixTable]) {
    val first = datasets(0)
    val sampleIds = first.sampleIds
    val vaSchema = first.vaSignature
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
          colKeySchema.toPrettyString(compact = true),
          ssig.toPrettyString(compact = true)
        )
      } else if (vsig != rowKeySchema) {
        fatal(
          s"""cannot combine datasets with different row key schemata
             |  Schema in datasets[0]: @1
             |  Schema in datasets[$i]: @2""".stripMargin,
          rowKeySchema.toPrettyString(compact = true),
          vsig.toPrettyString(compact = true)
        )
      } else if (ids != sampleIds) {
        fatal(
          s"""cannot combine datasets with different column identifiers or ordering
             |  IDs in datasets[0]: @1
             |  IDs in datasets[$i]: @2""".stripMargin, sampleIds, ids)
      } else if (vas != vaSchema) {
        fatal(
          s"""cannot combine datasets with different row annotation schemata
             |  Schema in datasets[0]: @1
             |  Schema in datasets[$i]: @2""".stripMargin,
          vaSchema.toPrettyString(compact = true),
          vas.toPrettyString(compact = true)
        )
      } else if (gsig != genotypeSchema) {
        fatal(
          s"""cannot read datasets with different cell schemata
             |  Schema in datasets[0]: @1
             |  Schema in datasets[$i]: @2""".stripMargin,
          genotypeSchema.toPrettyString(compact = true),
          gsig.toPrettyString(compact = true)
        )
      }
    }
  }

  def unionRows(datasets: java.util.ArrayList[MatrixTable]): MatrixTable =
    unionRows(datasets.asScala.toArray)

  def unionRows(datasets: Array[MatrixTable]): MatrixTable = {
    require(datasets.length >= 2)

    checkDatasetSchemasCompatible(datasets)
    val (first, others) = (datasets.head, datasets.tail)
    first.copyLegacy(rdd = first.sparkContext.union(datasets.map(_.rdd)))
  }

  def fromKeyTable(kt: Table): MatrixTable = {
    val vType: Type = kt.keyFields.map(_.typ) match {
      case Array(t@TVariant(_, _)) => t
      case arr => fatal("Require one key column of type Variant to produce a variant dataset, " +
        s"but found [ ${ arr.mkString(", ") } ]")
    }

    val rdd = kt.keyedRDD()
      .map { case (k, v) => (k.asInstanceOf[Row].get(0), v) }
      .filter(_._1 != null)
      .mapValues(a => (a: Annotation, Iterable.empty[Annotation]))

    val metadata = VSMMetadata(
      saSignature = TStruct.empty(),
      vSignature = vType,
      vaSignature = kt.valueSignature,
      globalSignature = TStruct.empty())

    MatrixTable.fromLegacy(kt.hc, metadata,
      VSMLocalValue(Annotation.empty, Array.empty[Annotation], Array.empty[Annotation]), rdd)
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
  tGen: (Type, Annotation) => Gen[Annotation]) {

  def gen(hc: HailContext): Gen[MatrixTable] =
    for (size <- Gen.size;
      (l, w) <- Gen.squareOfAreaAtMostSize.resize((size / 3 / 10) * 8);

      vSig <- vSigGen.resize(3);
      vaSig <- vaSigGen.map(t => t.deepOptional()).resize(3);
      sSig <- sSigGen.resize(3);
      saSig <- saSigGen.map(t => t.deepOptional()).resize(3);
      globalSig <- globalSigGen.resize(5);
      tSig <- tSigGen.map(t => t.structOptional()).resize(3);
      global <- globalGen(globalSig).resize(25);
      nPartitions <- Gen.choose(1, 10);

      sampleIds <- Gen.buildableOfN[Array](w, sGen(sSig).resize(3))
        .map(ids => ids.distinct);
      nSamples = sampleIds.length;
      saValues <- Gen.buildableOfN[Array](nSamples, saGen(saSig).resize(5));
      rows <- Gen.buildableOfN[Array](l,
        for (
          v <- vGen(vSig).resize(3);
          va <- vaGen(vaSig).resize(5);
          ts <- Gen.buildableOfN[Array](nSamples, tGen(tSig, v).resize(3)))
          yield (v, (va, ts: Iterable[Annotation]))))
      yield {
        assert(sampleIds.forall(_ != null))
        assert(rows.forall(_._1 != null))
        MatrixTable.fromLegacy(hc,
          VSMMetadata(sSig, saSig, vSig, vaSig, globalSig, tSig),
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
    tSigGen = Gen.const(Genotype.htsGenotypeType),
    sGen = (t: Type) => Gen.identifier.map(s => s: Annotation),
    saGen = (t: Type) => t.genValue,
    vaGen = (t: Type) => t.genValue,
    globalGen = (t: Type) => t.genValue,
    vGen = (t: Type) => Variant.gen,
    tGen = (t: Type, v: Annotation) => Genotype.genExtreme(v.asInstanceOf[Variant]))

  val plinkSafeBiallelic = random.copy(
    sGen = (t: Type) => Gen.plinkSafeIdentifier,
    vGen = (t: Type) => VariantSubgen.plinkCompatible.copy(nAllelesGen = Gen.const(2)).gen)

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

class MatrixTable(val hc: HailContext, val metadata: VSMMetadata,
  val ast: MatrixIR) extends JoinAnnotator {

  implicit val kOk: OrderedKey[Annotation, Annotation] = ast.typ.vType.orderedKey

  implicit val kOrd: Ordering[Annotation] = kOk.kOrd

  def this(hc: HailContext,
    metadata: VSMMetadata,
    localValue: VSMLocalValue,
    rdd2: OrderedRVD) =
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

  def genomeReference: GenomeReference = vSignature.asInstanceOf[TVariant].gr.asInstanceOf[GenomeReference]

  val VSMMetadata(sSignature, saSignature, vSignature, vaSignature, globalSignature, genotypeSignature) = metadata

  lazy val value: MatrixValue = {
    val opt = MatrixIR.optimize(ast)
    opt.execute(hc)
  }

  lazy val MatrixValue(matrixType, VSMLocalValue(globalAnnotation, sampleIds, sampleAnnotations), rdd2) = value

  lazy val rdd: OrderedRDD[Annotation, Annotation, (Annotation, Iterable[Annotation])] = value.rdd

  def typedRDD[RPK, RK](implicit rkct: ClassTag[RK]): OrderedRDD[RPK, RK, (Annotation, Iterable[Annotation])] = {
    implicit val kOk = vSignature.typedOrderedKey[RPK, RK]
    rdd.map { case (v, (va, gs)) =>
      (v.asInstanceOf[RK], (va, gs))
    }
      .toOrderedRDD
  }

  def partitionCounts(): Array[Long] = {
    ast.partitionCounts match {
      case Some(counts) => counts
      case None => rdd2.countPerPartition()
    }
  }

  // length nPartitions + 1, first element 0, last element rdd2 count
  def partitionStarts(): Array[Long] = partitionCounts().scanLeft(0L)(_ + _)

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

  def groupSamplesBy(keyExpr: String, aggExpr: String): MatrixTable = {
    val localRowType = rowType
    val sEC = EvalContext(Map(Annotation.GLOBAL_HEAD -> (0, globalSignature),
      "s" -> (1, sSignature),
      Annotation.SAMPLE_HEAD -> (2, saSignature)))
    val (keyType, keyF) = Parser.parseExpr(keyExpr, sEC)
    sEC.set(0, globalAnnotation)

    val keysBySample = sampleIds.zip(sampleAnnotations).map { case (s, sa) =>
      sEC.set(1, s)
      sEC.set(2, sa)
      keyF()
    }
    val newKeys = keysBySample.toSet.toArray
    val keyMap = newKeys.zipWithIndex.toMap
    val samplesMap = keysBySample.map { k => if (k == null) -1 else keyMap(k) }.toArray

    val nKeys = newKeys.size

    val ec = variantEC
    val (resultNames, resultTypes, resultF) = Parser.parseAnnotationExprs(aggExpr, ec, None)
    val entryType = TStruct(resultNames.map(_.head).zip(resultTypes).toSeq: _*)
    val mt = matrixType.copy(sType = keyType, saType = TStruct.empty(), genotypeType = entryType)
    val newRowType = mt.rowType

    val aggregate = Aggregators.buildVariantAggregationsByKey(this, nKeys, samplesMap, ec)

    val groupedRDD2 = rdd2.mapPartitionsPreservesPartitioning(mt.orderedRVType) { it =>
      val region2 = Region()
      val rv2 = RegionValue(region2)
      val rv2b = new RegionValueBuilder(region2)
      val ur = new UnsafeRow(localRowType)
      it.map { rv =>
        ur.set(rv)
        val aggArr = aggregate(rv)
        region2.clear()
        rv2b.start(newRowType)
        rv2b.startStruct()
        rv2b.addAnnotation(mt.locusType, ur.get(0))
        rv2b.addAnnotation(mt.vType, ur.get(1))
        rv2b.addAnnotation(mt.vaType, ur.get(2))

        rv2b.startArray(nKeys)
        var i = 0
        while (i < nKeys) {
          aggArr(i)()
          rv2b.startStruct()
          val fields = resultF()
          var j = 0
          while (j < fields.size) {
            rv2b.addAnnotation(entryType.fieldType(j), fields(j))
            j += 1
          }
          rv2b.endStruct()
          i += 1
        }
        rv2b.endArray()
        rv2b.endStruct()
        rv2.setOffset(rv2b.end())
        rv2
      }
    }

    copy2(rdd2 = groupedRDD2,
      sampleIds = newKeys,
      sampleAnnotations = Array.fill(newKeys.length)(Annotation.empty),
      sSignature = keyType,
      saSignature = TStruct.empty(),
      genotypeSignature = entryType)
  }

  def groupVariantsBy(keyExpr: String, aggExpr: String): MatrixTable = {
    val localRowType = rowType
    val vEC = EvalContext(Map(Annotation.GLOBAL_HEAD -> (0, globalSignature),
      "v" -> (1, vSignature),
      Annotation.VARIANT_HEAD -> (2, vaSignature)))
    val (keyType, keyF) = Parser.parseExpr(keyExpr, vEC)
    vEC.set(0, globalAnnotation)

    val keyedRDD = rdd2.rdd.mapPartitions { it =>
      val ur = new UnsafeRow(localRowType)
      it.flatMap { rv =>
        ur.set(rv)
        vEC.set(1, ur.get(1))
        vEC.set(2, ur.get(2))
        Option(keyF()).map { key => (Annotation.copy(keyType, key), ur) }
      }
    }

    val SampleFunctions(zero, seqOp, combOp, resultOp, resultType) = Aggregators.makeSampleFunctions(this, aggExpr)

    val (pkType, pkF: (Annotation => Annotation)) = keyType match {
      case TVariant(gr, _) => (TLocus(gr), { key: Variant => key.locus })
      case t => (t, { key: Annotation => key })
    }

    val signature = TStruct("pk" -> pkType, "v" -> keyType, Annotation.VARIANT_HEAD -> TStruct.empty(), Annotation.GENOTYPE_HEAD -> TArray(resultType))
    val rdd = keyedRDD
      .aggregateByKey(zero)(seqOp, combOp)
      .mapPartitions { it =>
        val region = Region()
        val rv = RegionValue(region)
        val rvb = new RegionValueBuilder(region)
        it.map { case (key, agg) =>
          region.clear()
          rvb.start(signature)
          rvb.startStruct()
          rvb.addAnnotation(pkType, pkF(key))
          rvb.addAnnotation(keyType, key)
          rvb.startStruct()
          rvb.endStruct()
          resultOp(agg, rvb)
          rvb.endStruct()
          rv.setOffset(rvb.end())
          rv
        }
      }

    val mt = matrixType.copy(vType = keyType, vaType = TStruct.empty(), genotypeType = resultType)

    copy2(rdd2 = OrderedRVD(mt.orderedRVType, rdd, None, None),
      vSignature = keyType,
      vaSignature = TStruct.empty(),
      genotypeSignature = resultType)
  }

  def annotateGlobal(a: Annotation, t: Type, code: String): MatrixTable = {
    val (newT, i) = insertGlobal(t, Parser.parseAnnotationRoot(code, Annotation.GLOBAL_HEAD))
    copy2(globalSignature = newT, globalAnnotation = i(globalAnnotation, a))
  }

  /**
    * Create and destroy global annotations with expression language.
    *
    * @param expr Annotation expression
    */
  def annotateGlobalExpr(expr: String): MatrixTable = {
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

  def annotateSamples(signature: Type, path: List[String], annotations: Array[Annotation]): MatrixTable = {
    val (t, ins) = insertSA(signature, path)

    val newAnnotations = new Array[Annotation](nSamples)

    for (i <- sampleAnnotations.indices) {
      newAnnotations(i) = ins(sampleAnnotations(i), annotations(i))
      t.typeCheck(newAnnotations(i))
    }

    copy(sampleAnnotations = newAnnotations, saSignature = t)
  }

  def annotateSamplesExpr(expr: String): MatrixTable = {
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

  def annotateSamples(annotations: Map[Annotation, Annotation], signature: Type, code: String): MatrixTable = {
    val (t, i) = insertSA(signature, Parser.parseAnnotationRoot(code, Annotation.SAMPLE_HEAD))
    annotateSamples(s => annotations.getOrElse(s, null), t, i)
  }

  def annotateSamplesTable(kt: Table, vdsKey: java.util.ArrayList[String],
    root: String, expr: String, product: Boolean): MatrixTable =
    annotateSamplesTable(kt, if (vdsKey != null) vdsKey.asScala else null, root, expr, product)

  def annotateSamplesTable(kt: Table, vdsKey: Seq[String] = null,
    root: String = null, expr: String = null, product: Boolean = false): MatrixTable = {

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

  def annotateSamples(annotation: (Annotation) => Annotation, newSignature: Type, inserter: Inserter): MatrixTable = {
    val newAnnotations = sampleIds.zipWithIndex.map { case (id, i) =>
      val sa = sampleAnnotations(i)
      val newAnnotation = inserter(sa, annotation(id))
      newSignature.typeCheck(newAnnotation)
      newAnnotation
    }

    copy2(sampleAnnotations = newAnnotations, saSignature = newSignature)
  }

  def mapAnnotations(newVASignature: Type, f: (Annotation, Annotation, Iterable[Annotation]) => Annotation): MatrixTable = {
    val localRowType = rowType
    insertIntoRow(() => new UnsafeRow(localRowType))(newVASignature, List("va"), { case (ur, rv, rvb) =>
      ur.set(rv)
      val v = ur.getAs[Annotation](1)
      val va = ur.get(2)
      val gs = ur.getAs[Iterable[Annotation]](3)
      rvb.addAnnotation(newVASignature, f(v, va, gs))
    })
  }

  def annotateVariantsExpr(expr: String): MatrixTable = {
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

  def annotateVariantsTable(kt: Table, vdsKey: java.util.ArrayList[String],
    root: String, expr: String, product: Boolean): MatrixTable =
    annotateVariantsTable(kt, if (vdsKey != null) vdsKey.asScala else null, root, expr, product)

  def annotateVariantsTable(kt: Table, vdsKey: Seq[String] = null,
    root: String = null, expr: String = null, product: Boolean = false): MatrixTable = {

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
          implicit val locusOrd = genomeReference.locusOrdering
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
    inserter: Inserter, product: Boolean): MatrixTable = {

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

  def nPartitions: Int = rdd2.partitions.length

  def annotateVariants2(rightRDD2: OrderedRVD, newVAType: Type, inserter: Inserter): MatrixTable = {
    val leftRowType = rowType
    val rightRowType = rightRDD2.typ.rowType

    val newMatrixType = matrixType.copy(vaType = newVAType)
    val newRowType = newMatrixType.rowType

    copy2(
      vaSignature = newVAType,
      rdd2 = OrderedRVD(
        newMatrixType.orderedRVType,
        rdd2.partitioner,
        rdd2.orderedJoinDistinct(rightRDD2, "left")
          .mapPartitions { it =>
            val rvb = new RegionValueBuilder()
            val rv2 = RegionValue()

            it.map { jrv =>
              val leftRV = jrv.rvLeft
              assert(leftRV != null)
              val region = leftRV.region
              val rightRV = jrv.rvRight

              val leftUR = new UnsafeRow(leftRowType, leftRV)
              val va = leftUR.get(2)

              val a =
                if (rightRV != null) {
                  val rightUR = new UnsafeRow(rightRowType, rightRV)
                  rightUR.get(2)
                } else
                  null

              val newVA = inserter(va, a)

              rvb.set(region)
              rvb.start(newRowType)
              rvb.startStruct()
              rvb.addField(leftRowType, leftRV, 0) // pk
              rvb.addField(leftRowType, leftRV, 1) // v
              rvb.addAnnotation(newVAType, newVA)
              rvb.addField(leftRowType, leftRV, 3) // gs
              rvb.endStruct()
              rv2.set(region, rvb.end())

              rv2
            }
          }))
  }

  def annotateVariants(otherRDD: OrderedRDD[Annotation, Annotation, Annotation], newSignature: Type,
    inserter: Inserter, product: Boolean): MatrixTable = {
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

  def annotateVariantsVDS(right: MatrixTable,
    root: Option[String] = None, code: Option[String] = None): MatrixTable = {

    val (isCode, annotationExpr) = (root, code) match {
      case (Some(r), None) => (false, r)
      case (None, Some(c)) => (true, c)
      case _ => fatal("this module requires one of `root' or 'code', but not both")
    }

    val (finalType, inserter): (Type, (Annotation, Annotation) => Annotation) =
      if (isCode) {
        val ec = EvalContext(Map(
          "va" -> (0, vaSignature),
          "vds" -> (1, right.vaSignature)))
        Annotation.buildInserter(annotationExpr, vaSignature, ec, Annotation.VARIANT_HEAD)
      } else insertVA(right.vaSignature, Parser.parseAnnotationRoot(annotationExpr, Annotation.VARIANT_HEAD))

    annotateVariants2(right.dropSamples().rdd2, finalType, inserter)
  }

  def count(): (Long, Long) = (nSamples, countVariants())

  def countVariants(): Long = partitionCounts().sum

  def variants: RDD[Annotation] = rdd.keys

  def deduplicate(): MatrixTable =
    copy2(rdd2 = rdd2.mapPartitionsPreservesPartitioning(rdd2.typ)(
      SortedDistinctRowIterator.transformer(rdd2.typ)))

  def deleteVA(args: String*): (Type, Deleter) = deleteVA(args.toList)

  def deleteVA(path: List[String]): (Type, Deleter) = vaSignature.delete(path)

  def dropSamples(): MatrixTable =
    copyAST(ast = FilterSamples(ast, Const(null, false, TBoolean())))

  def dropVariants(): MatrixTable = copy2(rdd2 = OrderedRVD.empty(sparkContext, matrixType.orderedRVType))

  def explodeVariants(code: String): MatrixTable = {
    val path = List(Annotation.VARIANT_HEAD) ++ Parser.parseAnnotationRoot(code, Annotation.VARIANT_HEAD)
    val (keysType, querier) = rowType.queryTyped(path)
    val keyType = keysType match {
      case TArray(e, _) => e
      case TSet(e, _) => e
      case t => fatal(s"Expected annotation of type Array or Set; found $t")
    }

    val (newRowType, inserter) = rowType.unsafeInsert(keyType, path)
    val newVAType = newRowType.asInstanceOf[TStruct].fieldType(2)
    val localRowType = rowType

    val explodedRDD = rdd2.rdd.mapPartitions { it =>
      val region2 = Region()
      val rv2 = RegionValue(region2)
      val rv2b = new RegionValueBuilder(region2)
      val ur = new UnsafeRow(localRowType)
      it.flatMap { rv =>
        ur.set(rv)
        val keys = querier(ur).asInstanceOf[Iterable[Any]]
        if (keys == null)
          None
        else
          keys.iterator.map { va =>
            region2.clear()
            rv2b.start(newRowType)
            inserter(rv.region, rv.offset, rv2b, { () =>
              rv2b.addAnnotation(keyType, va)
            })
            rv2.setOffset(rv2b.end())
            rv2
          }
      }
    }
    val newMatrixType = matrixType.copy(vaType = newVAType)
    copy2(vaSignature = newVAType, rdd2 = rdd2.copy(typ = newMatrixType.orderedRVType, rdd = explodedRDD))
  }

  def explodeSamples(code: String): MatrixTable = {
    val path = Parser.parseAnnotationRoot(code, Annotation.SAMPLE_HEAD)
    val (keysType, querier) = saSignature.queryTyped(path)
    val keyType = keysType match {
      case TArray(e, _) => e
      case TSet(e, _) => e
      case t => fatal(s"Expected annotation of type Array or Set; found $t")
    }
    var size = 0
    val keys = sampleAnnotations.map { sa =>
      val ks = querier(sa).asInstanceOf[Iterable[Any]]
      if (ks == null)
        Iterable.empty[Any]
      else {
        size += ks.size
        ks
      }
    }

    val (newSASig, inserter) = saSignature.insert(keyType, path)

    val sampleMap = new Array[Int](size)
    val newSampleIds = new Array[Annotation](size)
    val newSampleAnnotations = new Array[Annotation](size)

    var i = 0
    var j = 0
    while (i < nSamples) {
      keys(i).foreach { e =>
        sampleMap(j) = i
        newSampleIds(j) = sampleIds(i)
        newSampleAnnotations(j) = inserter(sampleAnnotations(i), e)
        j += 1
      }
      i += 1
    }

    val sampleMapBc = sparkContext.broadcast(sampleMap)
    val localRowType = rowType
    val localGSSig = rowType.fieldType(3).asInstanceOf[TArray]

    val newRDD = rdd2.rdd.mapPartitions { it =>
      val region2 = Region()
      val rv2 = RegionValue(region2)
      val rv2b = new RegionValueBuilder(region2)
      it.map { rv =>
        region2.clear()
        rv2b.start(localRowType)
        rv2b.startStruct()
        var i = 0
        while (i < 3) {
          rv2b.addRegionValue(localRowType.fieldType(i), rv.region, localRowType.loadField(rv, i))
          i += 1
        }
        rv2b.startArray(newSampleIds.length)
        i = 0
        val arrayOff = localRowType.loadField(rv, 3)
        while (i < newSampleIds.length) {
          rv2b.addRegionValue(localGSSig.elementType, rv.region,
            localGSSig.loadElement(rv.region, arrayOff, sampleMapBc.value(i)))
          i += 1
        }
        rv2b.endArray()
        rv2b.endStruct()
        rv2b.end()
        rv2.setOffset(rv2b.end())
        rv2
      }
    }

    copy2(sampleIds = newSampleIds,
      sampleAnnotations = newSampleAnnotations,
      saSignature = newSASig,
      rdd2 = rdd2.copy(rdd = newRDD))
  }

  def annotateGenotypesExpr(expr: String): MatrixTable = {
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

    mapValuesWithAll(finalType, { (v: Annotation, va: Annotation, s: Annotation, sa: Annotation, g: Annotation) =>
      ec.setAll(v, va, s, sa, g)
      f().zip(inserters)
        .foldLeft(g: Annotation) { case (ga, (a, inserter)) =>
          inserter(ga, a)
        }
    })
  }

  def filterVariants(p: (Annotation, Annotation, Iterable[Annotation]) => Boolean): MatrixTable = {
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

  def filterSamples(p: (Annotation, Annotation) => Boolean): MatrixTable = {
    copyAST(ast = MatrixLiteral(matrixType, value.filterSamples(p)))
  }

  /**
    * Filter samples using the Hail expression language.
    *
    * @param filterExpr Filter expression involving `s' (sample) and `sa' (sample annotations)
    * @param keep       keep where filterExpr evaluates to true
    */
  def filterSamplesExpr(filterExpr: String, keep: Boolean = true): MatrixTable = {
    var filterAST = Parser.expr.parse(filterExpr)
    if (!keep)
      filterAST = Apply(filterAST.getPos, "!", Array(filterAST))
    copyAST(ast = FilterSamples(ast, filterAST))
  }

  def filterSamplesList(samples: java.util.ArrayList[Annotation], keep: Boolean): MatrixTable =
    filterSamplesList(samples.asScala.toSet, keep)

  /**
    * Filter samples using a text file containing sample IDs
    *
    * @param samples Set of samples to keep or remove
    * @param keep    Keep listed samples.
    */
  def filterSamplesList(samples: Set[Annotation], keep: Boolean = true): MatrixTable = {
    val p = (s: Annotation, sa: Annotation) => Filter.keepThis(samples.contains(s), keep)
    filterSamples(p)
  }

  def filterSamplesTable(table: Table, keep: Boolean): MatrixTable = {
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
  def filterVariantsExpr(filterExpr: String, keep: Boolean = true): MatrixTable = {
    var filterAST = Parser.expr.parse(filterExpr)
    if (!keep)
      filterAST = Apply(filterAST.getPos, "!", Array(filterAST))
    copyAST(ast = FilterVariants(ast, filterAST))
  }

  def filterVariantsList(variants: java.util.ArrayList[Annotation], keep: Boolean): MatrixTable =
    filterVariantsList(variants.asScala.toSet, keep)

  def filterVariantsList(variants: Set[Annotation], keep: Boolean): MatrixTable = {
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

  def filterVariantsTable(kt: Table, keep: Boolean = true): MatrixTable = {
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
        implicit val locusOrd = genomeReference.locusOrdering
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

  def head(n: Long): MatrixTable = {
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
    minimum: Option[Double] = None, maximum: Option[Double] = None): Table = {
    IBD(this, computeMafExpr, bounded, minimum, maximum)
  }

  def insertSA(sig: Type, args: String*): (Type, Inserter) = insertSA(sig, args.toList)

  def insertSA(sig: Type, path: List[String]): (Type, Inserter) = saSignature.insert(sig, path)

  def insertVA(sig: Type, path: List[String]): (Type, Inserter) = {
    vaSignature.insert(sig, path)
  }

  def insertIntoRow[PC](makePartitionContext: () => PC)(typeToInsert: Type, path: List[String],
    inserter: (PC, RegionValue, RegionValueBuilder) => Unit): MatrixTable = {
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
  def unionCols(right: MatrixTable): MatrixTable = {
    if (genotypeSignature != right.genotypeSignature) {
      fatal(
        s"""union_cols: cannot combine datasets with different entry schema
           |  left entry schema: @1
           |  right entry schema: @2""".stripMargin,
        genotypeSignature.toPrettyString(compact = true),
        right.genotypeSignature.toPrettyString(compact = true))
    }

    if (sSignature != right.sSignature) {
      fatal(
        s"""union_cols: cannot combine datasets with different column key schema
           |  left column schema: @1
           |  right column schema: @2""".stripMargin,
        sSignature.toPrettyString(compact = true),
        right.sSignature.toPrettyString(compact = true))
    }

    if (saSignature != right.saSignature) {
      fatal(
        s"""union_cols: cannot combine datasets with different column schema
           |  left column schema: @1
           |  right column schema: @2""".stripMargin,
        saSignature.toPrettyString(compact = true),
        right.saSignature.toPrettyString(compact = true))
    }

    if (vSignature != right.vSignature) {
      fatal(
        s"""union_cols: cannot combine datasets with different row key schema
           |  left row key schema: @1
           |  right row key schema: @2""".stripMargin,
        vSignature.toPrettyString(compact = true),
        right.vSignature.toPrettyString(compact = true))
    }

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

    copy2(sampleIds = sampleIds ++ right.sampleIds,
      sampleAnnotations = sampleAnnotations ++ right.sampleAnnotations,
      rdd2 = OrderedRVD(rdd2.typ, rdd2.partitioner, joined))
  }

  def makeKT(variantCondition: String, genotypeCondition: String, keyNames: Array[String] = Array.empty, seperator: String = "."): Table = {
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

    Table(hc,
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

  def mapValuesWithAll[U >: Null](newGSignature: Type, f: (Annotation, Annotation, Annotation, Annotation, Annotation) => U)
    (implicit uct: ClassTag[U]): MatrixTable = {
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc
    copy(genotypeSignature = newGSignature,
      rdd = rdd.mapValuesWithKey { case (v, (va, gs)) =>
        (va, localSampleIdsBc.value.lazyMapWith2[Annotation, Annotation, U](
          localSampleAnnotationsBc.value, gs, { case (s, sa, g) => f(v, va, s, sa, g) }))
      })
  }

  def mendelErrors(ped: Pedigree): (Table, Table, Table, Table) = {
    requireColKeyString("mendel errors")
    requireRowKeyVariant("mendel errors")

    val men = MendelErrors(this, ped.filterTo(stringSampleIdSet).completeTrios)

    (men.mendelKT(), men.fMendelKT(), men.iMendelKT(), men.lMendelKT())
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

  def reorderSamples(newIds: java.util.ArrayList[Annotation]): MatrixTable =
    reorderSamples(newIds.asScala.toArray)

  def reorderSamples(newIds: Array[Annotation]): MatrixTable = {
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

  def renameSamples(newIds: java.util.ArrayList[Annotation]): MatrixTable =
    renameSamples(newIds.asScala.toArray)

  def renameSamples(newIds: Array[Annotation]): MatrixTable = {
    if (newIds.length != sampleIds.length)
      fatal(s"dataset contains $nSamples samples, but new ID list contains ${ newIds.length }")
    copy2(sampleIds = newIds)
  }

  def renameSamples(mapping: java.util.Map[Annotation, Annotation]): MatrixTable =
    renameSamples(mapping.asScala.toMap)

  def renameSamples(mapping: Map[Annotation, Annotation]): MatrixTable = {
    val newSampleIds = sampleIds.map(s => mapping.getOrElse(s, s))
    copy2(sampleIds = newSampleIds)
  }

  def renameDuplicates(): MatrixTable = {
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

  def same(that: MatrixTable, tolerance: Double = utils.defaultTolerance): Boolean = {
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

  def sampleAnnotationsSimilar(that: MatrixTable, tolerance: Double = utils.defaultTolerance): Boolean = {
    require(saSignature == that.saSignature)
    sampleAnnotations.zip(that.sampleAnnotations)
      .forall { case (s1, s2) => saSignature.valuesSimilar(s1, s2, tolerance)
      }
  }

  def sampleVariants(fraction: Double, seed: Int = 1): MatrixTable = {
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
    genotypeSignature: Type = genotypeSignature): MatrixTable =
    MatrixTable(hc,
      VSMMetadata(sSignature, saSignature, vSignature, vaSignature, globalSignature, genotypeSignature),
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
    genotypeSignature: Type = genotypeSignature): MatrixTable =
    MatrixTable.fromLegacy(hc,
      VSMMetadata(sSignature, saSignature, vSignature, vaSignature, globalSignature, genotypeSignature),
      VSMLocalValue(globalAnnotation, sampleIds, sampleAnnotations),
      rdd)

  def copy2(rdd2: OrderedRVD = rdd2,
    sampleIds: IndexedSeq[Annotation] = sampleIds,
    sampleAnnotations: IndexedSeq[Annotation] = sampleAnnotations,
    globalAnnotation: Annotation = globalAnnotation,
    sSignature: Type = sSignature,
    saSignature: Type = saSignature,
    vSignature: Type = vSignature,
    vaSignature: Type = vaSignature,
    globalSignature: Type = globalSignature,
    genotypeSignature: Type = genotypeSignature): MatrixTable =
    new MatrixTable(hc,
      VSMMetadata(sSignature, saSignature, vSignature, vaSignature, globalSignature, genotypeSignature),
      VSMLocalValue(globalAnnotation, sampleIds, sampleAnnotations), rdd2)

  def copyAST(ast: MatrixIR = ast,
    sSignature: Type = sSignature,
    saSignature: Type = saSignature,
    vSignature: Type = vSignature,
    vaSignature: Type = vaSignature,
    globalSignature: Type = globalSignature,
    genotypeSignature: Type = genotypeSignature): MatrixTable =
    new MatrixTable(hc,
      VSMMetadata(sSignature, saSignature, vSignature, vaSignature, globalSignature, genotypeSignature),
      ast)

  def samplesKT(): Table = {
    Table(hc, sparkContext.parallelize(sampleIdsAndAnnotations)
      .map { case (s, sa) =>
        Row(s, sa)
      },
      TStruct(
        "s" -> sSignature,
        "sa" -> saSignature),
      Array("s"))
  }

  def storageLevel: String = rdd2.storageLevel.toReadableString()

  def summarize(): SummaryResult = {
    val localRowType = rowType
    val localNSamples = nSamples
    rdd2.aggregateWithContext(() =>
      (HardCallView(localRowType),
        new RegionValueVariant(localRowType.fieldType(1).asInstanceOf[TVariant]))
    )(new SummaryCombiner)(
      { case ((view, rvVariant), summary, rv) =>
        rvVariant.setRegion(rv.region, localRowType.loadField(rv, 1))
        view.setRegion(rv)
        summary.merge(view, rvVariant)
      }, _.merge(_))
      .result(localNSamples)
  }

  override def toString =
    s"MatrixTable(metadata=$metadata, rdd=$rdd, sampleIds=$sampleIds, nSamples=$nSamples, vaSignature=$vaSignature, saSignature=$saSignature, globalSignature=$globalSignature, sampleAnnotations=$sampleAnnotations, sampleIdsAndAnnotations=$sampleIdsAndAnnotations, globalAnnotation=$globalAnnotation)"

  def nSamples: Int = sampleIds.length

  def typecheck() {
    var foundError = false
    if (!globalSignature.typeCheck(globalAnnotation)) {
      foundError = true
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

  def variantsKT(): Table = {
    val localRowType = rowType
    val typ = TStruct(
      "v" -> vSignature,
      "va" -> vaSignature)
    new Table(hc, rdd2.mapPartitions { it =>
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

  def genotypeKT(): Table = {
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
    new Table(hc, rdd2.mapPartitions { it =>
      val rv2b = new RegionValueBuilder()
      val rv2 = RegionValue()
      it.flatMap { rv =>
        val rvEnd = rv.region.size
        rv2b.set(rv.region)
        val gsOffset = localRowType.loadField(rv, 3)
        (0 until localNSamples).iterator
          .filter { i =>
            gsType.isElementDefined(rv.region, gsOffset, i)
          }
          .map { i =>
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

  def writeMetadata(dirname: String, partitionCounts: Array[Long]) {
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
      version = MatrixTable.fileVersion,
      split = None,
      sample_schema = sSignature.toPrettyString(compact = true),
      sample_annotation_schema = saSignature.toPrettyString(compact = true),
      variant_schema = vSignature.toPrettyString(compact = true),
      variant_annotation_schema = vaSignature.toPrettyString(compact = true),
      genotype_schema = genotypeSignature.toPrettyString(compact = true),
      global_schema = globalSignature.toPrettyString(compact = true),
      sample_annotations = sampleAnnotationsJ,
      global_annotation = globalJ,
      n_partitions = partitionCounts.length,
      partition_counts = Some(partitionCounts))

    hConf.writeTextFile(dirname + "/metadata.json.gz")(out =>
      Serialization.write(metadata, out))
  }

  def coalesce(k: Int, shuffle: Boolean = true): MatrixTable = copy2(rdd2 = rdd2.coalesce(k, shuffle))

  def persist(storageLevel: String = "MEMORY_AND_DISK"): MatrixTable = {
    val level = try {
      StorageLevel.fromString(storageLevel)
    } catch {
      case e: IllegalArgumentException =>
        fatal(s"unknown StorageLevel `$storageLevel'")
    }

    copy2(rdd2 = rdd2.persist(level))
  }

  def cache(): MatrixTable = persist("MEMORY_ONLY")

  def unpersist(): MatrixTable = copy2(rdd2 = rdd2.unpersist())

  def naiveCoalesce(maxPartitions: Int): MatrixTable =
    copy2(rdd2 = rdd2.naiveCoalesce(maxPartitions))

  /**
    * @param filterExpr filter expression involving v (Variant), va (variant annotations), s (sample),
    *                   sa (sample annotations), and g (genotype annotation), which returns a boolean value
    * @param keep       keep genotypes where filterExpr evaluates to true
    */
  def filterGenotypes(filterExpr: String, keep: Boolean = true): MatrixTable = {

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

    val partitionCounts = rdd2.rdd.writeRows(dirname, rowType)

    writeMetadata(dirname, partitionCounts)

    hadoopConf.writeTextFile(dirname + "/partitioner.json.gz") { out =>
      Serialization.write(rdd2.partitioner.toJSON, out)
    }

    hadoopConf.writeTextFile(dirname + "/_SUCCESS")(out => ())
  }

  def linreg(ysExpr: Array[String], xExpr: String, covExpr: Array[String] = Array.empty[String], root: String = "va.linreg", variantBlockSize: Int = 16): MatrixTable = {
    LinearRegression(this, ysExpr, xExpr, covExpr, root, variantBlockSize)
  }

  def logreg(test: String,
    y: String, x: String, covariates: Array[String] = Array.empty[String],
    root: String = "va.logreg"): MatrixTable = {
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
    optDroppedVarianceFraction: Option[Double] = None): MatrixTable = {
    LinearMixedRegression(this, kinshipMatrix, y, x, covariates, useML, rootGA, rootVA,
      runAssoc, delta, sparsityThreshold, nEigs, optDroppedVarianceFraction)
  }

  def skat(keyExpr: String,
    weightExpr: String,
    y: String,
    x: String,
    covariates: Array[String] = Array.empty[String],
    logistic: Boolean = false,
    maxSize: Int = 46340, // floor(sqrt(Int.MaxValue))
    accuracy: Double = 1e-6,
    iterations: Int = 10000): Table = {
    Skat(this, keyExpr, weightExpr, y, x, covariates, logistic, maxSize, accuracy, iterations)
  }

  def minRep(leftAligned: Boolean = false): MatrixTable = {
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
        OrderedRVD(rdd2.typ,
          rdd2.partitioner,
          minRep1(removeLeftAligned = false, removeMoving = false, verifyLeftAligned = true))
      else
        SplitMulti.unionMovedVariants(
          OrderedRVD(rdd2.typ,
            rdd2.partitioner,
            minRep1(removeLeftAligned = false, removeMoving = true, verifyLeftAligned = false)),
          minRep1(removeLeftAligned = true, removeMoving = false, verifyLeftAligned = false))

    copy2(rdd2 = newRDD2)
  }

  def trioMatrix(pedigree: Pedigree, completeTrios: Boolean): MatrixTable = {
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

    val newRDD = rdd2.mapPartitionsPreservesPartitioning(new OrderedRVType(rdd2.typ.partitionKey, rdd2.typ.key, newRowType)) { it =>
      val region = Region()
      val rvb = new RegionValueBuilder(region)
      val rv = RegionValue(region)
      it.map { r =>
        region.clear()
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
        rv.setOffset(rvb.end())
        rv
      }
    }

    copy2(rdd2 = newRDD,
      sampleIds = kidIds,
      sampleAnnotations = newSampleAnnotations,
      saSignature = newSaSignature,
      genotypeSignature = newEntryType)
  }

  def toIndexedRowMatrix(expr: String, getVariants: Boolean): (IndexedRowMatrix, Option[Array[Any]]) = {
    val partStarts = partitionStarts()
    assert(partStarts.length == rdd2.getNumPartitions + 1)
    val partStartsBc = sparkContext.broadcast(partStarts)

    val localRowType = rowType
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    val ec = EvalContext(Map(
      "global" -> (0, globalSignature),
      "v" -> (1, vSignature),
      "va" -> (2, vaSignature),
      "s" -> (3, sSignature),
      "sa" -> (4, saSignature),
      "g" -> (5, genotypeSignature)))
    val f = RegressionUtils.parseExprAsDouble(expr, ec)
    ec.set(0, globalAnnotation)

    val indexedRows = rdd2.mapPartitionsWithIndex { case (i, it) =>
      val start = partStartsBc.value(i)
      var j = 0
      val ur = new UnsafeRow(localRowType)
      it.map { rv =>
        ur.set(rv)
        ec.set(1, ur.get(1))
        ec.set(2, ur.get(2))
        val gs = ur.getAs[IndexedSeq[Any]](3)
        val ns = gs.length
        val a = new Array[Double](ns)
        var k = 0
        while (k < ns) {
          ec.set(3, localSampleIdsBc.value(k))
          ec.set(4, localSampleAnnotationsBc.value(k))
          ec.set(5, gs(k))
          a(k) = f() match {
            case null => fatal(s"Entry expr must be non-missing. Found missing value for sample ${ localSampleIdsBc.value(k) } and variant ${ ur.get(1) }")
            case t => t.toDouble
          }
          k += 1
        }
        val row = IndexedRow(start + j, Vectors.dense(a))
        j += 1
        row
      }
    }

    val irm = new IndexedRowMatrix(indexedRows, partStarts.last, nSamples)

    val optionVariants =
      someIf(getVariants,
        rdd2.mapPartitions { it =>
          val ur = new UnsafeRow(localRowType)
          it.map { rv =>
            ur.set(rv)
            ur.get(1)
          }
        }.collect())

    (irm, optionVariants)
  }

  def writeBlockMatrix(dirname: String, expr: String, blockSize: Int): Unit = {
    val partStarts = partitionStarts()
    assert(partStarts.length == rdd2.getNumPartitions + 1)

    val ec = EvalContext(Map(
      "global" -> (0, globalSignature),
      "v" -> (1, vSignature),
      "va" -> (2, vaSignature),
      "s" -> (3, sSignature),
      "sa" -> (4, saSignature),
      "g" -> (5, genotypeSignature)))
    val f = RegressionUtils.parseExprAsDouble(expr, ec)
    ec.set(0, globalAnnotation)

    val nRows = partStarts.last
    val nCols = nSamples

    val hadoop = sparkContext.hadoopConfiguration
    hadoop.mkDir(dirname)

    // write metadata
    hadoop.writeDataFile(dirname + BlockMatrix.metadataRelativePath) { os =>
      jackson.Serialization.write(
        BlockMatrixMetadata(blockSize, nRows, nCols),
        os)
    }

    // write blocks
    hadoop.mkDir(dirname + "/parts")
    val gp = GridPartitioner(blockSize, nRows, nCols)
    val blockCount =
      new WriteBlocksRDD(dirname, rdd2, sparkContext, rowType, sampleIdsBc, sampleAnnotationsBc, partStarts, f, ec, gp)
        .reduce(_ + _)

    assert(blockCount == gp.numPartitions)
    info(s"Wrote all $blockCount blocks of $nRows x $nCols matrix with block size $blockSize.")

    hadoop.writeTextFile(dirname + "/_SUCCESS")(out => ())
  }

  // FIXME remove when filter_alleles tests are migrated to Python
  def filterAlleles(filterExpr: String, variantExpr: String = "",
    keep: Boolean = true, subset: Boolean = true, leftAligned: Boolean = false, keepStar: Boolean = false): MatrixTable = {
    if (!genotypeSignature.isOfType(Genotype.htsGenotypeType))
      fatal(s"filter_alleles: genotype_schema must be the HTS genotype schema, found: ${ genotypeSignature }")

    val genotypeExpr = if (subset) {
      """
g = let newpl = if (isDefined(g.PL))
        let unnorm = range(newV.nGenotypes).map(newi =>
            let oldi = gtIndex(newToOld[gtj(newi)], newToOld[gtk(newi)])
             in g.PL[oldi]) and
            minpl = unnorm.min()
         in unnorm - minpl
      else
        NA: Array[Int] and
    newgt = gtFromPL(newpl) and
    newad = if (isDefined(g.AD))
        range(newV.nAlleles).map(newi => g.AD[newToOld[newi]])
      else
        NA: Array[Int] and
    newgq = gqFromPL(newpl) and
    newdp = g.DP
 in { GT: Call(newgt), AD: newad, DP: newdp, GQ: newgq, PL: newpl }
        """
    } else {
      // downcode
      s"""
g = let newgt = gtIndex(oldToNew[gtj(g.GT)], oldToNew[gtk(g.GT)]) and
    newad = if (isDefined(g.AD))
        range(newV.nAlleles).map(i => range(v.nAlleles).filter(j => oldToNew[j] == i).map(j => g.AD[j]).sum())
      else
        NA: Array[Int] and
    newdp = g.DP and
    newpl = if (isDefined(g.PL))
        range(newV.nGenotypes).map(gi => range(v.nGenotypes).filter(gj => gtIndex(oldToNew[gtj(gj)], oldToNew[gtk(gj)]) == gi).map(gj => g.PL[gj]).min())
      else
        NA: Array[Int] and
    newgq = gqFromPL(newpl)
 in { GT: Call(newgt), AD: newad, DP: newdp, GQ: newgq, PL: newpl }
        """
    }

    FilterAlleles(this, filterExpr, variantExpr, genotypeExpr,
      keep = keep, leftAligned = leftAligned, keepStar = keepStar)
  }

  def indexRows(name: String): MatrixTable = {
    val path = List("va", name)

    val (newRowType, inserter) = rowType.unsafeInsert(TInt64(), path)
    val newVAType = newRowType.asInstanceOf[TStruct].fieldType(2)
    val localRowType = rowType

    val partitionStarts = partitionCounts().scanLeft(0L)(_ + _)
    val newMatrixType = matrixType.copy(vaType = newVAType)

    val numberedRDD = rdd2.mapIndexedPartitionsPreservesPartitioning(newMatrixType.orderedRVType) { case (i, it) =>
      val region2 = Region()
      val rv2 = RegionValue(region2)
      val rv2b = new RegionValueBuilder(region2)

      var idx = partitionStarts(i)

      it.map { rv =>
        region2.clear()
        rv2b.start(newRowType)
        inserter(rv.region, rv.offset, rv2b, () => rv2b.addLong(idx))
        idx += 1
        rv2.setOffset(rv2b.end())
        rv2
        }
      }
    copy2(vaSignature = newVAType, rdd2 = numberedRDD)
  }

  def indexCols(name: String): MatrixTable = {
    val path = List(name)
    val (newColType, inserter) = saSignature.insert(TInt32(), path)
    val newSampleAnnotations = Array.tabulate(nSamples) { i =>
      inserter(sampleAnnotations(i), i)
    }
    copy2(saSignature = newColType, sampleAnnotations = newSampleAnnotations)
  }
}
