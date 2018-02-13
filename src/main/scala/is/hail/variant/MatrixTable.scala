package is.hail.variant

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.distributedmatrix._
import is.hail.expr._
import is.hail.methods._
import is.hail.rvd._
import is.hail.table.{Table, TableSpec}
import is.hail.methods.Aggregators.SampleFunctions
import is.hail.stats.RegressionUtils
import is.hail.utils._
import is.hail.{HailContext, utils}
import is.hail.expr.types._
import is.hail.io.CodecSpec
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

abstract class ComponentSpec

object RelationalSpec {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(List(
      classOf[ComponentSpec], classOf[RVDComponentSpec], classOf[PartitionCountsComponentSpec],
      classOf[RelationalSpec], classOf[TableSpec], classOf[MatrixTableSpec]))
    override val typeHintFieldName = "name"
  } +
    new TableTypeSerializer +
    new MatrixTypeSerializer

  def read(hc: HailContext, path: String): RelationalSpec = {
    val metadataFile = path + "/metadata.json.gz"
    val jv = hc.hadoopConf.readFile(metadataFile) { in => parse(in) }

    val fileVersion = jv \ "file_version" match {
      case JInt(rep) => SemanticVersion(rep.toInt)
      case _ =>
        fatal(s"metadata does not contain file version: $metadataFile")
    }

    if (!FileFormat.version.supports(fileVersion))
      fatal(s"incompatible file format when reading: $path\n  supported version: ${ FileFormat.version }, found $fileVersion")

    val referencesRelPath = jv \ "references_rel_path" match {
      case JString(p) => p
    }

    GenomeReference.importReferences(hc.hadoopConf, path + "/" + referencesRelPath)

    jv.extract[RelationalSpec]
  }
}

abstract class RelationalSpec {
  def file_version: Int

  def hail_version: String

  def components: Map[String, ComponentSpec]

  def getComponent[T <: ComponentSpec](name: String): T = components(name).asInstanceOf[T]

  def globalsComponent: RVDComponentSpec = getComponent[RVDComponentSpec]("globals")

  def partitionCounts: Array[Long] = getComponent[PartitionCountsComponentSpec]("partition_counts").counts

  def write(hc: HailContext, path: String) {
    hc.hadoopConf.writeTextFile(path + "/metadata.json.gz") { out =>
      implicit val formats = RelationalSpec.formats
      Serialization.write(this, out)
    }
  }
}

case class RVDComponentSpec(rel_path: String) extends ComponentSpec {
  def read(hc: HailContext, path: String): RVD = {
    val rvdPath = path + "/" + rel_path
    RVDSpec.read(hc, rvdPath)
      .read(hc, rvdPath)
  }

  def readLocal(hc: HailContext, path: String): IndexedSeq[Row] = {
    val rvdPath = path + "/" + rel_path
    RVDSpec.read(hc, rvdPath)
      .readLocal(hc, rvdPath)
  }
}

case class PartitionCountsComponentSpec(counts: Array[Long]) extends ComponentSpec

case class MatrixTableSpec(
  file_version: Int,
  hail_version: String,
  references_rel_path: String,
  matrix_type: MatrixType,
  components: Map[String, ComponentSpec]) extends RelationalSpec {
  def colsComponent: RVDComponentSpec = getComponent[RVDComponentSpec]("cols")

  def rowsComponent: RVDComponentSpec = getComponent[RVDComponentSpec]("rows")

  def entriesComponent: RVDComponentSpec = getComponent[RVDComponentSpec]("entries")
}

object FileFormat {
  val version: SemanticVersion = SemanticVersion(1, 0, 0)
}

object MatrixTable {
  def read(hc: HailContext, path: String,
    dropSamples: Boolean = false, dropVariants: Boolean = false): MatrixTable = {
    val spec = RelationalSpec.read(hc, path).asInstanceOf[MatrixTableSpec]
    new MatrixTable(hc,
      MatrixRead(path, spec, dropSamples, dropVariants))
  }

  def fromLegacy[T](hc: HailContext,
    matrixType: MatrixType,
    globals: Annotation,
    colValues: IndexedSeq[Annotation],
    rdd: RDD[(Annotation, Iterable[T])]): MatrixTable = {

    val localGType = matrixType.entryType
    val localRVRowType = matrixType.rvRowType

    assert(matrixType.rowPartitionKey == IndexedSeq("v"))
    assert(matrixType.rowKey == IndexedSeq("v"))
    val localNSamples = colValues.length

    var ds = new MatrixTable(hc, matrixType, globals, colValues,
      OrderedRVD(matrixType.orvdType,
        rdd.mapPartitions { it =>
          val region = Region()
          val rvb = new RegionValueBuilder(region)
          val rv = RegionValue(region)

          it.map { case (va, gs) =>
            val vaRow = va.asInstanceOf[Row]
            assert(matrixType.rowType.typeCheck(vaRow), s"${ matrixType.rowType }, $vaRow")

            region.clear()
            rvb.start(localRVRowType)
            rvb.startStruct()
            var i = 0
            while (i < vaRow.length) {
              rvb.addAnnotation(localRVRowType.fieldType(i), vaRow.get(i))
              i += 1
            }
            rvb.startArray(localNSamples) // gs
            gs.foreach { g => rvb.addAnnotation(localGType, g) }
            rvb.endArray() // gs
            rvb.endStruct()
            rv.setOffset(rvb.end())

            rv
          }
        }, None, None))
    if (matrixType.rowType.fieldByName("v").typ.isInstanceOf[TVariant])
      ds = ds.annotateVariantsExpr("locus = va.v.locus, alleles = [va.v.ref].extend(va.v.altAlleles.map(x => x.alt))")
        .keyRowsBy(Array("locus", "alleles"), Array("locus"))
        .dropRows("v")
    ds.typecheck()
    ds
  }

  def gen(hc: HailContext, gen: VSMSubgen): Gen[MatrixTable] =
    gen.gen(hc)

  def genGeneric(hc: HailContext): Gen[MatrixTable] =
    VSMSubgen(
      sSigGen = Type.genArb,
      saSigGen = Type.genInsertableStruct,
      vSigGen = GenomeReference.gen.map(TVariant(_)),
      vaSigGen = Type.genInsertableStruct,
      globalSigGen = Type.genInsertableStruct,
      tSigGen = Type.genInsertableStruct,
      sGen = (t: Type) => t.genNonmissingValue,
      saGen = (t: Type) => t.genValue,
      vaGen = (t: Type) => t.genValue,
      globalGen = (t: Type) => t.genNonmissingValue,
      vGen = (t: Type) => t.genNonmissingValue,
      tGen = (t: Type, v: Annotation) => t.genValue.resize(20))
      .gen(hc)

  def checkDatasetSchemasCompatible(datasets: Array[MatrixTable]) {
    val first = datasets(0)
    val vaSchema = first.rowType
    val genotypeSchema = first.entryType
    val rowKeySchema = first.rowKeyTypes
    val nPartitionKeys = first.rowPartitionKey.length
    val colKeySchema = first.colKeyTypes
    val colKeys = first.colKeys

    datasets.indices.tail.foreach { i =>
      val vds = datasets(i)
      val vas = vds.rowType
      val gsig = vds.entryType
      val vsig = vds.rowKeyTypes
      val nrpk = vds.rowPartitionKey.length
      val ssig = vds.colKeyTypes
      val cks = vds.colKeys

      if (!ssig.sameElements(colKeySchema)) {
        fatal(
          s"""cannot combine datasets with incompatible column keys
             |  Schema in datasets[0]: @1
             |  Schema in datasets[$i]: @2""".stripMargin,
          colKeySchema.map(_.toString).mkString(", "),
          ssig.map(_.toString).mkString(", ")
        )
      } else if (!vsig.sameElements(rowKeySchema)) {
        fatal(
          s"""cannot combine datasets with different row key schemata
             |  Schema in datasets[0]: @1
             |  Schema in datasets[$i]: @2""".stripMargin,
          rowKeySchema.toString,
          vsig.toString
        )
      } else if (nrpk != nPartitionKeys) {
        fatal(
          s"""cannot combine datasets with different partition keys""")
      } else if (colKeys != cks) {
        fatal(
          s"""cannot combine datasets with different column identifiers or ordering
             |  IDs in datasets[0]: @1
             |  IDs in datasets[$i]: @2""".stripMargin, colKeys, cks)
      } else if (vas != vaSchema) {
        fatal(
          s"""cannot combine datasets with different row annotation schemata
             |  Schema in datasets[0]: @1
             |  Schema in datasets[$i]: @2""".stripMargin,
          vaSchema.toString,
          vas.toString
        )
      } else if (gsig != genotypeSchema) {
        fatal(
          s"""cannot read datasets with different cell schemata
             |  Schema in datasets[0]: @1
             |  Schema in datasets[$i]: @2""".stripMargin,
          genotypeSchema.toString,
          gsig.toString
        )
      }
    }
  }

  def unionRows(datasets: java.util.ArrayList[MatrixTable]): MatrixTable =
    unionRows(datasets.asScala.toArray)

  def unionRows(datasets: Array[MatrixTable]): MatrixTable = {
    require(datasets.length >= 2)
    val first = datasets(0)
    val sc = first.sparkContext

    checkDatasetSchemasCompatible(datasets)

    first.copyMT(
      rvd = OrderedRVD(
        first.rvd.typ,
        sc.union(datasets.map(_.rvd.rdd)),
        None, None))
  }

  def fromRowsTable(kt: Table): MatrixTable = {
    val matrixType = MatrixType.fromParts(
      kt.globalSignature,
      Array.empty[String],
      TStruct.empty(),
      kt.key,
      kt.key,
      kt.signature,
      TStruct.empty()
    )
    val rvRowType = matrixType.rvRowType

    val oldRowType = kt.signature

    val rdd = kt.rvd.mapPartitions { it =>
      val rvb = new RegionValueBuilder()
      val rv2 = RegionValue()

      it.map { rv =>
        rvb.set(rv.region)
        rvb.start(rvRowType)
        rvb.startStruct()
        rvb.addAllFields(oldRowType, rv)
        rvb.startArray(0) // gs
        rvb.endArray()
        rvb.endStruct()
        rv2.set(rv.region, rvb.end())
        rv2
      }
    }

    new MatrixTable(kt.hc, matrixType, Annotation.empty, Array.empty[Annotation],
      OrderedRVD(matrixType.orvdType, rdd, None, None))
  }
}

case class VSMSubgen(
  sSigGen: Gen[Type],
  saSigGen: Gen[TStruct],
  vSigGen: Gen[Type],
  vaSigGen: Gen[TStruct],
  globalSigGen: Gen[TStruct],
  tSigGen: Gen[TStruct],
  sGen: (Type) => Gen[Annotation],
  saGen: (TStruct) => Gen[Annotation],
  vaGen: (TStruct) => Gen[Annotation],
  globalGen: (TStruct) => Gen[Annotation],
  vGen: (Type) => Gen[Annotation],
  tGen: (TStruct, Annotation) => Gen[Annotation]) {

  def gen(hc: HailContext): Gen[MatrixTable] =
    for (size <- Gen.size;
    (l, w) <- Gen.squareOfAreaAtMostSize.resize((size / 3 / 10) * 8);

    vSig <- vSigGen.resize(3);
    vaSig <- vaSigGen.map(t => t.deepOptional().asInstanceOf[TStruct]).resize(3);
    sSig <- sSigGen.resize(3);
    saSig <- saSigGen.map(t => t.deepOptional().asInstanceOf[TStruct]).resize(3);
    globalSig <- globalSigGen.resize(5);
    tSig <- tSigGen.map(t => t.structOptional().asInstanceOf[TStruct]).resize(3);
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
        val (finalSaSchema, ins) = saSig.structInsert(sSig, List("s"))
        val (finalVaSchema, ins2) = vaSig.structInsert(vSig, List("v"))
        MatrixTable.fromLegacy(hc,
          MatrixType.fromParts(globalSig, Array("s"), finalSaSchema, Array("v"), Array("v"), finalVaSchema, tSig),
          global,
          sampleIds.zip(saValues).map { case (id, sa) => ins(sa, id) },
          hc.sc.parallelize(rows.map { case (v, (va, gs)) => (ins2(va, v), gs) }, nPartitions))
          .deduplicate()
      }
}

object VSMSubgen {
  val random = VSMSubgen(
    sSigGen = Gen.const(TString()),
    saSigGen = Type.genInsertable,
    vSigGen = GenomeReference.gen.map(TVariant(_)),
    vaSigGen = Type.genInsertable,
    globalSigGen = Type.genInsertable,
    tSigGen = Gen.const(Genotype.htsGenotypeType),
    sGen = (t: Type) => Gen.identifier.map(s => s: Annotation),
    saGen = (t: Type) => t.genValue,
    vaGen = (t: Type) => t.genValue,
    globalGen = (t: Type) => t.genNonmissingValue,
    vGen = (t: Type) => t.genNonmissingValue,
    tGen = (t: Type, v: Annotation) => Genotype.genExtreme(v.asInstanceOf[Variant]))

  val plinkSafeBiallelic = random.copy(
    vSigGen = Gen.const(TVariant(GenomeReference.GRCh37)),
    sGen = (t: Type) => Gen.plinkSafeIdentifier,
    vGen = (t: Type) => VariantSubgen.plinkCompatible.copy(nAllelesGen = Gen.const(2),
      contigGen = Contig.gen(t.asInstanceOf[TVariant].gr.asInstanceOf[GenomeReference])).gen)

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
    vGen = (t: Type) => t.genNonmissingValue,
    tGen = (t: Type, v: Annotation) => Genotype.genGenericDosageGenotype(v.asInstanceOf[Variant]))

  val realistic = random.copy(
    tGen = (t: Type, v: Annotation) => Genotype.genRealistic(v.asInstanceOf[Variant]))
}

class MatrixTable(val hc: HailContext, val ast: MatrixIR) {

  def this(hc: HailContext,
    matrixType: MatrixType,
    globals: Annotation,
    colValues: IndexedSeq[Annotation],
    rdd2: OrderedRVD) =
    this(hc,
      MatrixLiteral(
        matrixType,
        MatrixValue(matrixType, globals, colValues, rdd2)))

  def requireRowKeyVariant(method: String) {
    rowKey.zip(rowKeyTypes) match {
      case IndexedSeq(("locus", TLocus(_, _)), ("alleles", TArray(TString(_), _))) =>
      case _ =>
        fatal(s"in $method: row key (variant) schema must be ('locus': Locus, 'alleles': Array[String], found: ${
          rowKey.zip(rowKeyTypes).mkString(", ")
        }")
    }
  }

  def requireColKeyString(method: String) {
    colKeyTypes match {
      case Array(_: TString) =>
      case t =>
        fatal(s"in $method: column key schema must be String, found: $t")
    }
  }

  def genomeReference: GenomeReference = {
    val firstKeyField = rowKeyTypes(0)
    firstKeyField match {
      case TLocus(gr: GenomeReference, _) => gr
    }
  }

  val matrixType: MatrixType = ast.typ

  val colType: TStruct = matrixType.colType
  val rowType: TStruct = matrixType.rowType
  val entryType: TStruct = matrixType.entryType
  val globalType: TStruct = matrixType.globalType

  val rvRowType: TStruct = matrixType.rvRowType
  val rowKey: IndexedSeq[String] = matrixType.rowKey
  val rowPartitionKey: IndexedSeq[String] = matrixType.rowPartitionKey
  val entriesIndex: Int = matrixType.entriesIdx

  val colKey: IndexedSeq[String] = matrixType.colKey

  def colKeyTypes: Array[Type] = colKey
    .map(s => matrixType.colType.fieldType(matrixType.colType.fieldIdx(s)))
    .toArray

  val rowKeyTypes: Array[Type] = rowKey
    .map(s => matrixType.rowType.fieldType(matrixType.rowType.fieldIdx(s)))
    .toArray

  val rowKeyStruct: TStruct = TStruct(rowKey.zip(rowKeyTypes): _*)

  val rowPartitionKeyTypes: Array[Type] = rowPartitionKey
    .map(s => matrixType.rowType.fieldType(matrixType.rowType.fieldIdx(s)))
    .toArray

  lazy val value: MatrixValue = {
    val opt = MatrixIR.optimize(ast)
    val v = opt.execute(hc)
    assert(v.rvd.typ == matrixType.orvdType, s"\n${ v.rvd.typ }\n${ matrixType.orvdType }")
    v
  }

  lazy val MatrixValue(_, globals, colValues, rvd) = value

  def partitionCounts(): Array[Long] = {
    ast.partitionCounts match {
      case Some(counts) => counts
      case None => rvd.countPerPartition()
    }
  }

  // length nPartitions + 1, first element 0, last element rdd2 count
  def partitionStarts(): Array[Long] = partitionCounts().scanLeft(0L)(_ + _)

  def colKeys: IndexedSeq[Annotation] = {
    val queriers = colKey.map(colType.query(_))
    colValues.map(a => Row.fromSeq(queriers.map(q => q(a)))).toArray[Annotation]
  }

  def rowKeysF: (Row) => Row = {
    val localRowType = rowType
    val queriers = rowKey.map(localRowType.query(_)).toArray
    (r: Row) => Row.fromSeq(queriers.map(_ (r)))
  }

  def keyRowsBy(keys: java.util.ArrayList[String], partitionKeys: java.util.ArrayList[String]): MatrixTable =
    keyRowsBy(keys.asScala.toArray, partitionKeys.asScala.toArray)

  def keyRowsBy(keys: Array[String], partitionKeys: Array[String]): MatrixTable = {
    require(keys.nonEmpty)
    require(partitionKeys.nonEmpty)
    val rowFields = rowType.fieldNames.toSet
    assert(keys.forall(rowFields.contains), s"${ keys.filter(k => !rowFields.contains(k)).mkString(", ") }")
    assert(partitionKeys.forall(rowFields.contains))

    val newMatrixType = matrixType.copy(rowKey = keys,
      rowPartitionKey = partitionKeys)

    copyMT(matrixType = newMatrixType,
      rvd = OrderedRVD(newMatrixType.orvdType, rvd.rdd, None, None))
  }

  def keyColsBy(keys: java.util.ArrayList[String]): MatrixTable = keyColsBy(keys.asScala: _*)

  def keyColsBy(keys: String*): MatrixTable = {
    val colFields = colType.fieldNames.toSet
    assert(keys.forall(colFields.contains))
    copyMT(matrixType = matrixType.copy(colKey = keys.toArray[String]))
  }

  def stringSampleIds: IndexedSeq[String] = {
    assert(colKeyTypes.length == 1 && colKeyTypes(0).isInstanceOf[TString], colKeyTypes.toSeq)
    val querier = colType.query(colKey(0))
    colValues.map(querier(_).asInstanceOf[String])
  }

  def stringSampleIdSet: Set[String] = stringSampleIds.toSet

  lazy val colValuesBc = {
    val saArrayType = TArray(colType, required = true)
    sparkContext.broadcast(UnsafeIndexedSeq(saArrayType, colValues))
  }

  def requireUniqueSamples(method: String) {
    val dups = stringSampleIds.counter().filter(_._2 > 1).toArray
    if (dups.nonEmpty)
      fatal(s"Method '$method' does not support duplicate sample IDs. Duplicates:" +
        s"\n  @1", dups.sortBy(-_._2).map { case (id, count) => s"""($count) "$id"""" }.truncatable("\n  "))
  }

  def unsafeRowRDD(): RDD[UnsafeRow] = {
    val localRVRowType = rvRowType
    rvd.map { rv =>
      new UnsafeRow(localRVRowType, rv.region.copy(), rv.offset)
    }
  }

  def collect(): Array[UnsafeRow] = unsafeRowRDD().collect()

  def take(n: Int): Array[UnsafeRow] = unsafeRowRDD().take(n)

  def groupSamplesBy(keyExpr: String, aggExpr: String): MatrixTable = {
    val sEC = EvalContext(Map(Annotation.GLOBAL_HEAD -> (0, globalType),
      Annotation.SAMPLE_HEAD -> (1, colType)))
    val (keyNames, keyTypes, keyFs) = Parser.parseNamedExprs(keyExpr, sEC)
    sEC.set(0, globals)

    val keysBySample = colValues.map { sa =>
      sEC.set(1, sa)
      Row.fromSeq(keyFs())
    }
    val newKeys = keysBySample.toSet.toArray
    val keyMap = newKeys.zipWithIndex.toMap
    val samplesMap = keysBySample.map { k => if (k == null) -1 else keyMap(k) }.toArray

    val nKeys = newKeys.size

    val ec = variantEC
    val (newEntryNames, newEntryTypes, entryF) = Parser.parseNamedExprs(aggExpr, ec)
    val newEntryType = TStruct(newEntryNames.zip(newEntryTypes): _*)

    val newColKey = keyNames
    val newColType = TStruct(keyNames.zip(keyTypes): _*)

    val aggregate = Aggregators.buildVariantAggregationsByKey(this, nKeys, samplesMap, ec)

    insertEntries(noOp,
      newColType = newColType,
      newColValues = newKeys,
      newColKey = newColKey)(newEntryType, { case (_, rv, rvb) =>

      val aggArr = aggregate(rv)
      rvb.startArray(nKeys)

      var i = 0
      while (i < nKeys) {
        aggArr(i)()
        rvb.startStruct()
        val fields = entryF()
        var j = 0
        while (j < fields.length) {
          rvb.addAnnotation(newEntryType.fieldType(j), fields(j))
          j += 1
        }
        rvb.endStruct()
        i += 1
      }
      rvb.endArray()
    })
  }

  def groupVariantsBy(keyExpr: String, aggExpr: String, partitionKey: java.util.ArrayList[String]): MatrixTable =
    groupVariantsBy(keyExpr, aggExpr, Option(partitionKey).map(_.asScala.toArray))

  def groupVariantsBy(keyExpr: String, aggExpr: String, partitionKey: Option[Array[String]] = None): MatrixTable = {
    val fullRowType = rvRowType
    val vEC = EvalContext(Map(Annotation.GLOBAL_HEAD -> (0, globalType),
      "va" -> (1, rowType)))
    vEC.set(0, globals)
    val (keyNames, keyTypes, keyFs) = Parser.parseNamedExprs(keyExpr, vEC)
    partitionKey.foreach(pk => assert(keyNames.startsWith(pk)))
    partitionKey.foreach(pk => assert(pk.nonEmpty))

    val keyStruct = TStruct(keyNames.zip(keyTypes): _*)

    val SampleFunctions(zero, seqOp, combOp, resultOp, newEntryType) = Aggregators.makeSampleFunctions(this, aggExpr)
    val pk: IndexedSeq[String] = partitionKey.map(_.toFastIndexedSeq).getOrElse(keyNames.toFastIndexedSeq)
    val newMatrixType = MatrixType.fromParts(globalType, colKey, colType,
      pk, keyNames, keyStruct, newEntryType)

    val localRowType = rowType
    val localEntriesIndex = entriesIndex
    val keyedRDD = rvd.rdd.mapPartitions { it =>
      val fullRow = new UnsafeRow(fullRowType)
      it.map { rv =>
        fullRow.set(rv)
        val row = fullRow.deleteField(localEntriesIndex)
        vEC.set(1, row)
        val k = Annotation.copy(keyStruct, Row.fromSeq(keyFs()))
        k -> rv
      }
    }

    // FIXME: this shuffles twice
    val newRVType = newMatrixType.rvRowType
    val rdd = keyedRDD
      .aggregateByKey(zero)(seqOp, combOp)
      .mapPartitions { it =>
        val region = Region()
        val rv = RegionValue(region)
        val rvb = new RegionValueBuilder(region)
        it.map { case (key, agg) =>
          val k = key.asInstanceOf[Row]
          region.clear()
          rvb.start(newRVType)
          rvb.startStruct()
          var i = 0
          while (i < keyStruct.size) {
            rvb.addAnnotation(keyStruct.fieldType(i), k.get(i))
            i += 1
          }
          resultOp(agg, rvb)
          rvb.endStruct()
          rv.setOffset(rvb.end())
          rv
        }
      }

    copyMT(rvd = OrderedRVD(newMatrixType.orvdType, rdd, None, None),
      matrixType = newMatrixType)
  }

  def annotateGlobal(a: Annotation, t: Type, code: String): MatrixTable = {
    val (newT, i) = insertGlobal(t, Parser.parseAnnotationRoot(code, Annotation.GLOBAL_HEAD))
    copyMT(matrixType = matrixType.copy(globalType = newT), globals = i(globals, a))
  }

  /**
    * Create and destroy global annotations with expression language.
    *
    * @param expr Annotation expression
    */
  def annotateGlobalExpr(expr: String): MatrixTable = {
    val ec = EvalContext(Map(
      "global" -> (0, globalType)))

    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Option(Annotation.GLOBAL_HEAD))

    val inserterBuilder = new ArrayBuilder[Inserter]()

    val finalType = (paths, types).zipped.foldLeft(globalType) { case (v, (ids, signature)) =>
      val (s, i) = v.structInsert(signature, ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    ec.set(0, globals)
    val ga = inserters
      .zip(f())
      .foldLeft(globals) { case (a, (ins, res)) =>
        ins(a, res)
      }

    copyMT(matrixType = matrixType.copy(globalType = finalType), globals = ga)

  }

  def insertGlobal(sig: Type, path: List[String]): (TStruct, Inserter) = {
    globalType.structInsert(sig, path)
  }

  def annotateSamples(signature: Type, path: List[String], annotations: Array[Annotation]): MatrixTable = {
    val (t, ins) = insertSA(signature, path)

    val newAnnotations = new Array[Annotation](numCols)

    for (i <- colValues.indices) {
      newAnnotations(i) = ins(colValues(i), annotations(i))
      t.typeCheck(newAnnotations(i))
    }

    copyMT(matrixType = matrixType.copy(colType = t), colValues = newAnnotations)
  }

  def annotateSamplesExpr(expr: String): MatrixTable = {
    val ec = sampleEC

    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, None)

    val inserterBuilder = new ArrayBuilder[Inserter]()
    val finalType = (paths, types).zipped.foldLeft(colType) { case (sas, (ids, signature)) =>
      val (s, i) = sas.structInsert(signature, ids)
      inserterBuilder += i
      s
    }
    val inserters = inserterBuilder.result()

    val sampleAggregationOption = Aggregators.buildSampleAggregations(hc, value, ec)

    ec.set(0, globals)

    val newAnnotations = new Array[Annotation](numCols)

    var i = 0
    while (i < numCols) {
      sampleAggregationOption.foreach(_.apply(i))
      val sa = colValues(i)
      ec.set(1, sa)

      newAnnotations(i) = f().zip(inserters)
        .foldLeft(sa) { case (sa, (v, inserter)) =>
          inserter(sa, v)
        }
      i += 1
    }

    val saFields = finalType.fieldNames.toSet
    copyMT(colValues = newAnnotations,
      matrixType = matrixType.copy(colKey = colKey.filter(saFields.contains), colType = finalType))
  }

  def annotateSamples(annotations: Map[Annotation, Annotation], signature: Type, root: String): MatrixTable = {
    val (t, i) = insertSA(signature, List(root))
    annotateSamples(t, i) { case (s, _) => annotations.getOrElse(s, null) }
  }

  def annotateSamplesTable(kt: Table, vdsKey: java.util.ArrayList[String],
    root: String, product: Boolean): MatrixTable =
    annotateSamplesTable(kt, if (vdsKey != null) vdsKey.asScala else null, root, product)

  def annotateSamplesTable(kt: Table, vdsKey: Seq[String] = null,
    root: String = null, product: Boolean = false): MatrixTable = {

    val (finalType, inserter) = colType.structInsert(
      if (product) TArray(kt.valueSignature) else kt.valueSignature,
      List(root))

    val keyTypes = kt.keyFields.map(_.typ).toSeq

    val keyedRDD = kt.keyedRDD()
      .filter { case (k, v) => k.toSeq.forall(_ != null) }

    val nullValue: IndexedSeq[Annotation] = if (product) IndexedSeq() else null

    if (vdsKey != null) {
      val keyEC = EvalContext(Map("sa" -> (0, colType)))
      val (vdsKeyType, vdsKeyFs) = vdsKey.map(Parser.parseExpr(_, keyEC)).unzip

      if (keyTypes != vdsKeyType)
        fatal(
          s"""method `annotateSamplesTable' encountered a mismatch between table keys and computed keys.
             |  Computed keys:  [ ${ vdsKeyType.mkString(", ") } ]
             |  Key table keys: [ ${ keyTypes.mkString(", ") } ]""".stripMargin)

      val keyFuncArray = vdsKeyFs.toArray

      val vdsKeys = colValues.map { sa =>
        keyEC.set(0, sa)
        (Row.fromSeq(keyFuncArray.map(_ ())), ())
      }.toArray

      val thisRDD = sparkContext.parallelize(vdsKeys)
      var r = keyedRDD.join(thisRDD).map { case (k, (tableAnnotation, _)) => (k, tableAnnotation: Annotation) }
      if (product)
        r = r.groupByKey().mapValues(is => (is.toArray[Annotation]: IndexedSeq[Annotation]): Annotation)

      val m = r.collectAsMap()

      annotateSamples(finalType, inserter) { case (_, i) => m.getOrElse(vdsKeys(i)._1, nullValue) }
    } else {
      val ssig = colKeyTypes.toSeq
      keyTypes match {
        case `ssig` =>
          var r = keyedRDD.map { case (k, v) => (k: Annotation, v: Annotation) }

          if (product)
            r = r.groupByKey()
              .map { case (s, rows) => (s, (rows.toArray[Annotation]: IndexedSeq[_]): Annotation) }

          val m = r.collectAsMap()

          annotateSamples(finalType, inserter) { case (ck, _) => m.getOrElse(ck, nullValue) }
        case other =>
          fatal(
            s"""method 'annotate_samples_table' expects a key table keyed by [ ${ colKeyTypes.mkString(",") } ]
               |  Found key [ ${ other.mkString(", ") } ] instead.""".stripMargin)
      }
    }
  }

  def annotateSamples(newSignature: TStruct, inserter: Inserter)(f: (Annotation, Int) => Annotation): MatrixTable = {
    val newAnnotations = colKeys.zip(colValues)
      .zipWithIndex
      .map { case ((ck, sa), i) =>
        val newAnnotation = inserter(sa, f(ck, i))
        newSignature.typeCheck(newAnnotation)
        newAnnotation
      }

    val newFields = newSignature.fieldNames.toSet
    copy2(colValues = newAnnotations, colType = newSignature, colKey = colKey.filter(newFields.contains))
  }

  def annotateVariantsExpr(expr: String): MatrixTable = {
    val localGlobals = globals

    val ec = variantEC
    ec.set(0, globals)
    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, None)

    var newRowType = rowType
    var touchesKeys = false
    val rowKeySet = rowKey.toSet
    val inserters = new Array[Inserter](types.length)
    var i = 0
    while (i < types.length) {
      if (rowKeySet.contains(paths(i).head))
        touchesKeys = true
      val (newSig, ins) = newRowType.structInsert(types(i), paths(i))
      inserters(i) = ins
      newRowType = newSig
      i += 1
    }

    val aggregateOption = Aggregators.buildVariantAggregations(this, ec)

    val newMatrixType = matrixType.copyParts(rowType = newRowType)

    val fullRowType = rvRowType
    val newRVType = newMatrixType.rvRowType
    val localEntriesIndex = entriesIndex

    val mapPartitionsF: (Iterator[RegionValue] => Iterator[RegionValue]) = { it =>
      val fullRow = new UnsafeRow(fullRowType)

      val rvb = new RegionValueBuilder()
      val rv2 = RegionValue()
      it.map { rv =>
        fullRow.set(rv)
        val row = fullRow.deleteField(localEntriesIndex)

        ec.set(1, row)

        aggregateOption.foreach(f => f(rv))

        rvb.set(rv.region)
        rvb.start(newRVType)
        rvb.startStruct()

        var newAnn = row: Annotation
        var i = 0
        var newA = f()
        while (i < newA.length) {
          newAnn = inserters(i)(newAnn, newA(i))
          i += 1
        }

        val newRow = newAnn.asInstanceOf[Row]
        i = 0
        while (i < newRow.size) {
          rvb.addAnnotation(newRowType.fieldType(i), newRow.get(i))
          i += 1
        }

        rvb.addField(fullRowType, rv, localEntriesIndex)
        rvb.endStruct()

        rv2.set(rv.region, rvb.end())
        rv2
      }
    }

    if (touchesKeys) {
      warn("modified row key, rescanning to compute ordering...")
      val newRDD = rvd.mapPartitions(mapPartitionsF)
      copyMT(matrixType = newMatrixType,
        rvd = OrderedRVD(newMatrixType.orvdType, newRDD, None, None))
    } else copyMT(matrixType = newMatrixType,
      rvd = rvd.mapPartitionsPreservesPartitioning(newMatrixType.orvdType)(mapPartitionsF))
  }

  def orderedRVDLeftJoinDistinctAndInsert(right: OrderedRVD, root: String, product: Boolean): MatrixTable = {
    assert(!rowKey.contains(root))
    assert(right.typ.pkType.fieldType.map(_.deepOptional())
      .sameElements(rowPartitionKeyTypes.map(_.deepOptional())))


    val (leftRVD, upcastKeys) = if (right.typ.kType.fieldType.map(_.deepOptional()).sameElements(rowPartitionKeyTypes.map(_.deepOptional()))) {
      (rvd.downcastToPK(), rowKey.drop(rowPartitionKey.length).toArray)
    } else (rvd, Array.empty[String])

    var valueType: Type = right.typ.valueType

    var rightRVD = right
    if (product) {
      valueType = TArray(valueType, required = true)
      rightRVD = rightRVD.groupByKey(" !!! values !!! ")
    }

    val (newRVType, ins) = rvRowType.unsafeStructInsert(valueType, List(root))

    val leftRowType = leftRVD.rowType
    val rightRowType = rightRVD.rowType
    val oldRVType = leftRVD.typ.rowType

    val rightValueIndices = rightRVD.typ.valueIndices
    assert(!product || rightValueIndices.length == 1)

    val newMatrixType = matrixType.copy(rvRowType = newRVType)
    val intermediateMatrixType = newMatrixType.copy(rowKey = newMatrixType.rowPartitionKey)

    copyMT(matrixType = newMatrixType,
      rvd = OrderedRVD(
        newMatrixType.orvdType,
        leftRVD.partitioner,
        leftRVD.orderedJoinDistinct(rightRVD, "left")
          .mapPartitions { it =>
            val rvb = new RegionValueBuilder()
            val rv = RegionValue()

            it.map { jrv =>
              val lrv = jrv.rvLeft

              rvb.set(lrv.region)
              rvb.start(newRVType)
              ins(lrv.region, lrv.offset, rvb,
                () => {
                  if (product) {
                    if (jrv.rvRight == null) {
                      rvb.startArray(0)
                      rvb.endArray()
                    } else
                      rvb.addField(rightRowType, jrv.rvRight, rightValueIndices(0))
                  } else {
                    if (jrv.rvRight == null)
                      rvb.setMissing()
                    else {
                      rvb.startStruct()
                      var i = 0
                      while (i < rightValueIndices.length) {
                        rvb.addField(rightRowType, jrv.rvRight, rightValueIndices(i))
                        i += 1
                      }
                      rvb.endStruct()
                    }
                  }
                })
              rv.set(lrv.region, rvb.end())
              rv
            }
          })
    )
  }

  private def annotateVariantsIntervalTable(kt: Table, root: String, product: Boolean): MatrixTable = {
    assert(rowPartitionKeyTypes.length == 1)
    assert(kt.keySignature.size == 1)
    assert(kt.keySignature.fieldType(0) == TInterval(rowPartitionKeyTypes(0)))

    val typOrdering = rowPartitionKeyTypes(0).ordering

    val typToInsert: Type = if (product) TArray(kt.valueSignature) else kt.valueSignature

    val (newRVType, ins) = rvRowType.unsafeStructInsert(typToInsert, List(root))

    val partBc = sparkContext.broadcast(rvd.partitioner)
    val ktSignature = kt.signature
    val ktKeyFieldIdx = kt.keyFieldIdx(0)
    val ktValueFieldIdx = kt.valueFieldIdx
    val partitionKeyedIntervals = kt.rvd.rdd
      .flatMap { rv =>
        val ur = new UnsafeRow(ktSignature, rv)
        val interval = ur.getAs[Interval](ktKeyFieldIdx)
        if (interval != null) {
          val start = partBc.value.getPartitionPK(Row(interval.start))
          val end = partBc.value.getPartitionPK(Row(interval.end))
          (start to end).view.map(i => (i, rv))
        } else
          Iterator()
      }

    val nParts = rvd.partitions.length
    val zipRDD = partitionKeyedIntervals.partitionBy(new Partitioner {
      def getPartition(key: Any): Int = key.asInstanceOf[Int]

      def numPartitions: Int = nParts
    }).values

    val localRVRowType = rvRowType
    val pkIndex = rvRowType.fieldIdx(rowPartitionKey(0))
    val newRDD = rvd.rdd.zipPartitions(zipRDD, preservesPartitioning = true) { case (it, intervals) =>
      val intervalAnnotations: Array[(BaseInterval[Any], Any)] =
        intervals.map { rv =>
          val ur = new UnsafeRow(ktSignature, rv)
          val interval = ur.getAs[Interval](ktKeyFieldIdx)
          (interval, Row.fromSeq(ktValueFieldIdx.map(ur.get)))
        }.toArray

      val iTree = IntervalTree.annotationTree(typOrdering, intervalAnnotations)

      val rvb = new RegionValueBuilder()
      val rv2 = RegionValue()

      it.map { rv =>
        val ur = new UnsafeRow(localRVRowType, rv)
        val pk = ur.get(pkIndex)
        val queries = iTree.queryValues(typOrdering, pk)
        val value: Annotation = if (product)
          queries: IndexedSeq[Annotation]
        else {
          if (queries.isEmpty)
            null
          else
            queries(0)
        }
        assert(typToInsert.typeCheck(value))

        rvb.set(rv.region)
        rvb.start(newRVType)

        ins(rv.region, rv.offset, rvb, () => rvb.addAnnotation(typToInsert, value))

        rv2.set(rv.region, rvb.end())

        rv2
      }
    }

    val newMatrixType = matrixType.copy(rvRowType = newRVType)

    val newRVD = OrderedRVD(
      newMatrixType.orvdType,
      rvd.partitioner,
      newRDD)

    copyMT(rvd = newRVD, matrixType = newMatrixType)
  }

  def annotateVariantsTable(kt: Table, root: String, product: Boolean = false): MatrixTable = {
    assert(!rowKey.contains(root))

    val keyTypes = kt.keyFields.map(_.typ)
    if (keyTypes.sameElements(rowKeyTypes) || keyTypes.sameElements(rowPartitionKeyTypes)) {
      orderedRVDLeftJoinDistinctAndInsert(
        kt.toOrderedRVD(Some(rvd.partitioner), rowPartitionKey.length),
        root, product)
    } else if (keyTypes.length == 1 &&
      rowPartitionKeyTypes.length == 1 &&
      keyTypes(0) == TInterval(rowPartitionKeyTypes(0))) {
      annotateVariantsIntervalTable(kt, root, product)
    } else {
      fatal(
        s"""method 'annotate_variants_table' expects a key table keyed by one of the following:
           |  [ ${ rowKeyTypes.mkString(", ") } ]
           |  [ ${ rowPartitionKeyTypes.mkString(", ") } ]
           |  Found key [ ${ keyTypes.mkString(", ") } ] instead.""".stripMargin)
    }
  }


  def selectGlobals(fields: java.util.ArrayList[String]): MatrixTable = selectGlobals(fields.asScala.toArray: _*)

  def selectGlobals(fields: String*): MatrixTable = {
    // FIXME: should support nested fields
    val fieldsToKeep = fields.toArray
    val globalFieldSet = globalType.fieldNames.toSet
    val keepSet = fieldsToKeep.toSet
    assert(fieldsToKeep.forall(globalFieldSet.contains))

    val keepIndices = fieldsToKeep.map(globalType.fieldIdx)

    val newGlobalType = TStruct(keepIndices.zip(fieldsToKeep).map { case (i, f) => f -> globalType.fieldType(i) }: _*)

    val newMatrixType = matrixType.copy(globalType = newGlobalType)

    val newGlobals = keepIndices.map(i => globals.asInstanceOf[Row].get(i))

    copyMT(matrixType = newMatrixType, globals = newGlobals)
  }


  def selectCols(selectExprs: java.util.ArrayList[String]): MatrixTable = selectCols(selectExprs.asScala.toArray: _*)

  def selectCols(exprs: String*): MatrixTable = {
    val ec = sampleEC
    val (maybePaths, types, f, isNamedExpr) = Parser.parseSelectExprs(exprs.toArray, ec)
    val topLevelFields = mutable.Set.empty[String]
    val finalNames = maybePaths.map(_.toArray).zip(isNamedExpr).map { case (l, named) =>
      if (named) {
        assert(l.length == 1)
        l.head
      } else {
        assert(l.head == Annotation.SAMPLE_HEAD)
        val name = l.last
        if (l.length == 2) {
          topLevelFields += name
        }
        name
      }
    }
    assert(finalNames.areDistinct())
    val newColType = TStruct(finalNames.zip(types): _*)
    val finalNameSet = finalNames.toSet
    val newColKey = colKey.filter(finalNameSet.contains)

    val newMatrixType = matrixType.copy(colType = newColType, colKey = newColKey)
    val aggOption = Aggregators.buildSampleAggregations(hc, value, ec)

    ec.set(0, globals)
    val newColValues = Array.tabulate(numCols) { i =>
      ec.set(1, colValues(i))
      aggOption.foreach(_ (i))
      val results = f()
      Row.fromSeq(results)
    }
    copyMT(matrixType = newMatrixType, colValues = newColValues)
  }

  def selectRows(selectExprs: java.util.ArrayList[String]): MatrixTable = selectRows(selectExprs.asScala.toArray: _*)

  def selectRows(exprs: String*): MatrixTable = {
    val ec = variantEC
    val (maybePaths, types, f, isNamedExpr) = Parser.parseSelectExprs(exprs.toArray, ec)
    val topLevelFields = mutable.Set.empty[String]
    val finalNames = maybePaths.map(_.toArray).zip(isNamedExpr).map { case (l, named) =>
      if (named) {
        assert(l.length == 1)
        l.head
      } else {
        assert(l.head == Annotation.VARIANT_HEAD)
        val name = l.last
        if (l.length == 2) {
          topLevelFields += name
        }
        name
      }
    }

    assert(finalNames.areDistinct())
    val touchesKeys = rowKey.forall(topLevelFields.contains)

    val finalNameSet = finalNames.toSet
    val newRowType = TStruct(finalNames.zip(types): _*)
    val newRowKey = rowKey.filter(finalNameSet.contains)
    val newPartitionKey = rowPartitionKey.filter(finalNameSet.contains)

    val newMatrixType = matrixType.copyParts(rowType = newRowType, rowKey = newRowKey, rowPartitionKey = newPartitionKey)
    val fullRowType = rvRowType
    val localEntriesIndex = entriesIndex
    val newRVType = newMatrixType.rvRowType

    val aggregateOption = Aggregators.buildVariantAggregations(this, ec)
    ec.set(0, globals)
    val mapPartitionsF: Iterator[RegionValue] => Iterator[RegionValue] = { it =>
      val fullRow = new UnsafeRow(fullRowType)
      val row = fullRow.deleteField(localEntriesIndex)
      val rv2 = RegionValue()
      val rvb = new RegionValueBuilder()
      it.map { rv =>
        fullRow.set(rv)
        ec.set(1, row)
        aggregateOption.foreach(_ (rv))
        val results = f()
        assert(results.length == types.length)

        rvb.set(rv.region)
        rvb.start(newRVType)
        rvb.startStruct()
        var i = 0
        while (i < types.length) {
          rvb.addAnnotation(types(i), results(i))
          i += 1
        }
        rvb.addField(fullRowType, rv, localEntriesIndex)
        rvb.endStruct()
        rv2.set(rv.region, rvb.end())
        rv2
      }
    }
    if (touchesKeys) {
      warn("modified row key, rescanning to compute ordering...")
      val newRDD = rvd.mapPartitions(mapPartitionsF)
      copyMT(matrixType = newMatrixType,
        rvd = OrderedRVD(newMatrixType.orvdType, newRDD, None, None))
    } else copyMT(matrixType = newMatrixType,
      rvd = rvd.mapPartitionsPreservesPartitioning(newMatrixType.orvdType)(mapPartitionsF))
  }

  def dropRows(fields: java.util.ArrayList[String]): MatrixTable = dropRows(fields.asScala.toArray: _*)

  def dropRows(fields: String*): MatrixTable = {
    if (fields.isEmpty)
      return this
    assert(fields.areDistinct())
    val dropSet = fields.toSet
    val allRowFields = rowType.fieldNames.toSet
    assert(fields.forall(allRowFields.contains))
    assert(!dropSet.contains(MatrixType.entriesIdentifier))

    // treat the entries like any other field
    val keepIndices = rvRowType.fields
      .filter(f => !dropSet.contains(f.name))
      .map(f => f.index)
      .toArray
    val newRVType = TStruct(keepIndices.map(rvRowType.fields(_)).map(f => f.name -> f.typ): _*)

    val fullRowType = rvRowType
    val localEntriesIndex = entriesIndex
    val newRowKey = rowKey.filter(f => !dropSet.contains(f))
    val newPartitionKey = rowPartitionKey.filter(f => !dropSet.contains(f))
    val touchesKeys = newRowKey != rowKey
    val newMatrixType = matrixType.copy(rvRowType = newRVType,
      rowKey = newRowKey,
      rowPartitionKey = newPartitionKey)

    // FIXME: replace with physical type
    val mapPartitionsF: Iterator[RegionValue] => Iterator[RegionValue] = { it =>
      val rv2 = RegionValue()
      val rvb = new RegionValueBuilder()
      it.map { rv =>
        rvb.set(rv.region)
        rvb.start(newRVType)
        rvb.startStruct()
        var i = 0
        while (i < keepIndices.length) {
          rvb.addField(fullRowType, rv, keepIndices(i))
          i += 1
        }
        rvb.endStruct()
        rv2.set(rv.region, rvb.end())
        rv2
      }
    }
    if (touchesKeys) {
      warn("modified row key, rescanning to compute ordering...")
      val newRDD = rvd.mapPartitions(mapPartitionsF)
      copyMT(matrixType = newMatrixType,
        rvd = OrderedRVD(newMatrixType.orvdType, newRDD, None, None))
    } else copyMT(matrixType = newMatrixType,
      rvd = rvd.mapPartitionsPreservesPartitioning(newMatrixType.orvdType)(mapPartitionsF))
  }

  def selectEntries(selectExprs: java.util.ArrayList[String]): MatrixTable = selectEntries(selectExprs.asScala.toArray: _*)

  def selectEntries(exprs: String*): MatrixTable = {
    val ec = entryEC
    ec.set(0, globals)

    val (maybePaths, types, f, isNamedExpr) = Parser.parseSelectExprs(exprs.toArray, ec)
    val topLevelFields = mutable.Set.empty[String]
    val finalNames = maybePaths.map(_.toArray).zip(isNamedExpr).map { case (l, named) =>
      if (named) {
        assert(l.length == 1)
        l.head
      } else {
        assert(l.head == Annotation.GENOTYPE_HEAD)
        val name = l.last
        if (l.length == 2) {
          topLevelFields += name
        }
        name
      }
    }
    assert(finalNames.areDistinct())

    val newEntryType = TStruct(finalNames.zip(types): _*)
    val fullRowType = rvRowType
    val localEntriesIndex = entriesIndex
    val localNCols = numCols
    val localColValuesBc = colValuesBc

    insertEntries(() => {
      val fullRow = new UnsafeRow(fullRowType)
      val row = fullRow.deleteField(localEntriesIndex)
      ec.set(1, row)
      fullRow -> row
    })(newEntryType, { case ((fullRow, row), rv, rvb) =>
      fullRow.set(rv)
      val entries = fullRow.getAs[IndexedSeq[Annotation]](localEntriesIndex)
      rvb.startArray(localNCols)
      var i = 0
      while (i < localNCols) {
        val entry = entries(i)
        ec.set(2, localColValuesBc)
        ec.set(3, entry)
        val results = f()
        var j = 0
        rvb.startStruct()
        while (j < types.length) {
          rvb.addAnnotation(types(j), results(j))
          j += 1
        }
        rvb.endStruct()
        i += 1
      }
      rvb.endArray()
    })
  }


  def dropEntries(fields: java.util.ArrayList[String]): MatrixTable = dropEntries(fields.asScala.toArray: _*)

  def dropEntries(fields: String*): MatrixTable = {
    if (fields.isEmpty)
      return this
    assert(fields.areDistinct())
    val dropSet = fields.toSet
    val allEntryFields = entryType.fieldNames.toSet
    assert(fields.forall(allEntryFields.contains))

    val keepIndices = entryType.fields
      .filter(f => !dropSet.contains(f.name))
      .map(f => f.index)
      .toArray
    val newEntryType = TStruct(keepIndices.map(entryType.fields(_)).map(f => f.name -> f.typ): _*)

    val fullRowType = rvRowType
    val localEntriesIndex = entriesIndex
    val localEntriesType = matrixType.entryArrayType
    val localEntryType = entryType
    val localNCols = numCols
    // FIXME: replace with physical type
    insertEntries(noOp)(newEntryType, { case (_, rv, rvb) =>
      val entriesOffset = fullRowType.loadField(rv, localEntriesIndex)
      rvb.startArray(localNCols)
      var i = 0
      while (i < localNCols) {
        if (localEntriesType.isElementMissing(rv.region, entriesOffset, i))
          rvb.setMissing()
        else {
          val eltOffset = localEntriesType.loadElement(rv.region, entriesOffset, localNCols, i)
          rvb.startStruct()
          var j = 0
          while (j < keepIndices.length) {
            rvb.addField(localEntryType, rv.region, eltOffset, j)
            j += 1
          }
          rvb.endStruct()
        }
        i += 1
      }
      rvb.endArray()
    })
  }

  def nPartitions: Int = rvd.partitions.length

  def annotateVariantsVDS(right: MatrixTable, root: String): MatrixTable = {
    // need to strip entries!
    // FIXME: HACK
    val rTyp = new OrderedRVDType(right.rowPartitionKey.toArray, right.rowKey.toArray, right.rowType)
    val rightRVD = OrderedRVD(rTyp, right.rvd.partitioner, right.rowsTable().rvd)
    orderedRVDLeftJoinDistinctAndInsert(rightRVD, root, product = false)
  }

  def count(): (Long, Long) = (numCols, countVariants())

  def countVariants(): Long = partitionCounts().sum

  def forceCountRows(): Long = rvd.count()

  def deduplicate(): MatrixTable =
    copy2(rvd = rvd.mapPartitionsPreservesPartitioning(rvd.typ)(
      SortedDistinctRowIterator.transformer(rvd.typ)))

  def deleteVA(args: String*): (Type, Deleter) = deleteVA(args.toList)

  def deleteVA(path: List[String]): (Type, Deleter) = rowType.delete(path)

  def dropSamples(): MatrixTable =
    copyAST(ast = FilterSamples(ast, Const(null, false, TBoolean())))

  def dropVariants(): MatrixTable = copy2(rvd = OrderedRVD.empty(sparkContext, matrixType.orvdType))

  def explodeVariants(root: String): MatrixTable = {
    val path = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)
    val (keysType, querier) = rvRowType.queryTyped(path)
    val keyType = keysType match {
      case TArray(e, _) => e
      case TSet(e, _) => e
      case t => fatal(s"Expected annotation of type Array or Set; found $t")
    }

    val (newRVType, inserter) = rvRowType.unsafeStructInsert(keyType, path)
    val newMatrixType = matrixType.copy(rvRowType = newRVType)
    val oldRVType = rvRowType

    val localEntriesIndex = entriesIndex

    val explodedRDD = rvd.mapPartitionsPreservesPartitioning(newMatrixType.orvdType) { it =>
      val region2 = Region()
      val rv2 = RegionValue(region2)
      val rv2b = new RegionValueBuilder(region2)
      val ur = new UnsafeRow(oldRVType)
      it.flatMap { rv =>
        ur.set(rv)
        val keys = querier(ur).asInstanceOf[Iterable[Any]]
        if (keys == null)
          None
        else
          keys.iterator.map { explodedElement =>
            region2.clear()
            rv2b.start(newRVType)
            inserter(rv.region, rv.offset, rv2b,
              () => rv2b.addAnnotation(keyType, explodedElement))
            rv2.setOffset(rv2b.end())
            rv2
          }
      }
    }
    copyMT(matrixType = newMatrixType, rvd = explodedRDD)
  }

  def explodeSamples(code: String): MatrixTable = {
    val path = Parser.parseAnnotationRoot(code, Annotation.SAMPLE_HEAD)
    val (keysType, querier) = colType.queryTyped(path)
    val keyType = keysType match {
      case TArray(e, _) => e
      case TSet(e, _) => e
      case t => fatal(s"Expected annotation of type Array or Set; found $t")
    }
    var size = 0
    val keys = colValues.map { sa =>
      val ks = querier(sa).asInstanceOf[Iterable[Any]]
      if (ks == null)
        Iterable.empty[Any]
      else {
        size += ks.size
        ks
      }
    }

    val (newColType, inserter) = colType.structInsert(keyType, path)

    val sampleMap = new Array[Int](size)
    val newColValues = new Array[Annotation](size)
    val newNCols = newColValues.length

    var i = 0
    var j = 0
    while (i < numCols) {
      keys(i).foreach { e =>
        sampleMap(j) = i
        newColValues(j) = inserter(colValues(i), e)
        j += 1
      }
      i += 1
    }

    val sampleMapBc = sparkContext.broadcast(sampleMap)
    val localEntriesIndex = entriesIndex
    val localEntriesType = matrixType.entryArrayType
    val fullRowType = rvRowType

    insertEntries(noOp, newColType = newColType,
      newColValues = newColValues)(entryType, { case (_, rv, rvb) =>

      val entriesOffset = fullRowType.loadField(rv, localEntriesIndex)
      rvb.startArray(newNCols)
      var i = 0
      while (i < newNCols) {
        rvb.addElement(localEntriesType, rv.region, entriesOffset, sampleMapBc.value(i))
        i += 1
      }

      rvb.endArray()

    })
  }

  def annotateGenotypesExpr(expr: String): MatrixTable = {
    val symTab = Map(
      "va" -> (0, rowType),
      "sa" -> (1, colType),
      "g" -> (2, entryType),
      "global" -> (3, globalType))
    val ec = EvalContext(symTab)

    ec.set(3, globals)

    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Some(Annotation.GENOTYPE_HEAD))

    val inserterBuilder = new ArrayBuilder[Inserter]()
    val newEntryType = (paths, types).zipped.foldLeft(entryType) { case (gsig, (ids, signature)) =>
      val (s, i) = gsig.structInsert(signature, ids)
      inserterBuilder += i
      s
    }
    val inserters = inserterBuilder.result()

    val localNSamples = numCols
    val fullRowType = rvRowType
    val localColValuesBc = colValuesBc
    val localEntriesIndex = entriesIndex

    insertEntries(() => {
      val fullRow = new UnsafeRow(fullRowType)
      val row = fullRow.deleteField(localEntriesIndex)
      (fullRow, row)
    })(newEntryType, { case ((fullRow, row), rv, rvb) =>
      fullRow.set(rv)
      val entries = fullRow.getAs[IndexedSeq[Annotation]](localEntriesIndex)

      rvb.startArray(localNSamples)

      var i = 0
      while (i < localNSamples) {
        val entry = entries(i)
        ec.setAll(row,
          localColValuesBc.value(i),
          entry)

        val newEntry = f().zip(inserters)
          .foldLeft(entry) { case (ga, (a, inserter)) =>
            inserter(ga, a)
          }
        rvb.addAnnotation(newEntryType, newEntry)

        i += 1
      }
      rvb.endArray()
    })
  }

  def filterSamples(p: (Annotation, Int) => Boolean): MatrixTable = {
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

  def sparkContext: SparkContext = hc.sc

  def hadoopConf: hadoop.conf.Configuration = hc.hadoopConf

  def head(n: Long): MatrixTable = {
    if (n < 0)
      fatal(s"n must be non-negative! Found `$n'.")
    copy2(rvd = rvd.head(n))
  }

  def insertSA(sig: Type, args: String*): (TStruct, Inserter) = insertSA(sig, args.toList)

  def insertSA(sig: Type, path: List[String]): (TStruct, Inserter) = colType.structInsert(sig, path)

  def insertEntries[PC](makePartitionContext: () => PC, newColType: TStruct = colType,
    newColKey: IndexedSeq[String] = colKey,
    newColValues: IndexedSeq[Annotation] = colValues,
    newGlobalType: TStruct = globalType,
    newGlobals: Annotation = globals)(newEntryType: TStruct,
    inserter: (PC, RegionValue, RegionValueBuilder) => Unit): MatrixTable = {
    insertIntoRow(makePartitionContext, newColType, newColKey, newColValues, newGlobalType, newGlobals)(
      TArray(newEntryType), MatrixType.entriesIdentifier, inserter)
  }

  def insertIntoRow[PC](makePartitionContext: () => PC, newColType: TStruct = colType,
    newColKey: IndexedSeq[String] = colKey,
    newColValues: IndexedSeq[Annotation] = colValues,
    newGlobalType: TStruct = globalType,
    newGlobals: Annotation = globals)(typeToInsert: Type, path: String,
    inserter: (PC, RegionValue, RegionValueBuilder) => Unit): MatrixTable = {
    assert(!rowKey.contains(path))


    val fullRowType = rvRowType
    val localEntriesIndex = entriesIndex

    val (newRVType, ins) = fullRowType.unsafeStructInsert(typeToInsert, List(path))

    val newMatrixType = matrixType.copy(rvRowType = newRVType, colType = newColType,
      colKey = newColKey, globalType = newGlobalType)

    copyMT(matrixType = newMatrixType,
      globals = newGlobals,
      colValues = newColValues,
      rvd = rvd.mapPartitionsPreservesPartitioning(newMatrixType.orvdType) { it =>

        val pc = makePartitionContext()

        val rv2 = RegionValue()
        val rvb = new RegionValueBuilder()
        it.map { rv =>
          rvb.set(rv.region)
          rvb.start(newRVType)

          ins(rv.region, rv.offset, rvb,
            () => inserter(pc, rv, rvb)
          )

          rv2.set(rv.region, rvb.end())
          rv2
        }
      })
  }

  /**
    *
    * @param right right-hand dataset with which to join
    */
  def unionCols(right: MatrixTable): MatrixTable = {
    if (entryType != right.entryType) {
      fatal(
        s"""union_cols: cannot combine datasets with different entry schema
           |  left entry schema: @1
           |  right entry schema: @2""".stripMargin,
        entryType.toString,
        right.entryType.toString)
    }

    if (!colKeyTypes.sameElements(right.colKeyTypes)) {
      fatal(
        s"""union_cols: cannot combine datasets with different column key schema
           |  left column schema: [${ colKeyTypes.map(_.toString).mkString(", ") }]
           |  right column schema: [${ right.colKeyTypes.map(_.toString).mkString(", ") }]""".stripMargin)
    }

    if (colType != right.colType) {
      fatal(
        s"""union_cols: cannot combine datasets with different column schema
           |  left column schema: @1
           |  right column schema: @2""".stripMargin,
        colType.toString,
        right.colType.toString)
    }

    if (!rowKeyTypes.sameElements(right.rowKeyTypes)) {
      fatal(
        s"""union_cols: cannot combine datasets with different row key schema
           |  left row key schema: @1
           |  right row key schema: @2""".stripMargin,
        rowKeyTypes.map(_.toString).mkString(", "),
        right.rowKeyTypes.map(_.toString).mkString(", "))
    }

    val leftRVType = rvRowType
    val localLeftSamples = numCols
    val localRightSamples = right.numCols
    val rightRVRowType = right.rvRowType
    val leftEntriesIndex = entriesIndex
    val rightEntriesIndex = right.entriesIndex
    val localEntriesType = matrixType.entryArrayType
    assert(right.matrixType.entryArrayType == localEntriesType)

    val joined = rvd.orderedJoinDistinct(right.rvd, "inner").mapPartitions({ it =>
      val rvb = new RegionValueBuilder()
      val rv2 = RegionValue()

      it.map { jrv =>
        val lrv = jrv.rvLeft
        val rrv = jrv.rvRight

        rvb.set(lrv.region)
        rvb.start(leftRVType)
        rvb.startStruct()
        var i = 0
        while (i < leftRVType.size) {
          if (i != leftEntriesIndex)
            rvb.addField(leftRVType, lrv, i)
          i += 1
        }
        rvb.startArray(localLeftSamples + localRightSamples)

        val leftEntriesOffset = leftRVType.loadField(lrv.region, lrv.offset, leftEntriesIndex)
        val leftEntriesLength = localEntriesType.loadLength(lrv.region, leftEntriesOffset)
        assert(leftEntriesLength == localLeftSamples)

        val rightEntriesOffset = rightRVRowType.loadField(rrv.region, rrv.offset, rightEntriesIndex)
        val rightEntriesLength = localEntriesType.loadLength(rrv.region, rightEntriesOffset)
        assert(rightEntriesLength == localRightSamples)

        i = 0
        while (i < localLeftSamples) {
          rvb.addElement(localEntriesType, lrv.region, leftEntriesOffset, i)
          i += 1
        }

        i = 0
        while (i < localRightSamples) {
          rvb.addElement(localEntriesType, rrv.region, rightEntriesOffset, i)
          i += 1
        }

        rvb.endArray()
        rvb.endStruct()
        rv2.set(lrv.region, rvb.end())
        rv2
      }
    }, preservesPartitioning = true)

    val newMatrixType = matrixType.copyParts() // move entries to the end

    copyMT(matrixType = newMatrixType,
      colValues = colValues ++ right.colValues,
      rvd = OrderedRVD(rvd.typ, rvd.partitioner, joined))
  }

  def makeKT(variantCondition: String, genotypeCondition: String, keyNames: Array[String] = Array.empty, seperator: String = "."): Table = {
    requireColKeyString("make table")

    val vSymTab = Map(
      "global" -> (0, globalType),
      "va" -> (1, rowType))
    val vEC = EvalContext(vSymTab)
    val vA = vEC.a

    val (vNames, vTypes, vf) = Parser.parseNamedExprs(variantCondition, vEC)

    val gSymTab = Map(
      "global" -> (0, globalType),
      "va" -> (1, rowType),
      "sa" -> (2, colType),
      "g" -> (3, entryType))
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

    val localNSamples = numCols
    val localColValuesBc = colValuesBc
    val localRVRowType = rvRowType
    val localGlobals = globals
    val localEntriesIndex = entriesIndex

    val n = vNames.length + gNames.length * localNSamples
    Table(hc,
      rvd.mapPartitions { it =>
        val fullRow = new UnsafeRow(localRVRowType)
        val row = fullRow.deleteField(localEntriesIndex)

        it.map { rv =>
          fullRow.set(rv)
          val gs = fullRow.getAs[IndexedSeq[Annotation]](localEntriesIndex)

          val a = new Array[Any](n)

          var j = 0
          vEC.setAll(localGlobals, row)
          vf().foreach { x =>
            a(j) = x
            j += 1
          }

          var i = 0
          while (i < localNSamples) {
            val sa = localColValuesBc.value(i)
            gEC.setAll(localGlobals, row, sa, gs(i))
            gf().foreach { x =>
              a(j) = x
              j += 1
            }

            i += 1
          }

          assert(j == n)
          Row.fromSeq(a)
        }
      },
      sig,
      keyNames)
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
      "global" -> (0, globalType),
      "g" -> (1, entryType),
      "va" -> (2, rowType),
      "sa" -> (3, colType))
    val ec = EvalContext(Map(
      "global" -> (0, globalType),
      "gs" -> (1, TAggregable(entryType, aggregationST))))

    val ts = exprs.map(e => Parser.parseExpr(e, ec))

    val localGlobals = globals
    val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[Annotation](ec, { case (ec, g) =>
      ec.set(1, g)
    })

    val globalBc = sparkContext.broadcast(globals)
    val localColValuesBc = colValuesBc
    val localRVRowType = rvRowType
    val localRowType = rowType
    val localEntriesIndex = entriesIndex

    val result = rvd.mapPartitions { it =>
      val fullRow = new UnsafeRow(localRVRowType)
      val row = new UnsafeRow(localRowType)

      val zv = zVal.map(_.copy())
      ec.set(0, globalBc.value)
      it.foreach { rv =>
        fullRow.set(rv)
        row.set(rv)
        val gs = fullRow.getAs[IndexedSeq[Any]](localEntriesIndex)

        var i = 0
        ec.set(2, row)
        gs.foreach { g =>
          ec.set(4, localColValuesBc.value(i))
          seqOp(zv, g)
          i += 1
        }
      }
      Iterator(zv)
    }.fold(zVal.map(_.copy()))(combOp)
    resOp(result)

    ec.set(0, localGlobals)
    ts.map { case (t, f) => (f(), t) }
  }

  def queryGlobal(path: String): (Type, Annotation) = {
    val st = Map(Annotation.GLOBAL_HEAD -> (0, globalType))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parseExpr(path, ec)

    val f2: Annotation => Any = { annotation =>
      a(0) = annotation
      f()
    }

    (t, f2(globals))
  }

  def querySamples(expr: String): (Annotation, Type) = {
    val qs = querySamples(Array(expr))
    assert(qs.length == 1)
    qs.head
  }

  def querySamples(exprs: Array[String]): Array[(Annotation, Type)] = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "sa" -> (1, colType))
    val ec = EvalContext(Map(
      "global" -> (0, globalType),
      "samples" -> (1, TAggregable(colType, aggregationST))))

    val ts = exprs.map(e => Parser.parseExpr(e, ec))

    val localGlobals = globals
    val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[Annotation](ec, { case (ec, (sa)) =>
      ec.setAll(localGlobals, sa)
    })

    val results = colValues
      .aggregate(zVal)(seqOp, combOp)
    resOp(results)
    ec.set(0, localGlobals)

    ts.map { case (t, f) => (f(), t) }
  }

  def queryVA(code: String): (Type, Querier) = {

    val st = Map(Annotation.VARIANT_HEAD -> (0, rowType))
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
      "global" -> (0, globalType),
      "va" -> (1, rowType))
    val ec = EvalContext(Map(
      "global" -> (0, globalType),
      "variants" -> (1, TAggregable(rowType, aggregationST))))
    ec.setAll(globals)

    val ts = exprs.map(e => Parser.parseExpr(e, ec))

    val fullRowType = rvRowType
    val localEntriesIndex = entriesIndex
    val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[RegionValue](ec, { case (ec, rv) =>
      val ur = new UnsafeRow(fullRowType, rv)
      val row = ur.deleteField(localEntriesIndex)
      ec.set(1, row)
    })

    val result = rvd
      .treeAggregate(zVal)(seqOp, combOp, depth = treeAggDepth(hc, nPartitions))
    resOp(result)

    ts.map { case (t, f) =>
      (f(), t)
    }
  }

  def reorderSamples(newIds: java.util.ArrayList[String]): MatrixTable =
    reorderSamples(newIds.asScala.toArray.map(Annotation(_)))

  def reorderSamples(newIds: IndexedSeq[Annotation]): MatrixTable = {
    require(newIds.length == numCols)
    requireUniqueSamples("reorder_samples")
    require(newIds.areDistinct())

    val sampleSet = colKeys.toSet[Annotation]
    val newSampleSet = newIds.toSet

    val missingSamples = sampleSet -- newSampleSet
    if (missingSamples.nonEmpty)
      fatal(s"Found ${ missingSamples.size } ${ plural(missingSamples.size, "sample ID") } in dataset that are not in new ordering:\n  " +
        s"@1", missingSamples.truncatable("\n  "))

    val notInDataset = newSampleSet -- sampleSet
    if (notInDataset.nonEmpty)
      fatal(s"Found ${ notInDataset.size } ${ plural(notInDataset.size, "sample ID") } in new ordering that are not in dataset:\n  " +
        s"@1", notInDataset.truncatable("\n  "))

    val oldIndex = colKeys.zipWithIndex.toMap
    val newToOld = newIds.map(oldIndex)

    val newColValues = Array.tabulate(numCols) { i =>
      colValues(newToOld(i))
    }

    val localNumCols = numCols
    val localRVRowType = rvRowType
    val localEntriesIndex = entriesIndex
    val localEntriesType = matrixType.entryArrayType

    insertEntries(noOp, colType, colKey, newColValues.toFastIndexedSeq)(entryType, { case (_, rv, rvb) =>
      val entriesOffset = localRVRowType.loadField(rv, localEntriesIndex)
      rvb.startArray(localNumCols)
      var i = 0
      while (i < localNumCols) {
        rvb.addElement(localEntriesType, rv.region, entriesOffset, newToOld(i))
        i += 1
      }
      rvb.endArray()
    })
  }

  def renameDuplicates(id: String): MatrixTable = {
    requireColKeyString("rename duplicates")
    val (newIds, duplicates) = mangle(stringSampleIds.toArray)
    if (duplicates.nonEmpty)
      info(s"Renamed ${ duplicates.length } duplicate ${ plural(duplicates.length, "sample ID") }. " +
        s"Mangled IDs as follows:\n  @1", duplicates.map { case (pre, post) => s""""$pre" => "$post"""" }.truncatable("\n  "))
    else
      info(s"No duplicate sample IDs found.")
    val (newSchema, ins) = insertSA(TString(), id)
    val newAnnotations = colValues.zipWithIndex.map { case (sa, i) => ins(sa, newIds(i)) }.toArray
    copy2(colType = newSchema, colValues = newAnnotations)
  }

  def same(that: MatrixTable, tolerance: Double = utils.defaultTolerance): Boolean = {
    var metadataSame = true
    if (rowType != that.rowType) {
      metadataSame = false
      println(
        s"""different va signature:
           |  left:  ${ rowType.toString }
           |  right: ${ that.rowType.toString }""".stripMargin)
    }
    if (colType != that.colType) {
      metadataSame = false
      println(
        s"""different sa signature:
           |  left:  ${ colType.toString }
           |  right: ${ that.colType.toString }""".stripMargin)
    }
    if (globalType != that.globalType) {
      metadataSame = false
      println(
        s"""different global signature:
           |  left:  ${ globalType.toString }
           |  right: ${ that.globalType.toString }""".stripMargin)
    }
    if (!colValuesSimilar(that, tolerance)) {
      metadataSame = false
      println(
        s"""different sample annotations:
           |  left:  $colValues
           |  right: ${ that.colValues }""".stripMargin)
    }
    if (!globalType.valuesSimilar(globals, that.globals)) {
      metadataSame = false
      println(
        s"""different global annotation:
           |  left:  $globals
           |  right: ${ that.globals }""".stripMargin)
    }
    if (rowKey != that.rowKey || colKey != that.colKey || rowPartitionKey != that.rowPartitionKey) {
      metadataSame = false
      println(
        s"""
           |different keys:
           |  left:  rk $rowKey, rpk $rowPartitionKey, ck $colKey
           |  right: rk ${ that.rowKey }, rpk ${ that.rowPartitionKey }, ck ${ that.colKey }""".stripMargin)
    }
    if (!metadataSame)
      println("metadata were not the same")

    val leftRVType = rvRowType
    val rightRVType = that.rvRowType
    val localRowType = rowType
    val localLeftEntriesIndex = entriesIndex
    val localRightEntriesIndex = that.entriesIndex
    val localEntryType = entryType
    val localRKF = rowKeysF
    val localColKeys = colKeys

    metadataSame &&
      rvd.rdd.zipPartitions(
        OrderedRVD.shuffle(
          that.rvd.typ,
          rvd.partitioner.withKType(that.rvd.typ.partitionKey, that.rvd.typ.kType),
          that.rvd.rdd)
          .rdd) { (it1, it2) =>
        val fullRow1 = new UnsafeRow(leftRVType)
        val fullRow2 = new UnsafeRow(rightRVType)
        var partSame = true
        while (it1.hasNext && it2.hasNext) {
          val rv1 = it1.next()
          val rv2 = it2.next()

          fullRow1.set(rv1)
          fullRow2.set(rv2)
          val row1 = fullRow1.deleteField(localLeftEntriesIndex)
          val row2 = fullRow2.deleteField(localRightEntriesIndex)

          if (!localRowType.valuesSimilar(row1, row2, tolerance)) {
            println(
              s"""row fields not the same:
                 |  $row1
                 |  $row2""".stripMargin)
            partSame = false
          }

          val gs1 = fullRow1.getAs[IndexedSeq[Annotation]](localLeftEntriesIndex)
          val gs2 = fullRow2.getAs[IndexedSeq[Annotation]](localRightEntriesIndex)

          var i = 0
          while (partSame && i < gs1.length) {
            if (!localEntryType.valuesSimilar(gs1(i), gs2(i), tolerance)) {
              partSame = false
              println(
                s"""different entry at row ${ localRKF(row1) }, col ${ localColKeys(i) }
                   |  ${ gs1(i) }
                   |  ${ gs2(i) }""".stripMargin)
            }
            i += 1
          }
        }

        if ((it1.hasNext || it2.hasNext) && partSame) {
          println("partition has different number of rows")
          partSame = false
        }

        Iterator(partSame)
      }.forall(t => t)
  }

  def sampleEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "sa" -> (1, colType),
      "g" -> (2, entryType),
      "va" -> (3, rowType))
    EvalContext(Map(
      "global" -> (0, globalType),
      "sa" -> (1, colType),
      "gs" -> (2, TAggregable(entryType, aggregationST))))
  }

  def colValuesSimilar(that: MatrixTable, tolerance: Double = utils.defaultTolerance): Boolean = {
    require(colType == that.colType, s"\n${ colType }\n${ that.colType }")
    colValues.zip(that.colValues)
      .forall { case (s1, s2) => colType.valuesSimilar(s1, s2, tolerance)
      }
  }

  def sampleVariants(p: Double, seed: Int = 1): MatrixTable = {
    require(p > 0 && p < 1, s"the 'p' parameter must fall between 0 and 1, found $p")
    copyMT(rvd = rvd.sample(withReplacement = false, p, seed))
  }

  def copy2(rvd: OrderedRVD = rvd,
    colValues: IndexedSeq[Annotation] = colValues,
    colKey: IndexedSeq[String] = colKey,
    globals: Annotation = globals,
    colType: TStruct = colType,
    rvRowType: TStruct = rvRowType,
    rowPartitionKey: IndexedSeq[String] = rowPartitionKey,
    rowKey: IndexedSeq[String] = rowKey,
    globalType: TStruct = globalType,
    entryType: TStruct = entryType): MatrixTable = {
    val newMatrixType = matrixType.copy(
      globalType = globalType,
      colKey = colKey,
      colType = colType,
      rowPartitionKey = rowPartitionKey,
      rowKey = rowKey,
      rvRowType = rvRowType)
    new MatrixTable(hc,
      newMatrixType,
      globals, colValues, rvd)
  }

  def copyMT(rvd: OrderedRVD = rvd,
    matrixType: MatrixType = matrixType,
    globals: Annotation = globals,
    colValues: IndexedSeq[Annotation] = colValues): MatrixTable = {
    assert(rvd.typ == matrixType.orvdType,
      s"mismatch in orvdType:\n  rdd: ${ rvd.typ }\n  mat: ${ matrixType.orvdType }")
    new MatrixTable(hc,
      matrixType, globals, colValues, rvd)
  }

  def copyAST(ast: MatrixIR = ast): MatrixTable =
    new MatrixTable(hc, ast)

  def colsTable(): Table = {
    Table(hc, sparkContext.parallelize(colValues.map(_.asInstanceOf[Row])),
      colType,
      colKey,
      globalType,
      globals.asInstanceOf[Row])
  }

  def storageLevel: String = rvd.storageLevel.toReadableString()

  def summarize(): SummaryResult = {
    val localRVRowType = rvRowType
    val localNSamples = numCols
    rvd.aggregateWithContext(() =>
      (HardCallView(localRVRowType),
        new RegionValueVariant(localRVRowType))
    )(new SummaryCombiner)(
      { case ((view, rvVariant), summary, rv) =>
        rvVariant.setRegion(rv)
        view.setRegion(rv)
        summary.merge(view, rvVariant)
      }, _.merge(_))
      .result(localNSamples)
  }

  def numCols: Int = colValues.length

  def typecheck() {
    var foundError = false
    if (!globalType.typeCheck(globals)) {
      foundError = true
      warn(
        s"""found violation in global annotation
           |Schema: $globalType
           |Annotation: ${ Annotation.printAnnotation(globals) }""".stripMargin)
    }

    colValues.zipWithIndex.find { case (sa, i) => !colType.typeCheck(sa) }
      .foreach { case (sa, i) =>
        foundError = true
        warn(
          s"""found violation in sample annotations for col $i
             |Schema: $colType
             |Annotation: ${ Annotation.printAnnotation(sa) }""".stripMargin)
      }

    val localRVRowType = rvRowType
    rvd.map { rv =>
      new UnsafeRow(localRVRowType, rv)
    }.find(ur => !localRVRowType.typeCheck(ur))
      .foreach { ur =>

        foundError = true
        warn(
          s"""found violation in row
             |Schema: $localRVRowType
             |Annotation: ${ Annotation.printAnnotation(ur) }""".stripMargin)
      }

    if (foundError)
      fatal("found one or more type check errors")
  }

  def entryEC: EvalContext = EvalContext(Map(
    "global" -> (0, globalType),
    "va" -> (1, rowType),
    "sa" -> (2, colType),
    "g" -> (3, entryType)))

  def variantEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "va" -> (1, rowType),
      "g" -> (2, entryType),
      "sa" -> (3, colType))
    EvalContext(Map(
      "global" -> (0, globalType),
      "va" -> (1, rowType),
      "gs" -> (2, TAggregable(entryType, aggregationST))))
  }

  def globalsTable(): Table = {
    Table(hc,
      sparkContext.parallelize[Row](Array(globals.asInstanceOf[Row])),
      globalType,
      Array.empty[String])
  }

  def rowsTable(): Table = {
    val localRowType = rowType
    val fullRowType = rvRowType
    val localEntriesIndex = entriesIndex
    new Table(hc, rvd.mapPartitions { it =>
      val rv2b = new RegionValueBuilder()
      val rv2 = RegionValue()
      it.map { rv =>
        rv2b.set(rv.region)
        rv2b.start(localRowType)
        rv2b.startStruct()
        var i = 0
        while (i < fullRowType.size) {
          if (i != localEntriesIndex)
            rv2b.addField(fullRowType, rv, i)
          i += 1
        }
        rv2b.endStruct()
        rv2.set(rv.region, rv2b.end())
        rv2
      }
    },
      localRowType,
      rowKey,
      globalType,
      globals.asInstanceOf[Row])
  }

  def entriesTable(): Table = {
    val localNSamples = numCols

    val allFields = rowType.fields.map(f => f.name -> f.typ) ++
      colType.fields.map(f => f.name -> f.typ) ++
      entryType.fields.map(f => f.name -> f.typ)

    val resultStruct = TStruct(allFields: _*)

    val localColType = colType
    val localEntryType = entryType
    val fullRowType = rvRowType

    val localEntriesType = matrixType.entryArrayType
    val localEntriesIndex = entriesIndex
    val saArrayType = TArray(colType, required = true)

    val rowSize = rowType.size

    val localColValuesBc = colValuesBc
    new Table(hc, rvd.mapPartitions { it =>

      val colValues = localColValuesBc.value

      val rv2b = new RegionValueBuilder()
      val rv2 = RegionValue()
      it.flatMap { rv =>
        val rvEnd = rv.region.size
        rv2b.set(rv.region)
        val gsOffset = fullRowType.loadField(rv, localEntriesIndex)
        (0 until localNSamples).iterator
          .filter { i =>
            localEntriesType.isElementDefined(rv.region, gsOffset, i)
          }
          .map { i =>
            rv.region.clear(rvEnd)
            rv2b.clear()
            rv2b.start(resultStruct)
            rv2b.startStruct()

            var j = 0
            while (j < fullRowType.size) {
              if (j != localEntriesIndex)
                rv2b.addField(fullRowType, rv, j)
              j += 1
            }

            if (saArrayType.isElementDefined(colValues.region, colValues.aoff, i))
              rv2b.addAllFields(localColType, colValues.region, saArrayType.loadElement(colValues.region, colValues.aoff, i))
            else
              rv2b.skipFields(localColType.size)

            rv2b.addAllFields(localEntryType, rv.region, localEntriesType.elementOffsetInRegion(rv.region, gsOffset, i))
            rv2b.endStruct()
            rv2.set(rv.region, rv2b.end())
            rv2
          }
      }
    },
      resultStruct,
      rowKey ++ colKey,
      globalSignature = globalType,
      globals = globals.asInstanceOf[Row])
  }

  def coalesce(k: Int, shuffle: Boolean = true): MatrixTable = copy2(rvd = rvd.coalesce(k, shuffle))

  def persist(storageLevel: String = "MEMORY_AND_DISK"): MatrixTable = {
    val level = try {
      StorageLevel.fromString(storageLevel)
    } catch {
      case e: IllegalArgumentException =>
        fatal(s"unknown StorageLevel `$storageLevel'")
    }

    copy2(rvd = rvd.persist(level))
  }

  def cache(): MatrixTable = persist("MEMORY_ONLY")

  def unpersist(): MatrixTable = copy2(rvd = rvd.unpersist())

  def naiveCoalesce(maxPartitions: Int): MatrixTable =
    copy2(rvd = rvd.naiveCoalesce(maxPartitions))

  def unfilterEntries(): MatrixTable = {
    val localEntriesType = matrixType.entryArrayType
    val localEntriesIndex = entriesIndex
    val localEntryType = entryType
    val fullRowType = rvRowType
    val localNCols = numCols

    insertEntries(noOp)(localEntryType, { case (_, rv, rvb) =>
      val entriesOffset = fullRowType.loadField(rv, localEntriesIndex)

      rvb.startArray(localNCols)
      var i = 0
      while (i < localNCols) {
        if (localEntriesType.isElementMissing(rv.region, entriesOffset, i)) {
          rvb.startStruct()
          rvb.skipFields(localEntryType.size)
          rvb.endStruct()
        } else
          rvb.addElement(localEntriesType, rv.region, entriesOffset, i)
        i += 1
      }
      rvb.endArray()
    })
  }

  /**
    * @param filterExpr filter expression involving v (Variant), va (variant annotations), s (sample),
    *                   sa (sample annotations), and g (genotype annotation), which returns a boolean value
    * @param keep       keep genotypes where filterExpr evaluates to true
    */
  def filterGenotypes(filterExpr: String, keep: Boolean = true): MatrixTable = {
    val symTab = Map(
      "va" -> (0, rowType),
      "sa" -> (1, colType),
      "g" -> (2, entryType),
      "global" -> (3, globalType))

    val ec = EvalContext(symTab)
    ec.set(3, globals)
    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](filterExpr, ec)

    val localKeep = keep
    val localRVRowType = rvRowType
    val localNSamples = numCols
    val localEntryType = entryType
    val localColValuesBc = colValuesBc
    val localEntriesIndex = entriesIndex

    copy2(
      rvd = rvd.mapPartitionsPreservesPartitioning(rvd.typ) { it =>
        val rvb = new RegionValueBuilder()
        val rv2 = RegionValue()
        val fullRow = new UnsafeRow(localRVRowType)

        it.map { rv =>
          fullRow.set(rv)
          val row = fullRow.deleteField(localEntriesIndex)

          rvb.set(rv.region)
          rvb.start(localRVRowType)
          rvb.startStruct()

          var i = 0
          while (i < localEntriesIndex) {
            rvb.addField(localRVRowType, rv, i)
            i += 1
          }

          val gs = fullRow.getAs[IndexedSeq[Any]](localEntriesIndex)

          rvb.startArray(localNSamples)
          i = 0
          while (i < localNSamples) {
            val sa = localColValuesBc.value(i)
            val g = gs(i)
            ec.setAll(row, sa, g)
            if (Filter.boxedKeepThis(f(), localKeep))
              rvb.addAnnotation(localEntryType, g)
            else
              rvb.setMissing()

            i += 1
          }
          rvb.endArray()
          rvb.endStruct()
          rv2.set(rv.region, rvb.end())
          rv2
        }
      })
  }


  def writeCols(path: String, codecSpec: CodecSpec) {
    val partitionCounts = RVD.writeLocalUnpartitioned(hc, path + "/rows", matrixType.colType, codecSpec, colValues)

    val colsSpec = TableSpec(
      FileFormat.version.rep,
      hc.version,
      "../references",
      matrixType.colsTableType,
      Map("globals" -> RVDComponentSpec("../globals/rows"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    colsSpec.write(hc, path)

    hadoopConf.writeTextFile(path + "/_SUCCESS")(out => ())
  }

  def writeGlobals(path: String, codecSpec: CodecSpec) {
    val partitionCounts = RVD.writeLocalUnpartitioned(hc, path + "/rows", matrixType.globalType, codecSpec, Array(globals))

    RVD.writeLocalUnpartitioned(hc, path + "/globals", TStruct.empty(), codecSpec, Array[Annotation](Row()))

    val globalsSpec = TableSpec(
      FileFormat.version.rep,
      hc.version,
      "../references",
      TableType(globalType, Array.empty[String], TStruct.empty()),
      Map("globals" -> RVDComponentSpec("globals"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    globalsSpec.write(hc, path)

    hadoopConf.writeTextFile(path + "/_SUCCESS")(out => ())
  }

  def write(path: String, overwrite: Boolean = false, codecSpecJSONStr: String = null): Unit = {
    val codecSpec =
      if (codecSpecJSONStr != null) {
        implicit val formats = RVDSpec.formats
        val codecSpecJSON = parse(codecSpecJSONStr)
        codecSpecJSON.extract[CodecSpec]
      } else
        CodecSpec.default

    if (overwrite)
      hadoopConf.delete(path, recursive = true)
    else if (hadoopConf.exists(path))
      fatal(s"file already exists: $path")

    hc.hadoopConf.mkDir(path)

    val partitionCounts = rvd.rdd.writeRowsSplit(path, matrixType, codecSpec, rvd.partitioner)

    val globalsPath = path + "/globals"
    hadoopConf.mkDir(globalsPath)
    writeGlobals(globalsPath, codecSpec)

    val rowsSpec = TableSpec(
      FileFormat.version.rep,
      hc.version,
      "../references",
      matrixType.rowsTableType,
      Map("globals" -> RVDComponentSpec("../globals/rows"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    rowsSpec.write(hc, path + "/rows")

    hadoopConf.writeTextFile(path + "/rows/_SUCCESS")(out => ())

    val entriesSpec = TableSpec(
      FileFormat.version.rep,
      hc.version,
      "../references",
      matrixType.rowsTableType,
      Map("globals" -> RVDComponentSpec("../globals/rows"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    entriesSpec.write(hc, path + "/entries")

    hadoopConf.writeTextFile(path + "/entries/_SUCCESS")(out => ())

    hadoopConf.mkDir(path + "/cols")
    writeCols(path + "/cols", codecSpec)

    val refPath = path + "/references"
    hc.hadoopConf.mkDir(refPath)
    Array(colType, rowType, entryType, globalType).foreach { t =>
      GenomeReference.exportReferences(hc, refPath, t)
    }

    val spec = MatrixTableSpec(
      FileFormat.version.rep,
      hc.version,
      "references",
      matrixType,
      Map("globals" -> RVDComponentSpec("globals/rows"),
        "cols" -> RVDComponentSpec("cols/rows"),
        "rows" -> RVDComponentSpec("rows/rows"),
        "entries" -> RVDComponentSpec("entries/rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    spec.write(hc, path)

    hadoopConf.writeTextFile(path + "/_SUCCESS")(out => ())
  }

  def linreg(ysExpr: Array[String], xExpr: String, covExpr: Array[String] = Array.empty[String], root: String = "linreg", variantBlockSize: Int = 16): MatrixTable = {
    LinearRegression(this, ysExpr, xExpr, covExpr, root, variantBlockSize)
  }

  def logreg(test: String,
    y: String, x: String, covariates: Array[String] = Array.empty[String],
    root: String = "logreg"): MatrixTable = {
    LogisticRegression(this, test, y, x, covariates, root)
  }

  def lmmreg(kinshipMatrix: KinshipMatrix,
    y: String,
    x: String,
    covariates: Array[String] = Array.empty[String],
    useML: Boolean = false,
    rootGA: String = "global.lmmreg",
    rootVA: String = "lmmreg",
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

    val localRVRowType = rvRowType

    def minRep1(removeLeftAligned: Boolean, removeMoving: Boolean, verifyLeftAligned: Boolean): RDD[RegionValue] = {
      rvd.mapPartitions { it =>
        var prevLocus: Locus = null
        val rvb = new RegionValueBuilder()
        val rv2 = RegionValue()

        it.flatMap { rv =>
          val ur = new UnsafeRow(localRVRowType, rv.region, rv.offset)
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
            rvb.start(localRVRowType)
            rvb.startStruct()
            rvb.addAnnotation(localRVRowType.fieldType(0), minv.locus)
            rvb.addAnnotation(localRVRowType.fieldType(1), minv)
            rvb.addField(localRVRowType, rv, 2)
            rvb.addField(localRVRowType, rv, 3)
            rvb.endStruct()
            rv2.set(rv.region, rvb.end())
            Some(rv2)
          }
        }
      }
    }

    val newRDD2 =
      if (leftAligned)
        OrderedRVD(rvd.typ,
          rvd.partitioner,
          minRep1(removeLeftAligned = false, removeMoving = false, verifyLeftAligned = true))
      else
        SplitMulti.unionMovedVariants(
          OrderedRVD(rvd.typ,
            rvd.partitioner,
            minRep1(removeLeftAligned = false, removeMoving = true, verifyLeftAligned = false)),
          minRep1(removeLeftAligned = true, removeMoving = false, verifyLeftAligned = false))

    copy2(rvd = newRDD2)
  }

  def trioMatrix(pedigree: Pedigree, completeTrios: Boolean): MatrixTable = {
    colKeyTypes match {
      case Array(_: TString) =>
      case _ =>
        fatal(s"trio_matrix requires column keys of type 'String', found [${
          colKeyTypes.map(x => s"'$x'").mkString(", ")
        }]")
    }
    requireUniqueSamples("trio_matrix")

    val filteredPedigree = pedigree.filterTo(stringSampleIds.toSet)
    val trios = if (completeTrios) filteredPedigree.completeTrios else filteredPedigree.trios
    val nTrios = trios.length

    val sampleIndices = stringSampleIds.zipWithIndex.toMap

    val kidIndices = Array.fill[Int](nTrios)(-1)
    val dadIndices = Array.fill[Int](nTrios)(-1)
    val momIndices = Array.fill[Int](nTrios)(-1)

    val newColType = TStruct(
      "id" -> TString(),
      "proband" -> colType,
      "father" -> colType,
      "mother" -> colType,
      "is_female" -> TBooleanOptional,
      "fam_id" -> TStringOptional
    )

    val newColValues = new Array[Annotation](nTrios)

    var i = 0
    while (i < nTrios) {
      val t = trios(i)
      val kidIndex = sampleIndices(t.kid)
      kidIndices(i) = kidIndex
      val kidAnnotation = colValues(kidIndex)

      var dadAnnotation: Annotation = null
      t.dad.foreach { dad =>
        val index = sampleIndices(dad)
        dadIndices(i) = index
        dadAnnotation = colValues(index)
      }

      var momAnnotation: Annotation = null
      t.mom.foreach { mom =>
        val index = sampleIndices(mom)
        momIndices(i) = index
        momAnnotation = colValues(index)
      }

      val isFemale: java.lang.Boolean = (t.sex: @unchecked) match {
        case Some(Sex.Female) => true
        case Some(Sex.Male) => false
        case None => null
      }

      val famID = t.fam.orNull

      newColValues(i) = Row(t.kid, kidAnnotation, dadAnnotation, momAnnotation, isFemale, famID)
      i += 1
    }

    val newEntryType = TStruct(
      "proband_entry" -> entryType,
      "father_entry" -> entryType,
      "mother_entry" -> entryType
    )

    val fullRowType = rvRowType
    val localEntriesIndex = entriesIndex
    val localEntriesType = matrixType.entryArrayType

    insertEntries(noOp, newColType = newColType, newColKey = Array("id"),
      newColValues = newColValues)(newEntryType, { case (_, rv, rvb) =>
      val entriesOffset = fullRowType.loadField(rv, localEntriesIndex)

      rvb.startArray(nTrios)
      var i = 0
      while (i < nTrios) {
        rvb.startStruct()

        // append kid element
        rvb.addElement(localEntriesType, rv.region, entriesOffset, kidIndices(i))

        // append dad element if the dad is defined
        val dadIndex = dadIndices(i)
        if (dadIndex >= 0)
          rvb.addElement(localEntriesType, rv.region, entriesOffset, dadIndex)
        else
          rvb.setMissing()

        // append mom element if the mom is defined
        val momIndex = momIndices(i)
        if (momIndex >= 0)
          rvb.addElement(localEntriesType, rv.region, entriesOffset, momIndex)
        else
          rvb.setMissing()

        rvb.endStruct()

        i += 1
      }
      rvb.endArray()
    })
  }


  def toIndexedRowMatrix(expr: String): IndexedRowMatrix = {
    val partStarts = partitionStarts()
    assert(partStarts.length == rvd.getNumPartitions + 1)
    val partStartsBc = sparkContext.broadcast(partStarts)

    val fullRowType = rvRowType
    val localColValuesBc = colValuesBc
    val localEntriesIndex = entriesIndex
    val rowKeyQuerier = colKeys
    val localRKF = rowKeysF

    val ec = EvalContext(Map(
      "global" -> (0, globalType),
      "va" -> (1, rowType),
      "sa" -> (2, colType),
      "g" -> (3, entryType)))
    val f = RegressionUtils.parseExprAsDouble(expr, ec)
    ec.set(0, globals)

    val indexedRows = rvd.mapPartitionsWithIndex { case (i, it) =>
      val start = partStartsBc.value(i)
      var j = 0
      val fullRow = new UnsafeRow(fullRowType)
      it.map { rv =>
        fullRow.set(rv)
        val row = fullRow.deleteField(localEntriesIndex)
        val entries = fullRow.getAs[IndexedSeq[Any]](localEntriesIndex)
        val ns = entries.length
        val a = new Array[Double](ns)
        var k = 0
        ec.set(1, row)
        while (k < ns) {
          ec.set(2, localColValuesBc.value(k))
          ec.set(3, entries(k))
          a(k) = f() match {
            case null => fatal(s"entry_expr must be non-missing. Found missing value for col $k and row ${ localRKF(row) }")
            case t =>
              val td = t.toDouble
              if (td.isNaN || td.isInfinite)
                fatal(s"entry_expr cannot be NaN or infinite. Found ${ if (td.isNaN) "NaN" else "infinite" } value for col $k and row ${ localRKF(row) }")
              td
          }
          k += 1
        }
        val r = IndexedRow(start + j, Vectors.dense(a))
        j += 1
        r
      }
    }

    // caching is critical before use in computeSVD in PCA
    new IndexedRowMatrix(indexedRows.cache(), partStarts.last, numCols)
  }

  def collectRowKeys(): Keys = {
    val fullRowType = rvRowType
    val localRKF = rowKeysF
    val localKeyStruct = rowKeyStruct
    val values = rvd.mapPartitions { it =>
      val ur = new UnsafeRow(fullRowType)
      it.map { rv =>
        ur.set(rv)
        Annotation.copy(localKeyStruct, localRKF(ur))
      }
    }.collect()

    new Keys(TStruct(rowKey.zip(rowKeyTypes): _*), values)
  }

  def writeKeyedBlockMatrix(dirname: String, expr: String, blockSize: Int = BlockMatrix.defaultBlockSize,
    keepRowKeys: Boolean = true, keepColKeys: Boolean = true): Unit = {

    sparkContext.hadoopConfiguration.mkDir(dirname)

    if (keepRowKeys)
      collectRowKeys().write(sparkContext, dirname + "/rowkeys")

    if (keepColKeys)
      new Keys(TStruct(colKey.zip(colKeyTypes): _*), colKeys.toArray).write(sparkContext, dirname + "/colkeys")
    writeBlockMatrix(dirname + "/blockmatrix", expr, blockSize)
  }

  def writeBlockMatrix(dirname: String, expr: String, blockSize: Int = BlockMatrix.defaultBlockSize): Unit = {
    val partStarts = partitionStarts()
    assert(partStarts.length == rvd.getNumPartitions + 1)

    val ec = EvalContext(Map(
      "global" -> (0, globalType),
      "va" -> (1, rowType),
      "sa" -> (2, colType),
      "g" -> (3, entryType)))
    val f = RegressionUtils.parseExprAsDouble(expr, ec)
    ec.set(0, globals)

    val nRows = partStarts.last
    val localNCols = numCols

    val hadoop = sparkContext.hadoopConfiguration
    hadoop.mkDir(dirname)

    // write metadata
    hadoop.writeDataFile(dirname + BlockMatrix.metadataRelativePath) { os =>
      implicit val formats = defaultJSONFormats
      jackson.Serialization.write(
        BlockMatrixMetadata(blockSize, nRows, localNCols),
        os)
    }

    // write blocks
    hadoop.mkDir(dirname + "/parts")
    val gp = GridPartitioner(blockSize, nRows, localNCols)
    val blockCount =
      new WriteBlocksRDD(dirname, rvd, sparkContext, matrixType,
        sparkContext.broadcast(colValues), partStarts, f, ec, gp)
        .reduce(_ + _)

    assert(blockCount == gp.numPartitions)
    info(s"Wrote all $blockCount blocks of $nRows x $localNCols matrix with block size $blockSize.")

    hadoop.writeTextFile(dirname + "/_SUCCESS")(out => ())
  }

  def indexRows(name: String): MatrixTable = {
    val (newRVType, inserter) = rvRowType.unsafeStructInsert(TInt64(), List(name))

    val partStarts = partitionStarts()
    val newMatrixType = matrixType.copy(rvRowType = newRVType)
    val localEntriesIndex = entriesIndex
    val indexedRVD = rvd.mapPartitionsWithIndexPreservesPartitioning(newMatrixType.orvdType) { case (i, it) =>
      val region2 = Region()
      val rv2 = RegionValue(region2)
      val rv2b = new RegionValueBuilder(region2)

      var idx = partStarts(i)

      it.map { rv =>
        region2.clear()
        rv2b.start(newRVType)

        inserter(rv.region, rv.offset, rv2b,
          () => rv2b.addLong(idx))

        idx += 1
        rv2.setOffset(rv2b.end())
        rv2
      }
    }
    copyMT(matrixType = newMatrixType, rvd = indexedRVD)
  }

  def indexCols(name: String): MatrixTable = {
    val (newColType, inserter) = colType.structInsert(TInt32(), List(name))
    val newColValues = Array.tabulate(numCols) { i =>
      inserter(colValues(i), i)
    }
    copy2(colType = newColType, colValues = newColValues)
  }

  def filterPartitions(parts: java.util.ArrayList[Int], keep: Boolean): MatrixTable =
    filterPartitions(parts.asScala.toArray, keep)

  def filterPartitions(parts: Array[Int], keep: Boolean = true): MatrixTable = {
    copy2(rvd =
      rvd.subsetPartitions(
        if (keep)
          parts
        else {
          val partSet = parts.toSet
          (0 until rvd.getNumPartitions).filter(i => !partSet.contains(i)).toArray
        })
    )
  }
}
