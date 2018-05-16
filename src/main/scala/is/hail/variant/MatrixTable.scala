package is.hail.variant

import java.util

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.linalg._
import is.hail.expr._
import is.hail.expr.ir
import is.hail.expr.ir.{ContainsAgg, InsertFields}
import is.hail.methods._
import is.hail.rvd._
import is.hail.table.{Table, TableSpec}
import is.hail.methods.Aggregators.ColFunctions
import is.hail.utils._
import is.hail.{HailContext, utils}
import is.hail.expr.types._
import is.hail.io.CodecSpec
import is.hail.sparkextras.ContextRDD
import is.hail.io.gen.ExportGen
import is.hail.io.plink.ExportPlink
import is.hail.sparkextras.{AdjustedPartitionsRDD, AdjustedPartitionsRDDPartition, Adjustment, ContextRDD}
import org.apache.hadoop
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Partitioner, SparkContext}
import org.json4s._
import org.json4s.jackson.JsonMethods.parse
import org.json4s.jackson.{JsonMethods, Serialization}

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
    if (!hc.hadoopConf.isDir(path))
      fatal(s"MatrixTable and Table files are directories; path '$path' is not a directory")
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

    ReferenceGenome.importReferences(hc.hadoopConf, path + "/" + referencesRelPath)

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
    dropCols: Boolean = false, dropRows: Boolean = false): MatrixTable = {
    val spec = (RelationalSpec.read(hc, path): @unchecked) match {
      case mts: MatrixTableSpec => mts
      case _: TableSpec => fatal(s"file is a Table, not a MatrixTable: '$path'")
    }

    val typ = spec.matrix_type

    val f: (MatrixRead) => MatrixValue = { mr =>
      val hc = HailContext.get
      val globals = Annotation.copy(typ.globalType, spec.globalsComponent.readLocal(hc, path)(0)).asInstanceOf[Row]

      val colAnnotations =
        if (mr.dropCols)
          IndexedSeq.empty[Annotation]
        else
          Annotation.copy(TArray(typ.colType), spec.colsComponent.readLocal(hc, path)).asInstanceOf[IndexedSeq[Annotation]]

      val rvd =
        if (mr.dropRows)
          OrderedRVD.empty(hc.sc, typ.orvdType)
        else {
          val fullRowType = typ.rvRowType
          val rowType = typ.rowType
          val localEntriesIndex = typ.entriesIdx

          val rowsRVD = spec.rowsComponent.read(hc, path).asInstanceOf[OrderedRVD]
          if (mr.dropCols) {
            rowsRVD.mapPartitionsPreservesPartitioning(typ.orvdType) { it =>
              val rv2b = new RegionValueBuilder()
              val rv2 = RegionValue()

              it.map { rv =>
                rv2b.set(rv.region)
                rv2b.start(fullRowType)
                rv2b.startStruct()
                var i = 0
                while (i < localEntriesIndex) {
                  rv2b.addField(rowType, rv, i)
                  i += 1
                }
                rv2b.startArray(0)
                rv2b.endArray()
                i += 1
                while (i < fullRowType.size) {
                  rv2b.addField(rowType, rv, i - 1)
                  i += 1
                }
                rv2b.endStruct()
                rv2.set(rv.region, rv2b.end())
                rv2
              }
            }
          } else {
            val entriesRVD = spec.entriesComponent.read(hc, path)
            val entriesRowType = entriesRVD.rowType
            rowsRVD.zip(typ.orvdType, entriesRVD) { (ctx, rv1, rv2) =>
              val rvb = ctx.rvb
              val region = ctx.region
              rvb.start(fullRowType)
              rvb.startStruct()
              var i = 0
              while (i < localEntriesIndex) {
                rvb.addField(rowType, rv1, i)
                i += 1
              }
              rvb.addField(entriesRowType, rv2, 0)
              i += 1
              while (i < fullRowType.size) {
                rvb.addField(rowType, rv1, i - 1)
                i += 1
              }
              rvb.endStruct()
              rv2.set(region, rvb.end())
              rv2
            }
          }
        }

      MatrixValue(
        typ,
        BroadcastRow(globals, typ.globalType, hc.sc),
        BroadcastIndexedSeq(colAnnotations, TArray(typ.colType), hc.sc),
        rvd)
    }

    new MatrixTable(hc,
      MatrixRead(typ, Some(spec.partitionCounts), dropCols, dropRows, f))
  }

  def fromLegacy[T](hc: HailContext,
    matrixType: MatrixType,
    globals: Annotation,
    colValues: IndexedSeq[Annotation],
    rdd: RDD[(Annotation, Iterable[T])]): MatrixTable = {

    val localGType = matrixType.entryType
    val localRVRowType = matrixType.rvRowType

    val localNCols = colValues.length

    var ds = new MatrixTable(hc, matrixType,
      BroadcastRow(globals.asInstanceOf[Row], matrixType.globalType, hc.sc),
      BroadcastIndexedSeq(colValues, TArray(matrixType.colType), hc.sc),
      OrderedRVD.coerce(matrixType.orvdType,
        ContextRDD.weaken[RVDContext](rdd).cmapPartitions { (ctx, it) =>
          val region = ctx.region
          val rvb = new RegionValueBuilder(region)
          val rv = RegionValue(region)

          it.map { case (va, gs) =>
            val vaRow = va.asInstanceOf[Row]
            assert(matrixType.rowType.typeCheck(vaRow), s"${ matrixType.rowType }, $vaRow")

            rvb.start(localRVRowType)
            rvb.startStruct()
            var i = 0
            while (i < vaRow.length) {
              rvb.addAnnotation(localRVRowType.types(i), vaRow.get(i))
              i += 1
            }
            rvb.startArray(localNCols) // gs
            gs.foreach { g => rvb.addAnnotation(localGType, g) }
            rvb.endArray() // gs
            rvb.endStruct()
            rv.setOffset(rvb.end())

            rv
          }
        }, None, None))
    ds.typecheck()
    ds
  }

  def range(hc: HailContext, nRows: Int, nCols: Int, nPartitions: Option[Int]): MatrixTable = {
    val nPartitionsAdj = math.min(nRows, nPartitions.getOrElse(hc.sc.defaultParallelism))
    val partCounts = partition(nRows, nPartitionsAdj)

    val typ: MatrixType = MatrixType.fromParts(
      globalType = TStruct.empty(),
      colKey = Array("col_idx"),
      colType = TStruct("col_idx" -> TInt32()),
      rowPartitionKey = Array("row_idx"),
      rowKey = Array("row_idx"),
      rowType = TStruct("row_idx" -> TInt32()),
      entryType = TStruct.empty())

    val f: (MatrixRead) => MatrixValue = { mr =>
      val hc = HailContext.get
      val localRVType = typ.rvRowType
      val localPartCounts = partCounts
      val partStarts = partCounts.scanLeft(0)(_ + _)
      val localNCols = if (mr.dropCols) 0 else nCols

      val rvd = if (mr.dropRows)
        OrderedRVD.empty(hc.sc, typ.orvdType)
      else {
        OrderedRVD(typ.orvdType,
          new OrderedRVDPartitioner(typ.rowPartitionKey.toArray,
            typ.rowKeyStruct,
            Array.tabulate(nPartitionsAdj) { i =>
              val start = partStarts(i)
              Interval(Row(start), Row(start + localPartCounts(i)), includesStart = true, includesEnd = false)
            }),
          ContextRDD.parallelize[RVDContext](hc.sc, Range(0, nPartitionsAdj), nPartitionsAdj)
            .cmapPartitionsWithIndex { (i, ctx, _) =>
              val region = ctx.region
              val rvb = ctx.rvb
              val rv = RegionValue(region)

              val start = partStarts(i)
              Iterator.range(start, start + localPartCounts(i))
                .map { j =>
                  rvb.start(localRVType)
                  rvb.startStruct()

                  // row idx field
                  rvb.addInt(j)

                  // entries field
                  rvb.startArray(localNCols)
                  var i = 0
                  while (i < localNCols) {
                    rvb.startStruct()
                    rvb.endStruct()
                    i += 1
                  }
                  rvb.endArray()

                  rvb.endStruct()
                  rv.setOffset(rvb.end())
                  rv
                }
            })
      }

      MatrixValue(typ,
        BroadcastRow(Row(), typ.globalType, hc.sc),
        BroadcastIndexedSeq(
          Iterator.range(0, localNCols)
            .map(Row(_))
            .toFastIndexedSeq,
          TArray(typ.colType),
          hc.sc),
        rvd)
    }

    new MatrixTable(hc, MatrixRead(typ, Some(partCounts.map(_.toLong)), dropRows = false, dropCols = false, f = f))
  }

  def gen(hc: HailContext, gen: VSMSubgen): Gen[MatrixTable] =
    gen.gen(hc)

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
    checkDatasetSchemasCompatible(datasets)
    first.copyMT(rvd = OrderedRVD.union(datasets.map(_.rvd)))
  }

  def fromRowsTable(kt: Table, partitionKey: java.util.ArrayList[String] = null): MatrixTable = {
    require(!kt.key.isEmpty)
    val matrixType = MatrixType.fromParts(
      kt.globalSignature,
      Array.empty[String],
      TStruct.empty(),
      Option(partitionKey).map(_.asScala.toArray.toFastIndexedSeq).getOrElse(kt.key.get),
      kt.key.get,
      kt.signature,
      TStruct.empty()
    )
    val rvRowType = matrixType.rvRowType

    val oldRowType = kt.signature

    val rdd = kt.rvd.mapPartitions(matrixType.rvRowType) { it =>
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

    new MatrixTable(kt.hc, matrixType,
      BroadcastRow(Row(), matrixType.globalType, kt.hc.sc),
      BroadcastIndexedSeq(Array.empty[Annotation], TArray(matrixType.colType), kt.hc.sc),
      OrderedRVD.coerce(matrixType.orvdType, rdd))
  }
}

case class VSMSubgen(
  sSigGen: Gen[Type],
  saSigGen: Gen[TStruct],
  vSigGen: Gen[Type],
  rowPartitionKeyGen: (Type) => Gen[Array[String]],
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
    for {
      size <- Gen.size
      (l, w) <- Gen.squareOfAreaAtMostSize.resize((size / 3 / 10) * 8)

      vSig <- vSigGen.resize(3)
      rowPartitionKey <- rowPartitionKeyGen(vSig)
      vaSig <- vaSigGen.map(t => t.deepOptional().asInstanceOf[TStruct]).resize(3)
      sSig <- sSigGen.resize(3)
      saSig <- saSigGen.map(t => t.deepOptional().asInstanceOf[TStruct]).resize(3)
      globalSig <- globalSigGen.resize(5)
      tSig <- tSigGen.map(t => t.structOptional().asInstanceOf[TStruct]).resize(3)
      global <- globalGen(globalSig).resize(25)
      nPartitions <- Gen.choose(1, 10)

      sampleIds <- Gen.buildableOfN[Array](w, sGen(sSig).resize(3))
        .map(ids => ids.distinct)
      nSamples = sampleIds.length
      saValues <- Gen.buildableOfN[Array](nSamples, saGen(saSig).resize(5))
      rows <- Gen.buildableOfN[Array](l,
        for {
          v <- vGen(vSig).resize(3)
          va <- vaGen(vaSig).resize(5)
          ts <- Gen.buildableOfN[Array](nSamples, tGen(tSig, v).resize(3))
        } yield (v, (va, ts: Iterable[Annotation])))
    } yield {
      assert(sampleIds.forall(_ != null))
      val (finalSASig, sIns) = saSig.structInsert(sSig, List("s"))

      val (finalVASig, vaIns, finalRowPartitionKey, rowKey) =
        vSig match {
          case vSig: TStruct =>
            val (finalVASig, vaIns) = vaSig.annotate(vSig)
            (finalVASig, vaIns, rowPartitionKey, vSig.fieldNames)
          case _ =>
            val (finalVASig, vaIns) = vaSig.structInsert(vSig, List("v"))
            (finalVASig, vaIns, Array("v"), Array("v"))
        }

      MatrixTable.fromLegacy(hc,
        MatrixType.fromParts(globalSig, Array("s"), finalSASig, finalRowPartitionKey, rowKey, finalVASig, tSig),
        global,
        sampleIds.zip(saValues).map { case (id, sa) => sIns(sa, id) },
        hc.sc.parallelize(rows.map { case (v, (va, gs)) =>
          (vaIns(va, v), gs)
        }, nPartitions))
        .deduplicate()
    }
}

object VSMSubgen {
  val random = VSMSubgen(
    sSigGen = Gen.const(TString()),
    saSigGen = Type.genInsertable,
    vSigGen = ReferenceGenome.gen.map(rg =>
      TStruct(
        "locus" -> TLocus(rg),
        "alleles" -> TArray(TString()))),
    rowPartitionKeyGen = (t: Type) => Gen.const(Array("locus")),
    vaSigGen = Type.genInsertable,
    globalSigGen = Type.genInsertable,
    tSigGen = Gen.const(Genotype.htsGenotypeType),
    sGen = (t: Type) => Gen.identifier.map(s => s: Annotation),
    saGen = (t: Type) => t.genValue,
    vaGen = (t: Type) => t.genValue,
    globalGen = (t: Type) => t.genNonmissingValue,
    vGen = (t: Type) => {
      val rg = t.asInstanceOf[TStruct]
        .field("locus")
        .typ
        .asInstanceOf[TLocus]
        .rg.asInstanceOf[ReferenceGenome]
      VariantSubgen.random(rg).genLocusAlleles
    },
    tGen = (t: Type, v: Annotation) => Genotype.genExtreme(
      v.asInstanceOf[Row]
        .getAs[IndexedSeq[String]](1)
        .length))

  val plinkSafeBiallelic: VSMSubgen = random.copy(
    vSigGen = Gen.const(TStruct(
      "locus" -> TLocus(ReferenceGenome.GRCh37),
      "alleles" -> TArray(TString()))),
    sGen = (t: Type) => Gen.plinkSafeIdentifier,
    vGen = (t: Type) => VariantSubgen.plinkCompatibleBiallelic(ReferenceGenome.GRCh37).genLocusAlleles)

  val callAndProbabilities = VSMSubgen(
    sSigGen = Gen.const(TString()),
    saSigGen = Type.genInsertable,
    vSigGen = Gen.const(
      TStruct(
        "locus" -> TLocus(ReferenceGenome.defaultReference),
        "alleles" -> TArray(TString()))),
    rowPartitionKeyGen = (t: Type) => Gen.const(Array("locus")),
    vaSigGen = Type.genInsertable,
    globalSigGen = Type.genInsertable,
    tSigGen = Gen.const(TStruct(
      "GT" -> TCall(),
      "GP" -> TArray(TFloat64()))),
    sGen = (t: Type) => Gen.identifier.map(s => s: Annotation),
    saGen = (t: Type) => t.genValue,
    vaGen = (t: Type) => t.genValue,
    globalGen = (t: Type) => t.genValue,
    vGen = (t: Type) => VariantSubgen.random(ReferenceGenome.defaultReference).genLocusAlleles,
    tGen = (t: Type, v: Annotation) => Genotype.genGenericCallAndProbabilitiesGenotype(
      v.asInstanceOf[Row]
        .getAs[IndexedSeq[String]](1)
        .length))

  val realistic = random.copy(
    tGen = (t: Type, v: Annotation) => Genotype.genRealistic(
      v.asInstanceOf[Row]
        .getAs[IndexedSeq[String]](1)
        .length))
}

class MatrixTable(val hc: HailContext, val ast: MatrixIR) {

  def this(hc: HailContext,
    matrixType: MatrixType,
    globals: BroadcastRow,
    colValues: BroadcastIndexedSeq,
    rvd: OrderedRVD) =
    this(hc,
      MatrixLiteral(
        matrixType,
        MatrixValue(matrixType, globals, colValues, rvd)))

  private[this] type Axis = Int
  private[this] val globalAxis: Axis = 0
  private[this] val rowAxis: Axis = 1
  private[this] val colAxis: Axis = 2
  private[this] val entryAxis: Axis = 3

  private[this] def useIR(axis: Axis, ast: AST): Boolean = {
    if (hc.forceIR)
      return true
    axis match {
      case this.globalAxis | this.colAxis =>
        !ast.`type`.isInstanceOf[TStruct] ||
          ast.`type`.asInstanceOf[TStruct].size < 500

      case this.rowAxis | this.entryAxis =>
        !ast.`type`.isInstanceOf[TStruct] ||
          ast.`type`.asInstanceOf[TStruct].size < 500 &&
            globalType.size < 3 &&
            (colType.size < 1 ||
            colType.size == 1 &&
            Set(TInt32(), +TInt32(), TString(), +TString())
              .contains(colType.types(0)))
    }
  }

  def requireRowKeyVariant(method: String) {
    rowKey.zip(rowKeyTypes) match {
      case IndexedSeq(("locus", TLocus(_, _)), ("alleles", TArray(TString(_), _))) =>
      case _ =>
        fatal(s"in $method: row key must be ('locus' (type 'locus'), 'alleles': (type 'array<str>'), found: ${
          rowKey.zip(rowKeyTypes).mkString(", ")
        }")
    }
  }

  def requirePartitionKeyLocus(method: String) {
    rowPartitionKeyTypes match {
      case Array(_: TLocus) =>
      case t =>
        fatal(s"in $method: partition key must be type 'locus', found: $t")
    }
  }

  def requireColKeyString(method: String) {
    colKeyTypes match {
      case Array(_: TString) =>
      case t =>
        fatal(s"in $method: column key must be type 'str', found: $t")
    }
  }

  def referenceGenome: ReferenceGenome = {
    val firstKeyField = rowKeyTypes(0)
    firstKeyField match {
      case TLocus(rg: ReferenceGenome, _) => rg
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
    .map(s => matrixType.colType.types(matrixType.colType.fieldIdx(s)))
    .toArray

  val rowKeyTypes: Array[Type] = rowKey
    .map(s => matrixType.rowType.types(matrixType.rowType.fieldIdx(s)))
    .toArray

  val rowKeyStruct: TStruct = TStruct(rowKey.zip(rowKeyTypes): _*)

  val rowPartitionKeyTypes: Array[Type] = rowPartitionKey
    .map(s => matrixType.rowType.types(matrixType.rowType.fieldIdx(s)))
    .toArray

  lazy val value: MatrixValue = {
    val opt = ir.Optimize(ast)

    log.info("in MatrixTable.value: execute:\n" + ir.Pretty(opt))

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

  // length nPartitions + 1, first element 0, last element rvd count
  def partitionStarts(): Array[Long] = partitionCounts().scanLeft(0L)(_ + _)

  def colKeys: IndexedSeq[Annotation] = {
    val queriers = colKey.map(colType.query(_))
    colValues.value.map(a => Row.fromSeq(queriers.map(q => q(a)))).toArray[Annotation]
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
    assert(partitionKeys.length <= keys.length)
    assert(keys.zip(partitionKeys).forall{ case (k, pk) => k == pk })

    val newMatrixType = matrixType.copy(rowKey = keys,
      rowPartitionKey = partitionKeys)

    copyMT(matrixType = newMatrixType,
      rvd = OrderedRVD.coerce(newMatrixType.orvdType, rvd))
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
    colValues.value.map(querier(_).asInstanceOf[String])
  }

  def stringSampleIdSet: Set[String] = stringSampleIds.toSet

  def requireUniqueSamples(method: String) {
    val dups = stringSampleIds.counter().filter(_._2 > 1).toArray
    if (dups.nonEmpty)
      fatal(s"Method '$method' does not support duplicate column keys. Duplicates:" +
        s"\n  @1", dups.sortBy(-_._2).map { case (id, count) => s"""($count) "$id"""" }.truncatable("\n  "))
  }

  def collectColsByKey(): MatrixTable = new MatrixTable(hc, CollectColsByKey(ast))

  def groupColsBy(keyExpr: String, aggExpr: String): MatrixTable = {
    val sEC = EvalContext(Map(Annotation.GLOBAL_HEAD -> (0, globalType),
      Annotation.COL_HEAD -> (1, colType)))
    val (keyNames, keyTypes, keyFs) = Parser.parseNamedExprs(keyExpr, sEC)
    sEC.set(0, globals.value)
    val keysBySample = colValues.value.map { sa =>
      sEC.set(1, sa)
      Row.fromSeq(keyFs())
    }
    val newKeys = keysBySample.toSet.toArray
    val keyMap = newKeys.zipWithIndex.toMap
    val samplesMap = keysBySample.map { k => if (k == null) -1 else keyMap(k) }.toArray

    val nKeys = newKeys.size

    val ec = rowEC
    val (newEntryNames, newEntryTypes, entryF) = Parser.parseNamedExprs(aggExpr, ec)
    val newEntryType = TStruct(newEntryNames.zip(newEntryTypes): _*)

    val newColKey = keyNames
    val newColType = TStruct(keyNames.zip(keyTypes): _*)

    val aggregate = Aggregators.buildRowAggregationsByKey(this, nKeys, samplesMap, ec)

    insertEntries(noOp,
      newColType = newColType,
      newColValues = colValues.copy(value = newKeys, t = TArray(newColType)),
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
          rvb.addAnnotation(newEntryType.types(j), fields(j))
          j += 1
        }
        rvb.endStruct()
        i += 1
      }
      rvb.endArray()
    })
  }

  def aggregateRowsByKey(aggExpr: String): MatrixTable = {

    val ColFunctions(zero, seqOp, resultOp, newEntryType) = Aggregators.makeColFunctions(this, aggExpr)
    val newRowType = matrixType.orvdType.kType
    val newMatrixType = MatrixType.fromParts(globalType, colKey, colType,
      rowPartitionKey, rowKey, newRowType, newEntryType)

    val newRVType = newMatrixType.rvRowType
    val localRVType = rvRowType
    val selectIdx = matrixType.orvdType.kRowFieldIdx
    val keyOrd = matrixType.orvdType.kRowOrd

    val newRVD = rvd.boundary.mapPartitionsPreservesPartitioning(newMatrixType.orvdType, { (ctx, it) =>
      new Iterator[RegionValue] {
        var isEnd = false
        var current: RegionValue = null
        val rvRowKey: WritableRegionValue = WritableRegionValue(newRowType, ctx.freshRegion)
        val region = ctx.region
        val rvb = ctx.rvb
        val newRV = RegionValue(region)

        def hasNext: Boolean = {
          if (isEnd || (current == null && !it.hasNext)) {
            isEnd = true
            return false
          }
          if (current == null)
            current = it.next()
          true
        }

        def next(): RegionValue = {
          if (!hasNext)
            throw new java.util.NoSuchElementException()
          rvRowKey.setSelect(localRVType, selectIdx, current)
          var aggs = zero()
          while (hasNext && keyOrd.equiv(rvRowKey.value, current)) {
            aggs = seqOp(aggs, current)
            current = null
          }
          rvb.start(newRVType)
          rvb.startStruct()
          var i = 0
          while (i < newRowType.size) {
            rvb.addField(newRowType, rvRowKey.value, i)
            i += 1
          }
          resultOp(aggs, rvb)
          rvb.endStruct()
          newRV.setOffset(rvb.end())
          newRV
        }
      }
    })

    copyMT(rvd = newRVD, matrixType = newMatrixType)
  }

  def annotateGlobal(a: Annotation, t: Type, name: String): MatrixTable = {
    val at = TStruct(name -> t)
    val value = BroadcastRow(Row(a), at, hc.sc)
    new MatrixTable(hc, MatrixMapGlobals(ast,
      ir.InsertFields(ir.Ref("global", ast.typ.globalType), FastSeq(name -> ir.GetField(ir.Ref(s"value", at), name))), value))
  }

  def annotateGlobalJSON(data: String, t: TStruct): MatrixTable = {
    val ann = JSONAnnotationImpex.importAnnotation(JsonMethods.parse(data), t)
    val value = BroadcastRow(ann.asInstanceOf[Row], t, hc.sc)
    new MatrixTable(hc, MatrixMapGlobals(ast,
      ir.InsertFields(ir.Ref("global", matrixType.globalType),
        t.fieldNames.map(name => name -> ir.GetField(ir.Ref("value", t), name))),
      value))
  }

  def annotateCols(signature: Type, path: List[String], annotations: Array[Annotation]): MatrixTable = {
    val (t, ins) = insertSA(signature, path)

    val newAnnotations = new Array[Annotation](numCols)

    for (i <- colValues.value.indices) {
      newAnnotations(i) = ins(colValues.value(i), annotations(i))
      t.typeCheck(newAnnotations(i))
    }

    copyMT(matrixType = matrixType.copy(colType = t), colValues = colValues.copy(value = newAnnotations, t = TArray(t)))
  }

  def annotateColsTable(kt: Table, root: String): MatrixTable = {
    require(kt.keyFields.isDefined)

    val (finalType, inserter) = colType.structInsert(
      kt.valueSignature,
      List(root))

    val keyTypes = kt.keyFields.get.map(_.typ)

    val keyedRDD = kt.keyedRDD().filter { case (k, v) => k.toSeq.forall(_ != null) }

    assert(keyTypes.length == colKeyTypes.length
      && keyTypes.zip(colKeyTypes).forall { case (l, r) => l.isOfType(r) },
      s"MT col key: ${ colKeyTypes.mkString(", ") }, TB key: ${ keyTypes.mkString(", ") }")
    val r = keyedRDD.map { case (k, v) => (k: Annotation, v: Annotation) }

    val m = r.collectAsMap()

    annotateCols(finalType, inserter) { case (ck, _) => m.getOrElse(ck, null) }
  }

  def annotateCols(newSignature: TStruct, inserter: Inserter)(f: (Annotation, Int) => Annotation): MatrixTable = {
    val newAnnotations = colKeys.zip(colValues.value)
      .zipWithIndex
      .map { case ((ck, sa), i) =>
        val newAnnotation = inserter(sa, f(ck, i))
        newSignature.typeCheck(newAnnotation)
        newAnnotation
      }

    val newFields = newSignature.fieldNames.toSet
    copy2(colValues = colValues.copy(value = newAnnotations, t = TArray(newSignature)),
      colType = newSignature,
      colKey = colKey.filter(newFields.contains))
  }

  def orderedRVDLeftJoinDistinctAndInsert(right: OrderedRVD, root: String, product: Boolean): MatrixTable = {
    assert(!rowKey.contains(root))

    val valueType = if (product)
      TArray(right.typ.valueType, required = true)
    else
      right.typ.valueType

    val rightRVD = if (product)
      right.groupByKey(" !!! values !!! ")
    else
      right

    val (newRVType, ins) = rvRowType.unsafeStructInsert(valueType, List(root))

    val rightRowType = rightRVD.rowType
    val leftRowType = rvRowType

    val rightValueIndices = rightRVD.typ.valueIndices
    assert(!product || rightValueIndices.length == 1)

    val joiner = { (ctx: RVDContext, it: Iterator[JoinedRegionValue]) =>
      val rvb = ctx.rvb
      val rv = RegionValue()

      it.map { jrv =>
        val lrv = jrv.rvLeft
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
        rv.set(ctx.region, rvb.end())
        rv
      }
    }

    val newMatrixType = matrixType.copy(rvRowType = newRVType)
    val joinedRVD = this.rvd.keyBy(rowKey.take(right.typ.key.length).toArray).orderedJoinDistinct(
      right.keyBy(),
      "left",
      joiner,
      newMatrixType.orvdType
    )

    copyMT(matrixType = newMatrixType, rvd = joinedRVD)
  }

  private def annotateRowsIntervalTable(kt: Table, root: String, product: Boolean): MatrixTable = {
    assert(kt.key.isDefined)
    assert(rowPartitionKeyTypes.length == 1)
    assert(kt.keySignature.get.size == 1)
    assert(kt.keySignature.get.types(0) == TInterval(rowPartitionKeyTypes(0)))

    val typOrdering = rowPartitionKeyTypes(0).ordering

    val typToInsert: Type = if (product) TArray(kt.valueSignature) else kt.valueSignature

    val (newRVType, ins) = rvRowType.unsafeStructInsert(typToInsert, List(root))

    val partBc = sparkContext.broadcast(rvd.partitioner)
    val ktSignature = kt.signature
    val ktKeyFieldIdx = kt.keyFieldIdx.get(0)
    val pType = coerce[TInterval](ktSignature.types(ktKeyFieldIdx)).pointType
    val ktValueFieldIdx = kt.valueFieldIdx
    val partitionKeyedIntervals = kt.rvd.boundary.crdd
      .cflatMap { (ctx, rv) =>
        val region = ctx.region
        val r = SafeRow(ktSignature, rv)
        val ur = new UnsafeRow(ktSignature, rv)
        val interval = ur.getAs[Interval](ktKeyFieldIdx)
        if (interval != null) {
          val rangeTree = partBc.value.rangeTree
          val pkOrd = partBc.value.pkType.ordering
          val wrappedInterval = interval.copy(
            start = Row(Annotation.copy(pType, interval.start)),
            end = Row(Annotation.copy(pType, interval.end)))
          rangeTree.queryOverlappingValues(pkOrd, wrappedInterval).map(i => (i, r))
        } else
          Iterator()
      }

    val nParts = rvd.getNumPartitions
    val zipRDD = partitionKeyedIntervals.partitionBy(new Partitioner {
      def getPartition(key: Any): Int = key.asInstanceOf[Int]

      def numPartitions: Int = nParts
    }).values

    val localRVRowType = rvRowType
    val pkIndex = rvRowType.fieldIdx(rowPartitionKey(0))
    val newMatrixType = matrixType.copy(rvRowType = newRVType)
    val newRVD = rvd.zipPartitionsPreservesPartitioning(
      newMatrixType.orvdType,
      zipRDD
    ) { case (it, intervals) =>
      val intervalAnnotations: Array[(Interval, Any)] =
        intervals.map { r =>
          val interval = r.getAs[Interval](ktKeyFieldIdx)
          (interval, Row.fromSeq(ktValueFieldIdx.map(r.get)))
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

    copyMT(rvd = newRVD, matrixType = newMatrixType)
  }

  def annotateRowsTable(kt: Table, root: String, product: Boolean = false): MatrixTable = {
    assert(kt.key.isDefined)
    assert(!rowKey.contains(root))

    val keyTypes = kt.keyFields.get.map(_.typ)
    if (keyTypes.sameElements(rowKeyTypes) || keyTypes.sameElements(rowPartitionKeyTypes)) {
      val rightORVD = kt.rvd match {
        case ordered: OrderedRVD => ordered
        case unordered =>
          val ordType = new OrderedRVDType(
            kt.key.get.toArray.take(rowPartitionKey.length),
            kt.key.get.toArray,
            kt.signature)
          unordered.constrainToOrderedPartitioner(ordType, rvd.partitioner)
      }
      orderedRVDLeftJoinDistinctAndInsert(rightORVD, root, product)
    } else if (
      keyTypes.length == 1 &&
      rowPartitionKeyTypes.length == 1 &&
      keyTypes(0) == TInterval(rowPartitionKeyTypes(0))
    ) {
      annotateRowsIntervalTable(kt, root, product)
    } else {
      fatal(
        s"""method 'annotate_rows_table' expects a key table keyed by one of the following:
           |  [ ${ rowKeyTypes.mkString(", ") } ]
           |  [ ${ rowPartitionKeyTypes.mkString(", ") } ]
           |  Found key [ ${ keyTypes.mkString(", ") } ] instead.""".stripMargin)
    }
  }

  def selectGlobals(expr: String): MatrixTable = {
    val ec = EvalContext(Map("global" -> (0, globalType)))

    val globalAST = Parser.parseToAST(expr, ec)
    assert(globalAST.`type`.isInstanceOf[TStruct])

    globalAST.toIROpt() match {
      case Some(ir) if useIR(this.globalAxis, globalAST) =>
        new MatrixTable(hc, MatrixMapGlobals(ast, ir, BroadcastRow(Row(), TStruct(), hc.sc)))
      case _ =>
        val (t, f) = Parser.parseExpr(expr, ec)
        val newSignature = t.asInstanceOf[TStruct]

        ec.set(0, globals.value)
        val newGlobal = f().asInstanceOf[Row]

        copyMT(matrixType = matrixType.copy(globalType = newSignature), globals = globals.copy(value = newGlobal, t = newSignature))
    }
  }

  def selectCols(expr: String, newKey: java.util.ArrayList[String]): MatrixTable =
    selectCols(expr, Option(newKey).map(_.asScala.toFastIndexedSeq))

  def selectCols(expr: String, newKey: Option[IndexedSeq[String]]): MatrixTable = {
    val ec = colEC
    val colsAST = Parser.parseToAST(expr, ec)
    val newColKey = newKey.getOrElse(colKey)

    colsAST.toIROpt(Some("AGG" -> "g")) match {
      case Some(x) if useIR(this.colAxis, colsAST) =>
        new MatrixTable(hc, MatrixMapCols(ast, x, newKey))

      case _ =>
        val (t, f) = Parser.parseExpr(expr, ec)
        val newColType = coerce[TStruct](t)
        val namesSet = newColType.fieldNames.toSet
        val newColKey = newKey.getOrElse(colKey)

        val newMatrixType = matrixType.copy(colType = newColType, colKey = newColKey)
        val aggOption = Aggregators.buildColAggregations(hc, value, ec)

        ec.set(0, globals.value)
        val newColValues = Array.tabulate(numCols) { i =>
          ec.set(1, colValues.value(i))
          aggOption.foreach(_ (i))
          f()
        }
        copyMT(matrixType = newMatrixType,
          colValues = colValues.copy(newColValues, TArray(newColType)))
    }
  }

  def selectRows(expr: String, newKey: java.util.ArrayList[java.util.ArrayList[String]]): MatrixTable = {
    assert(Option(newKey).forall(_.size() == 2))
    selectRows(expr, Option(newKey).map(k => (k.get(0).asScala.toFastIndexedSeq, k.get(1).asScala.toFastIndexedSeq)))
  }

  def selectRows(expr: String, newKey: Option[(IndexedSeq[String], IndexedSeq[String])]): MatrixTable = {
    val ec = rowEC

    val rowsAST = Parser.parseToAST(expr, ec)

    rowsAST.toIROpt(Some("AGG" -> "g")) match {
      case Some(x) if useIR(this.rowAxis, rowsAST) =>
        new MatrixTable(hc, MatrixMapRows(ast, x, newKey))

      case _ =>
        val newRowType = coerce[TStruct](rowsAST.`type`)
        val newRowKey = newKey.map{ case (pk, k) => pk ++ k }.getOrElse(rowKey)
        val newPartitionKey = newKey.map{ case (pk, _) => pk }.getOrElse(rowPartitionKey)

        val newEntriesIndex = newRowType.fieldIdx.get(MatrixType.entriesIdentifier)

        val newMatrixType = matrixType.copyParts(
          rowType = newEntriesIndex.map(idx => newRowType.deleteKey(MatrixType.entriesIdentifier, idx)).getOrElse(newRowType),
          rowKey = newRowKey,
          rowPartitionKey = newPartitionKey)
        val fullRowType = rvRowType
        val localEntriesIndex = entriesIndex
        val newRVType = newMatrixType.rvRowType

        val (t, f) = Parser.parseExpr(expr, ec)
        assert(t == newRowType)
        val aggregateOption = Aggregators.buildRowAggregations(this, ec)
        val globalsBc = globals.broadcast
        val mapPartitionsF: Iterator[RegionValue] => Iterator[RegionValue] = { it =>
          val fullRow = new UnsafeRow(fullRowType)
          val rv2 = RegionValue()
          val rvb = new RegionValueBuilder()
          it.map { rv =>
            fullRow.set(rv)
            ec.set(0, globalsBc.value)
            ec.set(1, fullRow)
            aggregateOption.foreach(_ (rv))
            val results = f().asInstanceOf[Row]

            rvb.set(rv.region)
            rvb.start(newRVType)
            rvb.startStruct()
            var i = 0
            while (i < newRowType.size) {
              if (newEntriesIndex.forall(_ != i))
                rvb.addAnnotation(newRowType.types(i), results(i))
              i += 1
            }
            rvb.addField(fullRowType, rv, localEntriesIndex)
            rvb.endStruct()
            rv2.set(rv.region, rvb.end())
            rv2
          }
        }
        if (newKey.isDefined) {
          warn("modified row key, rescanning to compute ordering...")
          val newRDD = rvd.mapPartitions(newMatrixType.rvRowType)(mapPartitionsF)
          copyMT(matrixType = newMatrixType,
            rvd = OrderedRVD.coerce(newMatrixType.orvdType, newRDD, None, None))
        } else copyMT(matrixType = newMatrixType,
          rvd = rvd.mapPartitionsPreservesPartitioning(newMatrixType.orvdType)(mapPartitionsF))
    }
  }

  def selectEntries(expr: String): MatrixTable = {
    val ec = entryEC

    val entryAST = Parser.parseToAST(expr, ec)
    assert(entryAST.`type`.isInstanceOf[TStruct])

    entryAST.toIROpt() match {
      case Some(ir) if useIR(this.entryAxis, entryAST) =>
        new MatrixTable(hc, MatrixMapEntries(ast, ir))
      case _ =>
        val (t, f) = Parser.parseExpr(expr, ec)
        val newEntryType = t.asInstanceOf[TStruct]
        val globalsBc = globals.broadcast
        val fullRowType = rvRowType
        val localEntriesIndex = entriesIndex
        val localNCols = numCols
        val localColValuesBc = colValues.broadcast

        insertEntries(() => {
          val fullRow = new UnsafeRow(fullRowType)
          ec.set(0, globalsBc.value)
          fullRow
        })(newEntryType, { case (fullRow, rv, rvb) =>
          fullRow.set(rv)
          ec.set(1, fullRow)
          val entries = fullRow.getAs[IndexedSeq[Annotation]](localEntriesIndex)
          rvb.startArray(localNCols)
          var i = 0
          while (i < localNCols) {
            val entry = entries(i)
            ec.set(2, localColValuesBc.value(i))
            ec.set(3, entry)
            val result = f()
            rvb.addAnnotation(newEntryType, result)
            i += 1
          }
          rvb.endArray()
        })
    }
  }

  def nPartitions: Int = rvd.getNumPartitions

  def annotateRowsVDS(right: MatrixTable, root: String): MatrixTable =
    orderedRVDLeftJoinDistinctAndInsert(right.value.rowsRVD(), root, product = false)

  def count(): (Long, Long) = (countRows(), numCols)

  def countRows(): Long = partitionCounts().sum

  def forceCountRows(): Long = rvd.count()

  def deduplicate(): MatrixTable =
    copy2(rvd = rvd.boundary.mapPartitionsPreservesPartitioning(rvd.typ,
      SortedDistinctRowIterator.transformer(rvd.typ)))

  def deleteVA(args: String*): (Type, Deleter) = deleteVA(args.toList)

  def deleteVA(path: List[String]): (Type, Deleter) = rowType.delete(path)

  def dropCols(): MatrixTable =
    copyAST(ast = FilterColsIR(ast, ir.False()))

  def dropRows(): MatrixTable = copy2(rvd = OrderedRVD.empty(sparkContext, matrixType.orvdType))

  def explodeRows(root: String): MatrixTable = {
    val path = Parser.parseAnnotationRoot(root, Annotation.ROW_HEAD)
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

    val explodedRDD = rvd.boundary.mapPartitionsPreservesPartitioning(newMatrixType.orvdType, { (ctx, it) =>
      val region2 = ctx.region
      val rv2 = RegionValue(region2)
      val rv2b = ctx.rvb
      val ur = new UnsafeRow(oldRVType)
      it.flatMap { rv =>
        ur.set(rv)
        val keys = querier(ur).asInstanceOf[Iterable[Any]]
        if (keys == null)
          None
        else
          keys.iterator.map { explodedElement =>
            rv2b.start(newRVType)
            inserter(rv.region, rv.offset, rv2b,
              () => rv2b.addAnnotation(keyType, explodedElement))
            rv2.setOffset(rv2b.end())
            rv2
          }
      }
    })
    copyMT(matrixType = newMatrixType, rvd = explodedRDD)
  }

  def explodeCols(code: String): MatrixTable = {
    val path = Parser.parseAnnotationRoot(code, Annotation.COL_HEAD)
    val (keysType, querier) = colType.queryTyped(path)
    val keyType = keysType match {
      case TArray(e, _) => e
      case TSet(e, _) => e
      case t => fatal(s"Expected annotation of type Array or Set; found $t")
    }
    var size = 0
    val keys = colValues.value.map { sa =>
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
        newColValues(j) = inserter(colValues.value(i), e)
        j += 1
      }
      i += 1
    }

    val sampleMapBc = sparkContext.broadcast(sampleMap)
    val localEntriesIndex = entriesIndex
    val localEntriesType = matrixType.entryArrayType
    val fullRowType = rvRowType

    insertEntries(noOp, newColType = newColType,
      newColValues = colValues.copy(value = newColValues, t = TArray(newColType)))(entryType, { case (_, rv, rvb) =>

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

  def localizeEntries(entriesFieldName: String): Table = {
    val m = Map(MatrixType.entriesIdentifier -> entriesFieldName)
    val newRowType = rvRowType.rename(m)
    new Table(hc, TableLiteral(TableValue(
      TableType(newRowType, Some(rowKey), globalType),
      globals,
      rvd.copy(typ = rvd.typ.copy(rowType = newRowType)))))
  }

  def filterCols(p: (Annotation, Int) => Boolean): MatrixTable = {
    val (newType, filterF) = MatrixIR.filterCols(matrixType)
    copyAST(ast = MatrixLiteral(newType, filterF(value, p)))
  }

  def filterColsExpr(filterExpr: String, keep: Boolean = true): MatrixTable = {
    var filterAST = Parser.expr.parse(filterExpr)
    filterAST.typecheck(matrixType.colEC)
    var pred = filterAST.toIROpt(Some("AGG" -> "g"))
    pred match {
      case Some(irPred) if ContainsAgg(irPred) =>
        // FIXME: the IR path doesn't yet support aggs
        log.info("filterCols: predicate contains aggs - not yet supported in IR")
        pred = None
      case _ =>
    }
    pred match {
      case Some(irPred) =>
        new MatrixTable(hc,
          FilterColsIR(ast, ir.filterPredicateWithKeep(irPred, keep))
        )
      case None =>
        log.info(s"filterCols: No AST to IR conversion. Fallback for predicate ${ PrettyAST(filterAST) }")
        if (!keep)
          filterAST = Apply(filterAST.getPos, "!", Array(filterAST))
        copyAST(ast = FilterCols(ast, filterAST))
    }
  }

  def filterColsList(samples: java.util.ArrayList[Annotation], keep: Boolean): MatrixTable =
    filterColsList(samples.asScala.toSet, keep)

  def filterColsList(samples: Set[Annotation], keep: Boolean = true): MatrixTable = {
    val p = (s: Annotation, sa: Annotation) => Filter.keepThis(samples.contains(s), keep)
    filterCols(p)
  }

  def filterRowsExpr(filterExpr: String, keep: Boolean = true): MatrixTable = {
    var filterAST = Parser.expr.parse(filterExpr)
    filterAST.typecheck(matrixType.rowEC)
    var pred = filterAST.toIROpt(Some("AGG" -> "g"))
    pred match {
      case Some(irPred) if ContainsAgg(irPred) =>
        // FIXME: the IR path doesn't yet support aggs
        log.info("filterRows: predicate contains aggs - not yet supported in IR")
        pred = None
      case _ =>
    }
    pred match {
      case Some(irPred) =>
        new MatrixTable(hc,
          FilterRowsIR(ast, ir.filterPredicateWithKeep(irPred, keep)))
      case _ =>
        log.info(s"filterRows: No AST to IR conversion. Fallback for predicate ${ PrettyAST(filterAST) }")
        if (!keep)
          filterAST = Apply(filterAST.getPos, "!", Array(filterAST))
        copyAST(ast = FilterRows(ast, filterAST))
    }
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
    newColValues: BroadcastIndexedSeq = colValues,
    newGlobalType: TStruct = globalType,
    newGlobals: BroadcastRow = globals)(newEntryType: TStruct,
    inserter: (PC, RegionValue, RegionValueBuilder) => Unit): MatrixTable = {
    insertIntoRow(makePartitionContext, newColType, newColKey, newColValues, newGlobalType, newGlobals)(
      TArray(newEntryType), MatrixType.entriesIdentifier, inserter)
  }

  def insertIntoRow[PC](makePartitionContext: () => PC, newColType: TStruct = colType,
    newColKey: IndexedSeq[String] = colKey,
    newColValues: BroadcastIndexedSeq = colValues,
    newGlobalType: TStruct = globalType,
    newGlobals: BroadcastRow = globals)(typeToInsert: Type, path: String,
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

    val joiner = { (ctx: RVDContext, it: Iterator[JoinedRegionValue]) =>
      val rvb = ctx.rvb
      val rv2 = RegionValue()

      it.map { jrv =>
        val lrv = jrv.rvLeft
        val rrv = jrv.rvRight

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
        rv2.set(ctx.region, rvb.end())
        rv2
      }
    }

    val newMatrixType = matrixType.copyParts() // move entries to the end

    copyMT(matrixType = newMatrixType,
      colValues = colValues.copy(value = colValues.value ++ right.colValues.value),
      rvd = rvd.orderedJoinDistinct(right.rvd, "inner", joiner, rvd.typ))
  }

  def makeTable(separator: String = "."): Table = {
    requireColKeyString("make_table")
    requireUniqueSamples("make_table")

    val sampleIds = stringSampleIds

    val ttyp = TableType(
      TStruct(
        matrixType.rowType.fields.map { f => f.name -> f.typ } ++
          sampleIds.flatMap { s =>
            matrixType.entryType.fields.map { f =>
              val newName = if (f.name == "") s else s + separator + f.name
              newName -> f.typ
            }
          }: _*),
      Some(matrixType.rowKey),
      matrixType.globalType)

    val localNSamples = numCols
    val localRVRowType = rvRowType
    val localEntriesIndex = entriesIndex
    val localEntriesType = localRVRowType.types(entriesIndex).asInstanceOf[TArray]
    val localEntryType = matrixType.entryType

    new Table(hc,
      TableLiteral(
        TableValue(ttyp,
          globals,
          rvd.mapPartitions(ttyp.rowType) { it =>
            val rvb = new RegionValueBuilder()
            val rv2 = RegionValue()
            val fullRow = new UnsafeRow(localRVRowType)

            it.map { rv =>
              fullRow.set(rv)

              val region = rv.region

              rvb.set(region)
              rvb.start(ttyp.rowType)
              rvb.startStruct()

              var i = 0
              while (i < localRVRowType.size) {
                if (i != localEntriesIndex)
                  rvb.addField(localRVRowType, rv, i)
                i += 1
              }

              assert(localRVRowType.isFieldDefined(rv, localEntriesIndex))
              val entriesAOff = localRVRowType.loadField(rv, localEntriesIndex)

              i = 0
              while (i < localNSamples) {
                if (localEntriesType.isElementDefined(region, entriesAOff, i))
                  rvb.addAllFields(localEntryType, region, localEntriesType.loadElement(region, entriesAOff, i))
                else {
                  var j = 0
                  while (j < localEntryType.size) {
                    rvb.setMissing()
                    j += 1
                  }
                }

                i += 1
              }
              rvb.endStruct()

              rv2.set(region, rvb.end())
              rv2
            }
          })))
  }

  def aggregateRowsJSON(expr: String): String = {
    val (a, t) = aggregateRows(expr)
    val jv = JSONAnnotationImpex.exportAnnotation(a, t)
    JsonMethods.compact(jv)
  }

  def aggregateColsJSON(expr: String): String = {
    val (a, t) = aggregateCols(expr)
    val jv = JSONAnnotationImpex.exportAnnotation(a, t)
    JsonMethods.compact(jv)
  }

  def aggregateEntriesJSON(expr: String): String = {
    val (a, t) = aggregateEntries(expr)
    val jv = JSONAnnotationImpex.exportAnnotation(a, t)
    JsonMethods.compact(jv)
  }

  def aggregateEntries(expr: String): (Annotation, Type) = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "g" -> (1, entryType),
      "va" -> (2, rowType),
      "sa" -> (3, colType))

    val ec = EvalContext(Map(
      "global" -> (0, globalType),
      "AGG" -> (1, TAggregable(entryType, aggregationST))))

    val queryAST = Parser.parseToAST(expr, ec)

    queryAST.toIROpt(Some("AGG" -> "g")) match {
      case Some(qir) if useIR(entryAxis, queryAST) =>
        val et = entriesTable()

        val entriesRowType = et.typ.rowType
        val aggEnv = new ir.Env[ir.IR].bind(
          "g" -> ir.Ref("row", entriesRowType),
          "va" -> ir.Ref("row", entriesRowType),
          "sa" -> ir.Ref("row", entriesRowType))

        val sqir = ir.Subst(qir.unwrap, ir.Env.empty, aggEnv)
        et.aggregate(sqir)
      case _ =>
        val (t, f) = Parser.parseExpr(expr, ec)

        val localEntryType = entryType

        val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[Annotation](ec, { case (ec, g) =>
          ec.set(1, Annotation.copy(localEntryType, g))
        })

        val globalsBc = globals.broadcast
        val localColValuesBc = colValues.broadcast
        val localRVRowType = rvRowType
        val localRowType = rowType
        val localEntriesIndex = entriesIndex

        val result = rvd.mapPartitions { it =>
          val fullRow = new UnsafeRow(localRVRowType)
          val row = new UnsafeRow(localRowType)

          val zv = zVal.map(_.copy())
          ec.set(0, globalsBc.value)
          it.foreach { rv =>
            fullRow.set(rv)
            row.set(rv)
            val gs = fullRow.getAs[IndexedSeq[Any]](localEntriesIndex)

            var i = 0
            ec.set(2, Annotation.copy(row.t, row))
            gs.foreach { g =>
              ec.set(3, localColValuesBc.value(i))
              seqOp(zv, g)
              i += 1
            }
          }
          Iterator(zv)
        }.fold(zVal.map(_.copy()))(combOp)
        resOp(result)

        ec.set(0, globalsBc.value)
        (f(), t)
    }
  }

  def aggregateCols(expr: String): (Annotation, Type) = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "sa" -> (1, colType))

    val ec = EvalContext(Map(
      "global" -> (0, globalType),
      "AGG" -> (1, TAggregable(colType, aggregationST))))

    val queryAST = Parser.parseToAST(expr, ec)

    queryAST.toIROpt(Some("AGG" -> "g")) match {
      case Some(qir) if useIR(colAxis, queryAST) =>
        val ct = colsTable()
        val aggEnv = new ir.Env[ir.IR].bind("sa" -> ir.Ref("row", ct.typ.rowType))
        val sqir = ir.Subst(qir.unwrap, ir.Env.empty, aggEnv)
        ct.aggregate(sqir)
      case _ =>
        val (t, f) = Parser.parseExpr(expr, ec)

        val localGlobals = globals.value
        val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[Annotation](ec, { case (ec, (sa)) =>
          ec.setAll(localGlobals, sa)
        })

        val results = colValues.value
          .aggregate(zVal)(seqOp, combOp)
        resOp(results)
        ec.set(0, localGlobals)

        (f(), t)
    }
  }

  def aggregateRows(expr: String): (Annotation, Type) = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "va" -> (1, rvRowType))
    val ec = EvalContext(Map(
      "global" -> (0, globalType),
      "AGG" -> (1, TAggregable(rvRowType, aggregationST))))

    val qAST = Parser.parseToAST(expr, ec)

    qAST.toIROpt(Some("AGG" -> "g")) match {
      case Some(qir) if useIR(rowAxis, qAST) =>
        val rt = rowsTable()
        val aggEnv = new ir.Env[ir.IR].bind("va" -> ir.Ref("row", rt.typ.rowType))
        val sqir = ir.Subst(qir.unwrap, ir.Env.empty, aggEnv)
        rt.aggregate(sqir)
      case _ =>
        val globalsBc = globals.broadcast
        val (t, f) = Parser.parseExpr(expr, ec)

        val fullRowType = rvRowType
        val localEntriesIndex = entriesIndex
        val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[RegionValue](ec, { case (ec, rv) =>
          val fullRow = SafeRow(fullRowType, rv)
          ec.set(0, globalsBc.value)
          ec.set(1, fullRow)
        })

        val result = rvd
          .treeAggregate(zVal)(seqOp, combOp, depth = treeAggDepth(hc, nPartitions))
        resOp(result)

        (f(), t)
    }
  }

  def chooseCols(oldIndices: java.util.ArrayList[Int]): MatrixTable =
    chooseCols(oldIndices.asScala.toArray)

  def chooseCols(oldIndices: Array[Int]): MatrixTable = {
    require(oldIndices.forall { x => x >= 0 && x < numCols })
    copyAST(ast = ChooseCols(ast, oldIndices))
  }

  def renameFields(oldToNewRows: java.util.HashMap[String, String],
    oldToNewCols: java.util.HashMap[String, String],
    oldToNewEntries: java.util.HashMap[String, String],
    oldToNewGlobals: java.util.HashMap[String, String]): MatrixTable = {

    val fieldMapRows = oldToNewRows.asScala
    assert(fieldMapRows.keys.forall(k => matrixType.rowType.fieldNames.contains(k)),
      s"[${ fieldMapRows.keys.mkString(", ") }], expected [${ matrixType.rowType.fieldNames.mkString(", ") }]")

    val fieldMapCols = oldToNewCols.asScala
    assert(fieldMapCols.keys.forall(k => matrixType.colType.fieldNames.contains(k)),
      s"[${ fieldMapCols.keys.mkString(", ") }], expected [${ matrixType.colType.fieldNames.mkString(", ") }]")

    val fieldMapEntries = oldToNewEntries.asScala
    assert(fieldMapEntries.keys.forall(k => matrixType.entryType.fieldNames.contains(k)),
      s"[${ fieldMapEntries.keys.mkString(", ") }], expected [${ matrixType.entryType.fieldNames.mkString(", ") }]")

    val fieldMapGlobals = oldToNewGlobals.asScala
    assert(fieldMapGlobals.keys.forall(k => matrixType.globalType.fieldNames.contains(k)),
      s"[${ fieldMapGlobals.keys.mkString(", ") }], expected [${ matrixType.globalType.fieldNames.mkString(", ") }]")

    val (newColKey, newColType) = if (fieldMapCols.isEmpty) (colKey, colType) else {
      val newFieldNames = colType.fieldNames.map { n => fieldMapCols.getOrElse(n, n) }
      val newKey = colKey.map { f => fieldMapCols.getOrElse(f, f) }
      (newKey, TStruct(colType.required, newFieldNames.zip(colType.types): _*))
    }

    val newEntryType = if (fieldMapEntries.isEmpty) entryType else {
      val newFieldNames = entryType.fieldNames.map { n => fieldMapEntries.getOrElse(n, n) }
      TStruct(entryType.required, newFieldNames.zip(entryType.types): _*)
    }

    val (pk, newRowKey, newRVRowType) = {
      val newPK = rowPartitionKey.map { f => fieldMapRows.getOrElse(f, f) }
      val newKey = rowKey.map { f => fieldMapRows.getOrElse(f, f) }
      val newRVRowType = TStruct(rvRowType.required, rvRowType.fields.map { f =>
        f.name match {
          case x@MatrixType.entriesIdentifier => (x, TArray(newEntryType, f.typ.required))
          case x => (fieldMapRows.getOrElse(x, x), f.typ)
        }
      }: _*)
      (newPK, newKey, newRVRowType)
    }

    val newGlobalType = if (fieldMapGlobals.isEmpty) globalType else {
      val newFieldNames = globalType.fieldNames.map { n => fieldMapGlobals.getOrElse(n, n) }
      TStruct(globalType.required, newFieldNames.zip(globalType.types): _*)
    }

    val newMatrixType = MatrixType(newGlobalType,
      newColKey,
      newColType,
      pk,
      newRowKey,
      newRVRowType)

    val newRVD = if (newMatrixType.orvdType == rvd.typ)
      rvd
    else {
      val newType = newMatrixType.orvdType
      val newPartitioner = rvd.partitioner.withKType(pk.toArray, newType.kType)
      rvd.updateType(newType)
    }

    new MatrixTable(hc, newMatrixType, globals, colValues, newRVD)
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
    val newAnnotations = colValues.value.zipWithIndex.map { case (sa, i) => ins(sa, newIds(i)) }.toArray
    copy2(colType = newSchema, colValues = colValues.copy(value = newAnnotations, t = TArray(newSchema)))
  }

  def same(that: MatrixTable, tolerance: Double = utils.defaultTolerance, absolute: Boolean = false): Boolean = {
    var metadataSame = true
    if (rowType.deepOptional() != that.rowType.deepOptional()) {
      metadataSame = false
      println(
        s"""different row signature:
           |  left:  ${ rowType.toString }
           |  right: ${ that.rowType.toString }""".stripMargin)
    }
    if (colType.deepOptional() != that.colType.deepOptional()) {
      metadataSame = false
      println(
        s"""different column signature:
           |  left:  ${ colType.toString }
           |  right: ${ that.colType.toString }""".stripMargin)
    }
    if (globalType.deepOptional() != that.globalType.deepOptional()) {
      metadataSame = false
      println(
        s"""different global signature:
           |  left:  ${ globalType.toString }
           |  right: ${ that.globalType.toString }""".stripMargin)
    }
    if (entryType.deepOptional() != that.entryType.deepOptional()) {
      metadataSame = false
      println(
        s"""different entry signature:
           |  left:  ${ entryType.toString }
           |  right: ${ that.entryType.toString }""".stripMargin)
    }
    if (!colValuesSimilar(that, tolerance, absolute)) {
      metadataSame = false
      println(
        s"""different sample annotations:
           |  left:  $colValues
           |  right: ${ that.colValues }""".stripMargin)
    }
    if (!globalType.valuesSimilar(globals.value, that.globals.value, tolerance, absolute)) {
      metadataSame = false
      println(
        s"""different global annotation:
           |  left:  ${ globals.value }
           |  right: ${ that.globals.value }""".stripMargin)
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
      rvd.crdd.czip(
        OrderedRVD.adjustBoundsAndShuffle(
          that.rvd.typ,
          rvd.partitioner.withKType(that.rvd.typ.partitionKey, that.rvd.typ.kType),
          that.rvd)
          .crdd) { (ctx, rv1, rv2) =>
        var partSame = true

        val fullRow1 = new UnsafeRow(leftRVType)
        val fullRow2 = new UnsafeRow(rightRVType)

        fullRow1.set(rv1)
        fullRow2.set(rv2)
        val row1 = fullRow1.deleteField(localLeftEntriesIndex)
        val row2 = fullRow2.deleteField(localRightEntriesIndex)

        if (!localRowType.valuesSimilar(row1, row2, tolerance, absolute)) {
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
          if (!localEntryType.valuesSimilar(gs1(i), gs2(i), tolerance, absolute)) {
            partSame = false
            println(
              s"""different entry at row ${ localRKF(row1) }, col ${ localColKeys(i) }
                   |  ${ gs1(i) }
                   |  ${ gs2(i) }""".stripMargin)
          }
          i += 1
        }
        ctx.region.clear()
        partSame
      }.run.forall(t => t)
  }

  def colEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "sa" -> (1, colType),
      "g" -> (2, entryType),
      "va" -> (3, rvRowType))
    EvalContext(Map(
      "global" -> (0, globalType),
      "sa" -> (1, colType),
      "AGG" -> (2, TAggregable(entryType, aggregationST))))
  }

  def colValuesSimilar(that: MatrixTable, tolerance: Double = utils.defaultTolerance, absolute: Boolean = false): Boolean = {
    require(colType == that.colType, s"\n${ colType }\n${ that.colType }")
    colValues.value.zip(that.colValues.value)
      .forall { case (s1, s2) => colType.valuesSimilar(s1, s2, tolerance, absolute)
      }
  }

  def sampleRows(p: Double, seed: Int = 1): MatrixTable = {
    require(p > 0 && p < 1, s"the 'p' parameter must fall between 0 and 1, found $p")
    copyMT(rvd = rvd.sample(withReplacement = false, p, seed))
  }

  def copy2(rvd: OrderedRVD = rvd,
    colValues: BroadcastIndexedSeq = colValues,
    colKey: IndexedSeq[String] = colKey,
    globals: BroadcastRow = globals,
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
    globals: BroadcastRow = globals,
    colValues: BroadcastIndexedSeq = colValues): MatrixTable = {
    assert(rvd.typ == matrixType.orvdType,
      s"mismatch in orvdType:\n  rdd: ${ rvd.typ }\n  mat: ${ matrixType.orvdType }")
    new MatrixTable(hc,
      matrixType, globals, colValues, rvd)
  }

  def copyAST(ast: MatrixIR = ast): MatrixTable =
    new MatrixTable(hc, ast)

  def colsTable(): Table = new Table(hc, MatrixColsTable(ast))

  def storageLevel: String = rvd.storageLevel.toReadableString()

  def numCols: Int = colValues.value.length

  def typecheck() {
    var foundError = false
    if (!globalType.typeCheck(globals.value)) {
      foundError = true
      warn(
        s"""found violation in global annotation
           |Schema: $globalType
           |Annotation: ${ Annotation.printAnnotation(globals.value) }""".stripMargin)
    }

    colValues.value.zipWithIndex.find { case (sa, i) => !colType.typeCheck(sa) }
      .foreach { case (sa, i) =>
        foundError = true
        warn(
          s"""found violation in sample annotations for col $i
             |Schema: $colType
             |Annotation: ${ Annotation.printAnnotation(sa) }""".stripMargin)
      }

    val localRVRowType = rvRowType

    val predicate = { (rv: RegionValue) =>
      val ur = new UnsafeRow(localRVRowType, rv)
      !localRVRowType.typeCheck(ur)
    }

    Region.scoped { region =>
      rvd.find(region)(predicate).foreach { rv =>
        val ur = new UnsafeRow(localRVRowType, rv)
        foundError = true
        warn(
          s"""found violation in row
          |Schema: $localRVRowType
          |Annotation: ${ Annotation.printAnnotation(ur) }""".stripMargin)
      }
    }

    if (foundError)
      fatal("found one or more type check errors")
  }

  def entryEC: EvalContext = EvalContext(Map(
    "global" -> (0, globalType),
    "va" -> (1, rvRowType),
    "sa" -> (2, colType),
    "g" -> (3, entryType)))

  def rowEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalType),
      "va" -> (1, rvRowType),
      "g" -> (2, entryType),
      "sa" -> (3, colType))
    EvalContext(Map(
      "global" -> (0, globalType),
      "va" -> (1, rvRowType),
      "AGG" -> (2, TAggregable(entryType, aggregationST))))
  }

  def globalsTable(): Table = {
    Table(hc,
      sparkContext.parallelize[Row](Array(globals.value)),
      globalType,
      None)
  }

  def rowsTable(): Table = new Table(hc, MatrixRowsTable(ast))

  def entriesTable(): Table = new Table(hc, MatrixEntriesTable(ast))

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
    new MatrixTable(hc,
      MatrixMapEntries(ast,
        ir.If(
          ir.IsNA(ir.Ref("g", entryType)),
          ir.MakeStruct(
            entryType.fields.map(f => f.name -> (ir.NA(f.typ): ir.IR))),
          ir.Ref("g", entryType))))
  }

  def filterEntries(filterExpr: String, keep: Boolean = true): MatrixTable = {
    val symTab = Map(
      "va" -> (0, rvRowType),
      "sa" -> (1, colType),
      "g" -> (2, entryType),
      "global" -> (3, globalType))

    val ec = EvalContext(symTab)
    val filterAST = Parser.parseToAST(filterExpr, ec)
    filterAST.toIROpt() match {
      case Some(x) if useIR(entryAxis, filterAST) =>
        copyAST(MatrixFilterEntries(ast, ir.filterPredicateWithKeep(x, keep)))

      case _ =>
        val f: () => java.lang.Boolean = Parser.evalTypedExpr[java.lang.Boolean](filterAST, ec)

        val localKeep = keep
        val fullRowType = rvRowType
        val localNSamples = numCols
        val localEntryType = entryType
        val localColValuesBc = colValues.broadcast
        val localEntriesIndex = entriesIndex
        val localEntriesType = matrixType.entryArrayType
        val globalsBc = globals.broadcast

        insertEntries(() => {
          val fullRow = new UnsafeRow(fullRowType)
          fullRow
        })(localEntryType.copy(required = false), { case (fullRow, rv, rvb) =>
          fullRow.set(rv)
          val entries = fullRow.getAs[IndexedSeq[Annotation]](localEntriesIndex)
          val entriesOffset = fullRowType.loadField(rv, localEntriesIndex)

          rvb.startArray(localNSamples)

          var i = 0
          while (i < localNSamples) {
            val entry = entries(i)
            ec.setAll(fullRow,
              localColValuesBc.value(i),
              entry, globalsBc.value)
            if (Filter.boxedKeepThis(f(), localKeep)) {
              val isDefined = localEntriesType.isElementDefined(rv.region, entriesOffset, i)
              if (!isDefined)
                rvb.setMissing()
              else {
                // can't use addElement because we could be losing requiredness
                val elementOffset = localEntriesType.loadElement(rv.region, entriesOffset, i)
                rvb.startStruct()
                rvb.addAllFields(localEntryType, rv.region, elementOffset)
                rvb.endStruct()
              }
            } else
              rvb.setMissing()

            i += 1
          }
          rvb.endArray()
        })
    }
  }

  def write(path: String, overwrite: Boolean = false, codecSpecJSONStr: String = null) {
    ir.Interpret(ir.MatrixWrite(ast, _.write(path, overwrite, codecSpecJSONStr)))
  }

  def exportPlink(path: String) {
    ir.Interpret(ir.MatrixWrite(ast, (mv: MatrixValue) => ExportPlink(mv, path)))
  }

  def exportGen(path: String, precision: Int = 4) {
    ir.Interpret(ir.MatrixWrite(ast, (mv: MatrixValue) => ExportGen(mv, path, precision)))
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
      val kidAnnotation = colValues.value(kidIndex)

      var dadAnnotation: Annotation = null
      t.dad.foreach { dad =>
        val index = sampleIndices(dad)
        dadIndices(i) = index
        dadAnnotation = colValues.value(index)
      }

      var momAnnotation: Annotation = null
      t.mom.foreach { mom =>
        val index = sampleIndices(mom)
        momIndices(i) = index
        momAnnotation = colValues.value(index)
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

    insertEntries(noOp,
      newColType = newColType,
      newColKey = Array("id"),
      newColValues = colValues.copy(value = newColValues, t = TArray(newColType)))(newEntryType, { case (_, rv, rvb) =>
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

  def toRowMatrix(entryField: String): RowMatrix = {
    val partCounts = partitionCounts()
    val partStarts = partCounts.scanLeft(0L)(_ + _) // FIXME: use partitionStarts once partitionCounts is durable
    assert(partStarts.length == rvd.getNumPartitions + 1)
    val partStartsBc = sparkContext.broadcast(partStarts)

    val rvRowType = matrixType.rvRowType
    val entryArrayType = matrixType.entryArrayType
    val entryType = matrixType.entryType
    val fieldType = entryType.field(entryField).typ

    assert(fieldType.isOfType(TFloat64()))

    val entryArrayIdx = matrixType.entriesIdx
    val fieldIdx = entryType.fieldIdx(entryField)
    val numColsLocal = numCols

    val rows = rvd.mapPartitionsWithIndex { (pi, it) =>
      var i = partStartsBc.value(pi)
      it.map { rv =>
        val region = rv.region
        val data = new Array[Double](numColsLocal)
        val entryArrayOffset = rvRowType.loadField(rv, entryArrayIdx)
        var j = 0
        while (j < numColsLocal) {
          if (entryArrayType.isElementDefined(region, entryArrayOffset, j)) {
            val entryOffset = entryArrayType.loadElement(region, entryArrayOffset, j)
            if (entryType.isFieldDefined(region, entryOffset, fieldIdx)) {
              val fieldOffset = entryType.loadField(region, entryOffset, fieldIdx)
              data(j) = region.loadDouble(fieldOffset)
            } else
              fatal(s"Cannot create RowMatrix: missing value at row $i and col $j")
          } else
            fatal(s"Cannot create RowMatrix: missing entry at row $i and col $j")
          j += 1
        }
        val row = (i, data)
        i += 1
        row
      }
    }

    new RowMatrix(hc, rows, numCols, Some(partStarts.last), Some(partCounts))
  }

  def writeBlockMatrix(dirname: String, entryField: String, blockSize: Int = BlockMatrix.defaultBlockSize): Unit = {
    val partStarts = partitionStarts()
    assert(partStarts.length == rvd.getNumPartitions + 1)

    val nRows = partStarts.last
    val localNCols = numCols

    val hadoop = sparkContext.hadoopConfiguration
    hadoop.mkDir(dirname)

    // write blocks
    hadoop.mkDir(dirname + "/parts")
    val gp = GridPartitioner(blockSize, nRows, localNCols)
    val blockPartFiles =
      new WriteBlocksRDD(dirname, rvd.crdd, sparkContext, matrixType, partStarts, entryField, gp)
        .collect()

    val blockCount = blockPartFiles.length
    val partFiles = new Array[String](blockCount)
    blockPartFiles.foreach { case (i, f) => partFiles(i) = f }

    // write metadata
    hadoop.writeDataFile(dirname + BlockMatrix.metadataRelativePath) { os =>
      implicit val formats = defaultJSONFormats
      jackson.Serialization.write(
        BlockMatrixMetadata(blockSize, nRows, localNCols, gp.maybeSparse, partFiles),
        os)
    }

    assert(blockCount == gp.numPartitions)
    info(s"Wrote all $blockCount blocks of $nRows x $localNCols matrix with block size $blockSize.")

    hadoop.writeTextFile(dirname + "/_SUCCESS")(out => ())
  }

  def indexRows(name: String): MatrixTable = {
    val (newRVType, inserter) = rvRowType.unsafeStructInsert(TInt64(), List(name))

    val partStarts = partitionStarts()
    val newMatrixType = matrixType.copy(rvRowType = newRVType)
    val indexedRVD = rvd.boundary.mapPartitionsWithIndexPreservesPartitioning(newMatrixType.orvdType, { (i, ctx, it) =>
      val region2 = ctx.region
      val rv2 = RegionValue(region2)
      val rv2b = ctx.rvb

      var idx = partStarts(i)

      it.map { rv =>
        rv2b.start(newRVType)

        inserter(rv.region, rv.offset, rv2b,
          () => rv2b.addLong(idx))

        idx += 1
        rv2.setOffset(rv2b.end())
        rv2
      }
    })
    copyMT(matrixType = newMatrixType, rvd = indexedRVD)
  }

  def indexCols(name: String): MatrixTable = {
    val (newColType, inserter) = colType.structInsert(TInt32(), List(name))
    val newColValues = Array.tabulate(numCols) { i =>
      inserter(colValues.value(i), i)
    }
    copy2(colType = newColType,
      colValues = colValues.copy(value = newColValues, t = TArray(newColType)))
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

  def windowVariants(basePairs: Int): MatrixTable = {
    val newType = matrixType.copyParts(
      rowType = matrixType.rowType ++ TStruct("prev_rows" -> TArray(matrixType.rowType)),
      entryType = matrixType.entryType ++ TStruct("prev_entries" -> TArray(matrixType.entryType))
    )

    val partitioner = rvd.partitioner
    val adjustments = partitioner
      .rangeBounds
      .zipWithIndex
      .map { case (bound, i) =>
        val startLocus = bound.start.asInstanceOf[Row].getAs[Locus](0)
        val minPartitionNeeded = partitioner.getPartitionPK(
          Row(startLocus.copy(position = startLocus.position - basePairs)))
        (minPartitionNeeded to i)
          .map(Adjustment[RegionValue](_, identity))
          .toArray
      }.toArray

    val adjRDD = rvd.crdd.adjustPartitions(adjustments)

    val localRVRowType = rvRowType
    val locusIndex = localRVRowType.fieldIdx("locus")
    val entriesIndex = localRVRowType.fieldIdx(MatrixType.entriesIdentifier)
    val nonEntryIndices = (0 until localRVRowType.size).filter(_ != entriesIndex).toArray
    val entryArrayType = matrixType.entryArrayType
    val entryType = matrixType.entryType
    val rg = referenceGenome

    val locusStartsBc = sparkContext.broadcast(partitioner.rangeBounds.map { interval =>
      val locus = interval.start.asInstanceOf[Row].getAs[Locus](0)
      locus.copy(position = locus.position - basePairs)
    })

    val nCols = numCols
    val newRDD = adjRDD.cmapPartitionsAndContextWithIndex { (partitionIndex, context, it) =>
      val newContext = context.freshContext

      val rb = new mutable.ArrayStack[Region]()

      def fetchRegion(): Region = {
        if (rb.isEmpty)
          newContext.freshRegion
        else
          rb.pop()
      }
      def recycleRegion(r: Region): Unit = {
        r.clear()
        rb.push(r)
      }

      val deque = new java.util.ArrayDeque[(Locus, RegionValue)]()

      it.flatMap { f =>
        // these rows are loaded from previous partitions
        val inAdjustment = it.hasNext
        val rvIterator = f(newContext)

        val region = context.region
        val rv2 = RegionValue()
        val rvb = new RegionValueBuilder()
        rvIterator.flatMap { rv =>
          val locus = UnsafeRow.readLocus(rv.region, localRVRowType.loadField(rv, locusIndex), rg)

          val result = if (inAdjustment) {
            None
          } else {
            def discard(x: (Locus, RegionValue)): Boolean = x != null && (x._1.position < locus.position - basePairs
              || x._1.contig != locus.contig)

            while (discard(deque.peekLast()))
              recycleRegion(deque.removeLast()._2.region)

            val rvs = deque.iterator().asScala.map(_._2).toArray

            rvb.set(region)
            rvb.clear()
            region.clear()
            rvb.start(newType.rvRowType)
            rvb.startStruct()
            rvb.addFields(localRVRowType, rv, nonEntryIndices)

            // prev_rows
            rvb.startArray(rvs.length)
            var j = 0
            while (j < rvs.length) {
              val rvj = rvs(j)
              rvb.startStruct()
              rvb.addFields(localRVRowType, rvj, nonEntryIndices)
              rvb.endStruct()
              j += 1
            }
            rvb.endArray()

            rvb.startArray(nCols)

            val entriesOffset = localRVRowType.loadField(rv, entriesIndex)
            val prevEntriesOffsets = rvs.map(localRVRowType.loadField(_, entriesIndex))

            j = 0
            while (j < nCols) {
              rvb.startStruct()
              if (entryArrayType.isElementDefined(rv.region, entriesOffset, j))
                rvb.addAllFields(entryType, rv.region, entryArrayType.loadElement(rv.region, entriesOffset, j))
              else
                rvb.skipFields(entryType.size)

              // prev_entries
              rvb.startArray(rvs.length)
              var k = 0
              while (k < rvs.length) {
                rvb.startStruct()
                if (entryArrayType.isElementDefined(rvs(k).region, prevEntriesOffsets(k), j))
                  rvb.addAllFields(entryType, rvs(k).region, entryArrayType.loadElement(rvs(k).region, prevEntriesOffsets(k), j))
                else
                  rvb.skipFields(entryType.size)
                rvb.endStruct()
                k += 1
              }
              rvb.endArray()
              rvb.endStruct()

              j += 1
            }
            rvb.endArray()
            rvb.endStruct()

            rv2.set(region, rvb.end())
            Some(rv2)
          }

          if (!inAdjustment || rg.compare(locusStartsBc.value(partitionIndex), locus) <= 0) {
            val cpRegion = fetchRegion()
            rvb.set(cpRegion)
            rvb.clear()
            rvb.start(localRVRowType)
            rvb.startStruct()
            rvb.addAllFields(localRVRowType, rv)
            rvb.endStruct()
            deque.push(locus -> RegionValue(cpRegion, rvb.end()))
          }

          result
        }
      }
    }

    copyMT(rvd = new OrderedRVD(newType.orvdType, partitioner, newRDD), matrixType = newType)
  }
}
