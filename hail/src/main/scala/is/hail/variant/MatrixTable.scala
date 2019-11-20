package is.hail.variant

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.expr.ir
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.expr.types.physical.{PArray, PStruct}
import is.hail.expr.types.virtual._
import is.hail.rvd._
import is.hail.sparkextras.ContextRDD
import is.hail.table.{AbstractTableSpec, Table, TableSpec}
import is.hail.utils._
import is.hail.{HailContext, utils}
import is.hail.io.fs.FS
import org.apache.hadoop
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.json4s._
import org.json4s.jackson.JsonMethods.parse
import org.json4s.jackson.Serialization

import scala.language.{existentials, implicitConversions}

abstract class ComponentSpec

object RelationalSpec {
  implicit val formats: Formats = new DefaultFormats() {
    override val typeHints = ShortTypeHints(List(
      classOf[ComponentSpec], classOf[RVDComponentSpec], classOf[PartitionCountsComponentSpec],
      classOf[RelationalSpec], classOf[MatrixTableSpec], classOf[TableSpec]))
    override val typeHintFieldName = "name"
  } +
    new TableTypeSerializer +
    new MatrixTypeSerializer

  def readMetadata(hc: HailContext, path: String): JValue = {
    if (!hc.sFS.isDir(path))
      fatal(s"MatrixTable and Table files are directories; path '$path' is not a directory")
    val metadataFile = path + "/metadata.json.gz"
    val jv = hc.sFS.readFile(metadataFile) { in => parse(in) }

    val fileVersion = jv \ "file_version" match {
      case JInt(rep) => SemanticVersion(rep.toInt)
      case _ =>
        fatal(
          s"""cannot read file: metadata does not contain file version: $metadataFile
             |  Common causes:
             |    - File is an 0.1 VariantDataset or KeyTable (0.1 and 0.2 native formats are not compatible!)""".stripMargin)
    }

    if (!FileFormat.version.supports(fileVersion))
      fatal(s"incompatible file format when reading: $path\n  supported version: ${ FileFormat.version }, found $fileVersion")
    jv
  }

  def read(hc: HailContext, path: String): RelationalSpec = {
    val jv = readMetadata(hc, path)
    val references = readReferences(hc, path, jv)

    references.foreach { rg =>
      if (!ReferenceGenome.hasReference(rg.name))
        ReferenceGenome.addReference(rg)
    }

    jv.extract[RelationalSpec]
  }

  def readReferences(hc: HailContext, path: String): Array[ReferenceGenome] =
    readReferences(hc, path, readMetadata(hc, path))

  def readReferences(hc: HailContext, path: String, jv: JValue): Array[ReferenceGenome] = {
    // FIXME this violates the abstraction of the serialization boundary
    val referencesRelPath = (jv \ "references_rel_path": @unchecked) match {
      case JString(p) => p
    }
    ReferenceGenome.readReferences(hc.sFS, path + "/" + referencesRelPath)
  }
}

abstract class RelationalSpec {
  def file_version: Int

  def hail_version: String

  def components: Map[String, ComponentSpec]

  def getComponent[T <: ComponentSpec](name: String): T = components(name).asInstanceOf[T]

  def globalsComponent: RVDComponentSpec = getComponent[RVDComponentSpec]("globals")

  def partitionCounts: Array[Long] = getComponent[PartitionCountsComponentSpec]("partition_counts").counts.toArray

  def write(fs: is.hail.io.fs.FS, path: String) {
    fs.writeTextFile(path + "/metadata.json.gz") { out =>
      Serialization.write(this, out)(RelationalSpec.formats)
    }
  }

  def indexed(path: String): Boolean

  def version: SemanticVersion = SemanticVersion(file_version)
}

case class RVDComponentSpec(rel_path: String) extends ComponentSpec {
  def absolutePath(path: String): String = path + "/" + rel_path

  def rvdSpec(fs: is.hail.io.fs.FS, path: String): AbstractRVDSpec =
    AbstractRVDSpec.read(fs, absolutePath(path))

  def indexed(hc: HailContext, path: String): Boolean = rvdSpec(hc.sFS, path).indexed

  def read(
    hc: HailContext,
    path: String,
    requestedType: TStruct,
    ctx: ExecuteContext,
    newPartitioner: Option[RVDPartitioner] = None,
    filterIntervals: Boolean = false
  ): RVD = {
    val rvdPath = path + "/" + rel_path
    rvdSpec(hc.sFS, path)
      .read(hc, rvdPath, requestedType, ctx, newPartitioner, filterIntervals)
  }

  def readLocalSingleRow(hc: HailContext, path: String, requestedType: TStruct, r: Region): (PStruct, Long) = {
    val rvdPath = path + "/" + rel_path
    rvdSpec(hc.sFS, path)
      .readLocalSingleRow(hc, rvdPath, requestedType, r)
  }
}

case class PartitionCountsComponentSpec(counts: Seq[Long]) extends ComponentSpec

abstract class AbstractMatrixTableSpec extends RelationalSpec {
  def matrix_type: MatrixType

  def references_rel_path: String

  def colsComponent: RVDComponentSpec = getComponent[RVDComponentSpec]("cols")

  def rowsComponent: RVDComponentSpec = getComponent[RVDComponentSpec]("rows")

  def entriesComponent: RVDComponentSpec = getComponent[RVDComponentSpec]("entries")

  def indexed(path: String): Boolean = rowsComponent.indexed(HailContext.get, path)

  def rowsTableSpec(path: String): AbstractTableSpec = RelationalSpec.read(HailContext.get, path).asInstanceOf[AbstractTableSpec]
  def colsTableSpec(path: String): AbstractTableSpec = RelationalSpec.read(HailContext.get, path).asInstanceOf[AbstractTableSpec]
  def entriesTableSpec(path: String): AbstractTableSpec = RelationalSpec.read(HailContext.get, path).asInstanceOf[AbstractTableSpec]
}

case class MatrixTableSpec(
  file_version: Int,
  hail_version: String,
  references_rel_path: String,
  matrix_type: MatrixType,
  components: Map[String, ComponentSpec]) extends AbstractMatrixTableSpec {

  // some legacy files written as MatrixTableSpec wrote the wrong type to the entries table metadata
  override def entriesTableSpec(path: String): AbstractTableSpec = {
    val writtenETS = super.entriesTableSpec(path).asInstanceOf[TableSpec]
    writtenETS.copy(table_type = TableType(matrix_type.entriesRVType, FastIndexedSeq(), matrix_type.globalType))
  }
}

object FileFormat {
  val version: SemanticVersion = SemanticVersion(1, 3, 0)
}

object MatrixTable {
  def read(hc: HailContext, path: String, dropCols: Boolean = false, dropRows: Boolean = false): MatrixTable =
    new MatrixTable(hc, MatrixIR.read(hc, path, dropCols, dropRows, None))

  def range(hc: HailContext, nRows: Int, nCols: Int, nPartitions: Option[Int]): MatrixTable =
    if (nRows == 0) {
      new MatrixTable(hc, MatrixIR.range(hc, nRows, nCols, nPartitions, dropRows=true))
    } else
      new MatrixTable(hc, MatrixIR.range(hc, nRows, nCols, nPartitions))
}

class MatrixTable(val hc: HailContext, val ast: MatrixIR) {
  val matrixType: MatrixType = ast.typ

  val colType: TStruct = matrixType.colType
  val rowType: TStruct = matrixType.rowType
  val entryType: TStruct = matrixType.entryType
  val globalType: TStruct = matrixType.globalType

  val rowKey: IndexedSeq[String] = matrixType.rowKey

  val colKey: IndexedSeq[String] = matrixType.colKey

  def colKeyTypes: Array[Type] = colKey
    .map(s => matrixType.colType.types(matrixType.colType.fieldIdx(s)))
    .toArray

  val rowKeyTypes: Array[Type] = rowKey
    .map(s => matrixType.rowType.types(matrixType.rowType.fieldIdx(s)))
    .toArray

  val rowKeyStruct: TStruct = TStruct(rowKey.zip(rowKeyTypes): _*)

  lazy val (globals, colValues, rvRowPType, entriesIdx, rvd, lit) = {
    ExecuteContext.scoped { ctx =>
      val tv = Interpret(ast, ctx, optimize = true)
      val mv = tv.toMatrixValue(matrixType.colKey)
      (mv.globals.safeJavaValue, mv.colValues.safeJavaValue, mv.rvRowPType, mv.entriesIdx, mv.rvd, MatrixLiteral(matrixType, TableLiteral(tv, ctx)))
    }
  }

  def colKeys: IndexedSeq[Annotation] = {
    val queriers = colKey.map(colType.query(_))
    colValues.map(a => Row.fromSeq(queriers.map(q => q(a)))).toArray[Annotation]
  }

  def rowKeysF: (Row) => Row = {
    val localRowType = rowType
    val queriers = rowKey.map(localRowType.query(_)).toArray
    (r: Row) => Row.fromSeq(queriers.map(_ (r)))
  }

  def countRows(): Long = ExecuteContext.scoped { ctx => Interpret[Long](ctx, TableCount(MatrixRowsTable(ast))) }

  def same(
    that: MatrixTable,
    ctx: ExecuteContext,
    tolerance: Double = utils.defaultTolerance,
    absolute: Boolean = false
  ): Boolean = {
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
    if (!globalType.valuesSimilar(globals, that.globals, tolerance, absolute)) {
      metadataSame = false
      println(
        s"""different global annotation:
           |  left:  ${ globals }
           |  right: ${ that.globals }""".stripMargin)
    }
    if (rowKey != that.rowKey || colKey != that.colKey) {
      metadataSame = false
      println(
        s"""
           |different keys:
           |  left:  rk $rowKey, ck $colKey
           |  right: rk ${ that.rowKey }, ck ${ that.colKey }""".stripMargin)
    }
    if (!metadataSame)
      println("metadata were not the same")

    val leftRVType = rvRowPType
    val rightRVType = that.rvRowPType
    val localRowType = rowType
    val localLeftEntriesIndex = entriesIdx
    val localRightEntriesIndex = that.entriesIdx
    val localEntryType = entryType
    val localRKF = rowKeysF
    val localColKeys = colKeys

    val (_, jcrdd) = this.rvd.orderedZipJoin(that.rvd, ctx)

    metadataSame &&
      jcrdd.mapPartitions { it =>
        val fullRow1 = new UnsafeRow(leftRVType)
        val fullRow2 = new UnsafeRow(rightRVType)

        it.map { case Muple(rv1, rv2) =>
          if (rv2 == null) {
            fullRow1.set(rv1)
            val row1 = fullRow1.deleteField(localRightEntriesIndex)
            println(s"row ${ localRKF(row1) } present in left but not right")
            false
          } else if (rv1 == null) {
            fullRow2.set(rv2)
            val row2 = fullRow2.deleteField(localRightEntriesIndex)
            println(s"row ${ localRKF(row2) } present in right but not left")
            false
          }
          else {
            var partSame = true

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
            partSame
          }
        }
      }.clearingRun.forall(t => t)
  }

  def colValuesSimilar(that: MatrixTable, tolerance: Double = utils.defaultTolerance, absolute: Boolean = false): Boolean = {
    require(colType == that.colType, s"\n${ colType }\n${ that.colType }")
    colValues.zip(that.colValues)
      .forall { case (s1, s2) => colType.valuesSimilar(s1, s2, tolerance, absolute)
      }
  }

  def copyAST(ast: MatrixIR = ast): MatrixTable =
    new MatrixTable(hc, ast)

  def numCols: Int = colValues.length

  def write(path: String, overwrite: Boolean = false, stageLocally: Boolean = false, codecSpecJSONStr: String = null) {
    ExecuteContext.scoped { ctx =>
      ir.Interpret[Unit](ctx, ir.MatrixWrite(ast, MatrixNativeWriter(path, overwrite, stageLocally, codecSpecJSONStr, null)))
    }
  }
}
