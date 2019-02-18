package is.hail.variant

import is.hail.expr.types._
import is.hail.expr.types.physical.PStruct
import is.hail.expr.types.virtual._
import is.hail.rvd._
import is.hail.table.{AbstractTableSpec, TableSpec}
import is.hail.utils._
import is.hail.{HailContext, cxx}
import org.apache.spark.sql.Row
import org.json4s._
import org.json4s.jackson.{JsonMethods, Serialization}

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

  def read(hc: HailContext, path: String): RelationalSpec = {
    if (!hc.hadoopConf.isDir(path))
      fatal(s"MatrixTable and Table files are directories; path '$path' is not a directory")
    val metadataFile = path + "/metadata.json.gz"
    val jv = hc.hadoopConf.readFile(metadataFile) { in => JsonMethods.parse(in) }

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

    // FIXME this violates the abstraction of the serialization boundary
    val referencesRelPath = (jv \ "references_rel_path": @unchecked) match {
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

  def partitionCounts: Array[Long] = getComponent[PartitionCountsComponentSpec]("partition_counts").counts.toArray

  def write(hc: HailContext, path: String) {
    hc.hadoopConf.writeTextFile(path + "/metadata.json.gz") { out =>
      Serialization.write(this, out)(RelationalSpec.formats)
    }
  }

  def rvdType(path: String): RVDType
}

case class RVDComponentSpec(rel_path: String) extends ComponentSpec {
  def read(hc: HailContext, path: String, requestedType: PStruct): RVD = {
    val rvdPath = path + "/" + rel_path
    AbstractRVDSpec.read(hc, rvdPath)
      .read(hc, rvdPath, requestedType)
  }

  def readLocal(hc: HailContext, path: String, requestedType: PStruct): IndexedSeq[Row] = {
    val rvdPath = path + "/" + rel_path
    AbstractRVDSpec.read(hc, rvdPath)
      .readLocal(hc, rvdPath, requestedType)
  }

  def cxxEmitRead(hc: HailContext, path: String, requestedType: TStruct, tub: cxx.TranslationUnitBuilder): cxx.RVDEmitTriplet = {
    val rvdPath = path + "/" + rel_path
    AbstractRVDSpec.read(hc, rvdPath)
      .cxxEmitRead(hc, rvdPath, requestedType, tub)
  }
}

case class PartitionCountsComponentSpec(counts: Seq[Long]) extends ComponentSpec

abstract class AbstractMatrixTableSpec extends RelationalSpec {
  def matrix_type: MatrixType

  def references_rel_path: String

  def colsComponent: RVDComponentSpec = getComponent[RVDComponentSpec]("cols")

  def rowsComponent: RVDComponentSpec = getComponent[RVDComponentSpec]("rows")

  def entriesComponent: RVDComponentSpec = getComponent[RVDComponentSpec]("entries")

  def rvdType(path: String): RVDType = {
    val rows = AbstractRVDSpec.read(HailContext.get, path + "/" + rowsComponent.rel_path)
    val entries = AbstractRVDSpec.read(HailContext.get, path + "/" + entriesComponent.rel_path)
    RVDType(rows.encodedType.appendKey(MatrixType.entriesIdentifier, entries.encodedType), rows.key)
  }

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
  val version: SemanticVersion = SemanticVersion(1, 0, 0)
}
