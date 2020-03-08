package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.types._
import is.hail.expr.types.physical.PStruct
import is.hail.expr.types.virtual._
import is.hail.rvd._
import is.hail.utils._
import is.hail.variant.ReferenceGenome
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
  val version: SemanticVersion = SemanticVersion(1, 4, 0)
}
