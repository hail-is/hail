package is.hail.types.physical

import is.hail.annotations.{Annotation, Region, UnsafeUtils}
import is.hail.asm4s.{Code, Settable, SettableBuilder, Value, coerce, const}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, IEmitCode}
import is.hail.types.BaseStruct
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SBaseStructCode}
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.physical.stypes.concrete.SSubsetStruct
import is.hail.types.virtual.TStruct
import is.hail.utils._

object PSubsetStruct {
  def apply(ps: PStruct, fieldNames: String*): PSubsetStruct = {
    val f = fieldNames.toArray
    PSubsetStruct(ps, f)
  }
}

// Semantics: PSubsetStruct is a non-constructible view of another PStruct, which is not allowed to mutate
// that underlying PStruct's region data
final case class PSubsetStruct(ps: PStruct, _fieldNames: IndexedSeq[String]) extends PStruct {
  val fields: IndexedSeq[PField] = _fieldNames.zipWithIndex.map { case (name, i) => PField(name, ps.fieldType(name), i)}
  val required = ps.required

  if (fields == ps.fields) {
    log.warn("PSubsetStruct used without subsetting input PStruct")
  }

  private val idxMap: Array[Int] = _fieldNames.map(f => ps.fieldIdx(f)).toArray

  lazy val missingIdx: Array[Int] = idxMap.map(i => ps.missingIdx(i))
  lazy val nMissing: Int = missingIdx.length

  override lazy val virtualType = TStruct(fields.map(f => (f.name -> f.typ.virtualType)):_*)
  override val types: Array[PType] = fields.map(_.typ).toArray

  override val byteSize: Long = 8

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    sb.append("PSubsetStruct{")
    ps.pretty(sb, indent, compact)
    sb += '{'
    fieldNames.foreachBetween(f => sb.append(prettyIdentifier(f)))(sb += ',')
    sb += '}'
    sb += '}'
  }

  override def rename(m: Map[String, String]): PStruct = {
    val newNames = fieldNames.map(fieldName => m.getOrElse(fieldName, fieldName))
    val newPStruct = ps.rename(m)

    PSubsetStruct(newPStruct, newNames)
  }

  override def isFieldMissing(structAddress: Code[Long], fieldName: String): Code[Boolean] =
    ps.isFieldMissing(structAddress, fieldName)

  override def fieldOffset(structAddress: Code[Long], fieldName: String): Code[Long] =
    ps.fieldOffset(structAddress, fieldName)

  override def isFieldDefined(structAddress: Long, fieldIdx: Int): Boolean =
    ps.isFieldDefined(structAddress, idxMap(fieldIdx))

  override def isFieldMissing(structAddress: Code[Long], fieldIdx: Int): Code[Boolean] =
    ps.isFieldMissing(structAddress, idxMap(fieldIdx))

  override def fieldOffset(structAddress: Long, fieldIdx: Int): Long =
    ps.fieldOffset(structAddress, idxMap(fieldIdx))

  override def fieldOffset(structAddress: Code[Long], fieldIdx: Int): Code[Long] =
    ps.fieldOffset(structAddress, idxMap(fieldIdx))

  def loadField(structAddress: Code[Long], fieldName: String): Code[Long] =
    ps.loadField(structAddress, fieldName)

  override def loadField(structAddress: Long, fieldIdx: Int): Long =
    ps.loadField(structAddress, idxMap(fieldIdx))

  override def loadField(structAddress: Code[Long], fieldIdx: Int): Code[Long] =
    ps.loadField(structAddress, idxMap(fieldIdx))

  override def setFieldPresent(structAddress: Code[Long], fieldName: String): Code[Unit] = ???

  override def setFieldMissing(structAddress: Code[Long], fieldName: String): Code[Unit] = ???

  override def setFieldMissing(structAddress: Long, fieldIdx: Int): Unit = ???

  override def setFieldMissing(structAddress: Code[Long], fieldIdx: Int): Code[Unit] = ???

  override def setFieldPresent(structAddress: Long, fieldIdx: Int): Unit = ???

  override def setFieldPresent(structAddress: Code[Long], fieldIdx: Int): Code[Unit] = ???

  def insertFields(fieldsToInsert: TraversableOnce[(String, PType)]): PSubsetStruct = ???

  override def initialize(structAddress: Long, setMissing: Boolean): Unit =
    ps.initialize(structAddress, setMissing)

  override def stagedInitialize(structAddress: Code[Long], setMissing: Boolean): Code[Unit] =
    ps.stagedInitialize(structAddress, setMissing)

  def allocate(region: Region): Long =
    ps.allocate(region)

  def allocate(region: Code[Region]): Code[Long] =
    ps.allocate(region)

  override def setRequired(required: Boolean): PType =
    PSubsetStruct(ps.setRequired(required).asInstanceOf[PStruct], _fieldNames)

  override def copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long =
    throw new UnsupportedOperationException

  override def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long =
    throw new UnsupportedOperationException

  def sType: SSubsetStruct = SSubsetStruct(ps.sType.asInstanceOf[SBaseStruct], _fieldNames)

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): Code[Long] = throw new UnsupportedOperationException

  def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit = {
    throw new UnsupportedOperationException
  }

  def loadCheapPCode(cb: EmitCodeBuilder, addr: Code[Long]): SBaseStructCode = throw new UnsupportedOperationException

  def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    throw new UnsupportedOperationException
  }

  def loadFromNested(addr: Code[Long]): Code[Long] = addr

  override def unstagedLoadFromNested(addr: Long): Long = addr

  override def unstagedStoreJavaObject(annotation: Annotation, region: Region): Long =
    throw new UnsupportedOperationException

  override def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation, region: Region): Unit =
    throw new UnsupportedOperationException
}
