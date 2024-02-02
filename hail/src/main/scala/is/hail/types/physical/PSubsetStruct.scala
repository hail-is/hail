package is.hail.types.physical

import is.hail.annotations.{Annotation, Region}
import is.hail.asm4s.{Code, Value}
import is.hail.backend.HailStateManager
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.SSubsetStruct
import is.hail.types.physical.stypes.interfaces.SBaseStructValue
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
  val fields: IndexedSeq[PField] = _fieldNames.zipWithIndex.map { case (name, i) =>
    PField(name, ps.fieldType(name), i)
  }

  val required = ps.required

  if (fields == ps.fields) {
    log.warn("PSubsetStruct used without subsetting input PStruct")
  }

  private val idxMap: Array[Int] = _fieldNames.map(f => ps.fieldIdx(f)).toArray

  lazy val missingIdx: Array[Int] = idxMap.map(i => ps.missingIdx(i))
  lazy val nMissing: Int = missingIdx.length

  override lazy val virtualType = TStruct(fields.map(f => (f.name -> f.typ.virtualType)): _*)
  override val types: Array[PType] = fields.map(_.typ).toArray

  override val byteSize: Long = 8

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = {
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

  override def isFieldMissing(cb: EmitCodeBuilder, structAddress: Code[Long], fieldName: String)
    : Value[Boolean] =
    ps.isFieldMissing(cb, structAddress, fieldName)

  override def fieldOffset(structAddress: Code[Long], fieldName: String): Code[Long] =
    ps.fieldOffset(structAddress, fieldName)

  override def isFieldDefined(structAddress: Long, fieldIdx: Int): Boolean =
    ps.isFieldDefined(structAddress, idxMap(fieldIdx))

  override def isFieldMissing(cb: EmitCodeBuilder, structAddress: Code[Long], fieldIdx: Int)
    : Value[Boolean] =
    ps.isFieldMissing(cb, structAddress, idxMap(fieldIdx))

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

  override def setFieldPresent(cb: EmitCodeBuilder, structAddress: Code[Long], fieldName: String)
    : Unit = ???

  override def setFieldMissing(cb: EmitCodeBuilder, structAddress: Code[Long], fieldName: String)
    : Unit = ???

  override def setFieldMissing(structAddress: Long, fieldIdx: Int): Unit = ???

  override def setFieldMissing(cb: EmitCodeBuilder, structAddress: Code[Long], fieldIdx: Int)
    : Unit = ???

  override def setFieldPresent(structAddress: Long, fieldIdx: Int): Unit = ???

  override def setFieldPresent(cb: EmitCodeBuilder, structAddress: Code[Long], fieldIdx: Int)
    : Unit = ???

  def insertFields(fieldsToInsert: TraversableOnce[(String, PType)]): PSubsetStruct = ???

  override def initialize(structAddress: Long, setMissing: Boolean): Unit =
    ps.initialize(structAddress, setMissing)

  override def stagedInitialize(cb: EmitCodeBuilder, structAddress: Code[Long], setMissing: Boolean)
    : Unit =
    ps.stagedInitialize(cb, structAddress, setMissing)

  def allocate(region: Region): Long =
    ps.allocate(region)

  def allocate(region: Code[Region]): Code[Long] =
    ps.allocate(region)

  override def setRequired(required: Boolean): PType =
    PSubsetStruct(ps.setRequired(required).asInstanceOf[PStruct], _fieldNames)

  override def copyFromAddress(
    sm: HailStateManager,
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Long =
    throw new UnsupportedOperationException

  override def _copyFromAddress(
    sm: HailStateManager,
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Long =
    throw new UnsupportedOperationException

  def sType: SSubsetStruct =
    new SSubsetStruct(ps.sType, _fieldNames)

  def store(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean)
    : Value[Long] =
    throw new UnsupportedOperationException

  def storeAtAddress(
    cb: EmitCodeBuilder,
    addr: Code[Long],
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): Unit =
    throw new UnsupportedOperationException

  def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SBaseStructValue =
    throw new UnsupportedOperationException

  def unstagedStoreAtAddress(
    sm: HailStateManager,
    addr: Long,
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Unit =
    throw new UnsupportedOperationException

  def loadFromNested(addr: Code[Long]): Code[Long] = addr

  override def unstagedLoadFromNested(addr: Long): Long = addr

  override def unstagedStoreJavaObject(sm: HailStateManager, annotation: Annotation, region: Region)
    : Long =
    throw new UnsupportedOperationException

  override def unstagedStoreJavaObjectAtAddress(
    sm: HailStateManager,
    addr: Long,
    annotation: Annotation,
    region: Region,
  ): Unit =
    throw new UnsupportedOperationException

  override def copiedType: PType = ??? // PSubsetStruct on its way out
}
