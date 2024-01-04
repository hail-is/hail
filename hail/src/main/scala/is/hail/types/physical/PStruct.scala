package is.hail.types.physical

import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SBaseStructValue}
import is.hail.types.virtual.{Field, TStruct}

trait PStruct extends PBaseStruct {
  override lazy val virtualType: TStruct =
    TStruct(fields.map(f => Field(f.name, f.typ.virtualType, f.index)))

  override def sType: SBaseStruct

  final def deleteField(key: String): PCanonicalStruct = {
    assert(fieldIdx.contains(key))
    val index = fieldIdx(key)
    val newFields = Array.fill[PField](fields.length - 1)(null)
    for (i <- 0 until index)
      newFields(i) = fields(i)
    for (i <- index + 1 until fields.length)
      newFields(i - 1) = fields(i).copy(index = i - 1)
    PCanonicalStruct(newFields, required)
  }

  final def appendKey(key: String, sig: PType): PCanonicalStruct = {
    assert(!fieldIdx.contains(key))
    val newFields = Array.fill[PField](fields.length + 1)(null)
    for (i <- fields.indices)
      newFields(i) = fields(i)
    newFields(fields.length) = PField(key, sig, fields.length)
    PCanonicalStruct(newFields, required)
  }

  def rename(m: Map[String, String]): PStruct

  def identBase: String = "struct"

  final def selectFields(names: Seq[String]): PCanonicalStruct =
    PCanonicalStruct(required, names.map(f => f -> field(f).typ): _*)

  final def dropFields(names: Set[String]): PCanonicalStruct =
    selectFields(fieldNames.filter(!names.contains(_)))

  final def typeAfterSelect(keep: IndexedSeq[Int]): PCanonicalStruct =
    PCanonicalStruct(required, keep.map(i => fieldNames(i) -> types(i)): _*)

  def loadField(offset: Code[Long], fieldName: String): Code[Long]

  final def isFieldDefined(cb: EmitCodeBuilder, offset: Code[Long], fieldName: String)
    : Value[Boolean] =
    cb.memoize(!isFieldMissing(cb, offset, fieldName))

  def isFieldMissing(cb: EmitCodeBuilder, offset: Code[Long], fieldName: String): Value[Boolean]

  def fieldOffset(offset: Code[Long], fieldName: String): Code[Long]

  def setFieldPresent(cb: EmitCodeBuilder, offset: Code[Long], fieldName: String): Unit

  def setFieldMissing(cb: EmitCodeBuilder, offset: Code[Long], fieldName: String): Unit

  def insertFields(fieldsToInsert: TraversableOnce[(String, PType)]): PStruct

  def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SBaseStructValue
}
