package is.hail.types.physical

import is.hail.asm4s.{Code, Value}
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.interfaces.SBaseStruct
import is.hail.types.virtual.{Field, TStruct}

import scala.collection.compat._

trait PStruct extends PBaseStruct {
  override lazy val virtualType: TStruct =
    TStruct(fields.map(f => Field(f.name, f.typ.virtualType, f.index)))

  override def sType: SBaseStruct

  final def deleteField(key: String): PCanonicalStruct = {
    assert(fieldIdx.contains(key))
    val index = fieldIdx(key)
    val newFields = ArraySeq.newBuilder[PField]
    newFields.sizeHint(fields.length - 1)
    for (i <- 0 until index)
      newFields += fields(i)
    for (i <- index + 1 until fields.length)
      newFields += fields(i).copy(index = i - 1)
    PCanonicalStruct(newFields.result(), required)
  }

  final def appendKey(key: String, sig: PType): PCanonicalStruct = {
    assert(!fieldIdx.contains(key))
    val newFields = fields :+ PField(key, sig, fields.length)
    PCanonicalStruct(newFields, required)
  }

  def rename(m: Map[String, String]): PStruct

  override def identBase: String = "struct"

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

  def insertFields(fieldsToInsert: IterableOnce[(String, PType)]): PStruct
}
