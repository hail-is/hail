package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s.Code
import is.hail.expr.ir.{EmitMethodBuilder, SortOrder}
import is.hail.expr.types.virtual.{Field, TStruct}
import org.apache.spark.sql.Row

trait PStruct extends PBaseStruct {
  lazy val virtualType: TStruct = TStruct(fields.map(f => Field(f.name, f.typ.virtualType, f.index)))

  final def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering =
    codeOrdering(mb, other, null)

  final def codeOrdering(mb: EmitMethodBuilder[_], other: PType, so: Array[SortOrder]): CodeOrdering = {
    assert(other isOfType this)
    assert(so == null || so.size == types.size)
    CodeOrdering.rowOrdering(this, other.asInstanceOf[PStruct], mb, so)
  }

  def deleteField(key: String): PStruct

  def appendKey(key: String, sig: PType): PStruct

  def rename(m: Map[String, String]): PStruct

  def ++(that: PStruct): PStruct

  def identBase: String = "tuple"

  def selectFields(names: Seq[String]): PStruct

  def dropFields(names: Set[String]): PStruct

  def typeAfterSelect(keep: IndexedSeq[Int]): PStruct

  protected val structFundamentalType: PStruct
  override lazy val fundamentalType: PStruct = structFundamentalType

  def loadField(offset: Code[Long], fieldName: String): Code[Long]

  final def isFieldDefined(offset: Code[Long], fieldName: String): Code[Boolean] = !isFieldMissing(offset, fieldName)

  def isFieldMissing(offset: Code[Long], fieldName: String): Code[Boolean]

  def fieldOffset(offset: Code[Long], fieldName: String): Code[Long]

  def setFieldPresent(offset: Code[Long], fieldName: String): Code[Unit]

  def setFieldMissing(offset: Code[Long], fieldName: String): Code[Unit]

  def insertFields(fieldsToInsert: TraversableOnce[(String, PType)]): PStruct
}
