package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s.Code
import is.hail.expr.ir.{EmitMethodBuilder, SortOrder}
import is.hail.expr.types.virtual.{Field, TStruct, Type}
import is.hail.utils._
import org.apache.spark.sql.Row

object PStruct {
  def empty(required: Boolean = false): PStruct = PCanonicalStruct.empty(required)

  def apply(required: Boolean, args: (String, PType)*): PStruct = PCanonicalStruct(required, args:_*)

  def apply(args: IndexedSeq[PField], required: Boolean = false): PStruct =
    PCanonicalStruct(args, required)

  def apply(args: (String, PType)*): PStruct =
    apply(false, args: _*)

  def canonical(t: Type): PStruct = PType.canonical(t).asInstanceOf[PStruct]
  def canonical(t: PType): PStruct = PType.canonical(t).asInstanceOf[PStruct]
}

abstract class PStruct extends PBaseStruct {
  lazy val virtualType: TStruct = TStruct(fields.map(f => Field(f.name, f.typ.virtualType, f.index)), required)

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering =
    codeOrdering(mb, other, null)

  def codeOrdering(mb: EmitMethodBuilder, other: PType, so: Array[SortOrder]): CodeOrdering = {
    assert(other isOfType this)
    assert(so == null || so.size == types.size)
    CodeOrdering.rowOrdering(this, other.asInstanceOf[PStruct], mb, so)
  }

  def identBase: String = "tuple"

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("struct{")
    fields.foreachBetween({ field =>
      sb.append(prettyIdentifier(field.name))
      sb.append(": ")
      field.typ.pyString(sb)
    }) { sb.append(", ")}
    sb.append('}')
  }

  def copy(fields: IndexedSeq[PField] = this.fields, required: Boolean = this.required): PStruct

  protected def structFundamentalType: PStruct
  override def fundamentalType: PStruct = structFundamentalType

  def unsafeStructInsert(typeToInsert: PType, path: List[String]): (PStruct, UnsafeInserter)

  def deleteField(key: String): PStruct

  def appendKey(key: String, sig: PType): PStruct

  def rename(m: Map[String, String]): PStruct

  def ++(that: PStruct): PStruct

  def selectFields(names: Seq[String]): PStruct

  def select(keep: IndexedSeq[String]): (PStruct, (Row) => Row)

  def dropFields(names: Set[String]): PStruct

  def typeAfterSelect(keep: IndexedSeq[Int]): PStruct

  def loadField(region: Code[Region], offset: Code[Long], fieldName: String): Code[Long]

  def loadField(offset: Code[Long], field: String): Code[Long]

  def isFieldDefined(offset: Code[Long], field: String): Code[Boolean]

  def isFieldMissing(offset: Code[Long], field: String): Code[Boolean]

  def fieldOffset(offset: Code[Long], fieldName: String): Code[Long]

  def setFieldPresent(offset: Code[Long], field: String): Code[Unit]

  def setFieldMissing(offset: Code[Long], field: String): Code[Unit]

  def insertFields(fieldsToInsert: TraversableOnce[(String, PType)]): PStruct
}
