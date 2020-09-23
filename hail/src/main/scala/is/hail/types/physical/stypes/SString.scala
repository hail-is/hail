package is.hail.types.physical.stypes

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{???, Code, LongInfo, Settable, TypeInfo, Value, typeInfo}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.mtypes.{MCode, MString, MType, MValue}

trait SString extends SType {

}

trait SStringCode extends SCode {
  def typ: SString

  def stringCode(cb: EmitCodeBuilder): Code[String] = memoize(cb).asString.stringValue(cb)
}

trait SStringValue extends SValue {
  def typ: SString

  def stringValue(cb: EmitCodeBuilder): Value[String]

  def length(): Code[Int]
}

case object SJavaString extends SString {
  override def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = ???

  override def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = IndexedSeq(typeInfo[String])

  override def loadFrom(cb: EmitCodeBuilder, region: Value[Region], mv: MValue): SCode = {
    val ms = mv.typ.asInstanceOf[MString]
    ???
  }

  override def coerceOrCopySValue(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deep: Boolean): SCode = {
    value.typ match {
      case SJavaString => value
      case _ => value
    }
  }
}

case class SJavaStringCode(s: Code[String]) extends SStringCode {
  def typ: SString = SJavaString

  override def stringCode(cb: EmitCodeBuilder): Code[String] = s

  override def memoize(cb: EmitCodeBuilder): SValue = {
    val ss = cb.newLocal[String]("java_string_memo")
    cb.assign(ss, s)
    SJavaStringValue(ss)
  }
}

case class SJavaStringValue(s: Value[String]) extends SStringValue {
  def typ: SString = SJavaString

  def stringValue(cb: EmitCodeBuilder): Value[String] = s

  def length(): Code[Int] = s.length()
}

case class SStringPointer(mType: MType) extends SString with SPointer {
  override def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = ???

  override def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = IndexedSeq(LongInfo)

  override def loadFrom(cb: EmitCodeBuilder, region: Value[Region], mv: MValue): SCode = SStringPointerCode(mv)
}

case class SStringPointerCode(typ: SStringPointer, value: MCode) extends SStringCode {
  override def memoize(cb: EmitCodeBuilder): SValue = {
    SStringPointerValue(typ, value.memoize(cb))
  }
}

case class SStringPointerValue(typ: SStringPointer, value: MValue) extends SStringValue {
  override def length(): Code[Int] = ???

  override def stringValue(cb: EmitCodeBuilder): Value[String] = Code.newInstance[String, Array[Byte]](Region.loadBytes(value.addr, .loadBytes(bAddress))

}