package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.{PCanonicalString, PType}
import is.hail.types.physical.stypes.{SSettable, SType, SValue}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.primitives.SInt64Value
import is.hail.types.virtual.{TString, Type}
import is.hail.utils.FastSeq

case object SJavaString extends SString {
  override val virtualType: TString.type = TString

  override def storageType(): PType = PCanonicalString(false)

  override def copiedType: SType = this

  override def containsPointers: Boolean = false

  override def castRename(t: Type): SType = this

  override def _coerceOrCopy(
    cb: EmitCodeBuilder,
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): SJavaStringValue =
    value.st match {
      case SJavaString => value.asInstanceOf[SJavaStringValue]
      case _ => new SJavaStringValue(value.asString.loadString(cb))
    }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(classInfo[String])

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SJavaStringSettable = {
    val IndexedSeq(s: Settable[String @unchecked]) = settables
    new SJavaStringSettable(s)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SJavaStringValue = {
    val IndexedSeq(s: Value[String @unchecked]) = values
    new SJavaStringValue(s)
  }

  override def constructFromString(cb: EmitCodeBuilder, r: Value[Region], s: Code[String])
    : SJavaStringValue =
    new SJavaStringValue(cb.memoize(s))

  def construct(cb: EmitCodeBuilder, s: Code[String]): SJavaStringValue =
    new SJavaStringValue(cb.memoize(s))

  override def isIsomorphicTo(st: SType): Boolean =
    this == st
}

class SJavaStringValue(val s: Value[String]) extends SStringValue {
  override def st: SString = SJavaString

  override lazy val valueTuple: IndexedSeq[Value[_]] = FastSeq(s)

  override def loadLength(cb: EmitCodeBuilder): Value[Int] =
    cb.memoize(s.invoke[Int]("length"))

  override def loadString(cb: EmitCodeBuilder): Value[String] = s

  override def toBytes(cb: EmitCodeBuilder): SBinaryValue =
    new SJavaBytesValue(cb.memoize(s.invoke[Array[Byte]]("getBytes")))

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value = {
    val lengthInBytes: Value[Int] = cb.memoize(s.invoke[Array[Byte]]("getBytes").length())
    val storageTypeString = this.st.storageType().asInstanceOf[PCanonicalString]
    val contentsSize = storageTypeString.binaryRepresentation.contentByteSize(lengthInBytes)
    new SInt64Value(cb.memoize(contentsSize))
  }
}

object SJavaStringSettable {
  def apply(sb: SettableBuilder, name: String): SJavaStringSettable =
    new SJavaStringSettable(sb.newSettable[String](s"${name}_str"))
}

final class SJavaStringSettable(override val s: Settable[String])
    extends SJavaStringValue(s) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(s)

  override def store(cb: EmitCodeBuilder, v: SValue): Unit =
    cb.assign(s, v.asInstanceOf[SJavaStringValue].s)
}
