package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.interfaces.{SBinary, SBinaryValue}
import is.hail.types.physical.stypes.{SSettable, SType, SValue}
import is.hail.types.physical.{PCanonicalBinary, PType}
import is.hail.types.virtual._
import is.hail.utils.FastSeq

case object SJavaBytes extends SBinary {
  override val virtualType: TBinary.type = TBinary

  override def storageType(): PType = PCanonicalBinary(false)

  override def copiedType: SType = this

  override def castRename(t: Type): SType = this

  override def containsPointers: Boolean = false

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SJavaBytesValue =
    value.st match {
      case SJavaBytes => value.asInstanceOf[SJavaBytesValue]
      case _ => new SJavaBytesValue(value.asBinary.loadBytes(cb))
    }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(arrayInfo[Byte])

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SJavaBytesSettable = {
    val IndexedSeq(b: Settable[Array[Byte]@unchecked]) = settables
    new SJavaBytesSettable(b)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SJavaBytesValue = {
    val IndexedSeq(b: Value[Array[Byte]@unchecked]) = values
    new SJavaBytesValue(b)
  }
}

class SJavaBytesValue(val bytes: Value[Array[Byte]]) extends SBinaryValue {
  override def st: SBinary = SJavaBytes

  override lazy val valueTuple: IndexedSeq[Value[_]] = FastSeq(bytes)

  override def loadLength(cb: EmitCodeBuilder): Value[Int] =
    cb.memoize(bytes.length())

  override def loadByte(cb: EmitCodeBuilder, i: Code[Int]): Value[Byte] =
    cb.memoize(bytes(i))

  override def loadBytes(cb: EmitCodeBuilder): Value[Array[Byte]] = bytes
}

object SJavaBytesSettable {
  def apply(sb: SettableBuilder, name: String): SJavaBytesSettable = {
    new SJavaBytesSettable(sb.newSettable[Array[Byte]](s"${ name }_bytes"))
  }
}

final class SJavaBytesSettable(override val bytes: Settable[Array[Byte]]) extends SJavaBytesValue(bytes) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastSeq(bytes)

  override def store(cb: EmitCodeBuilder, v: SValue): Unit = {
    cb.assign(bytes, v.asInstanceOf[SJavaBytesValue].bytes)
  }
}
