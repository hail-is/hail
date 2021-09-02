package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.interfaces.{SBinary, SBinaryCode, SBinaryValue}
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SValue}
import is.hail.types.physical.{PCanonicalBinary, PType}
import is.hail.types.virtual._
import is.hail.utils.FastIndexedSeq

case object SJavaBytes extends SBinary {
  override val virtualType: TBinary.type = TBinary

  override def storageType(): PType = PCanonicalBinary(false)

  override def copiedType: SType = this

  override def castRename(t: Type): SType = this

  override def containsPointers: Boolean = false

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SJavaBytesValue =
    value.st match {
      case SJavaBytes => value.asInstanceOf[SJavaBytesValue]
      case _ => new SJavaBytesValue(cb.memoize(value.asBinaryValue.loadBytes()))
    }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(arrayInfo[Byte])

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SJavaBytesSettable = {
    val IndexedSeq(b: Settable[Array[Byte]@unchecked]) = settables
    new SJavaBytesSettable(b)
  }

  override def fromValues(values: IndexedSeq[Value[_]]): SJavaBytesValue = {
    val IndexedSeq(b: Value[Array[Byte]@unchecked]) = values
    new SJavaBytesValue(b)
  }
}

class SJavaBytesCode(val bytes: Code[Array[Byte]]) extends SBinaryCode {
  def st: SBinary = SJavaBytes

  def loadLength(): Code[Int] = bytes.invoke[Int]("length")

  private[this] def memoizeWithBuilder(cb: EmitCodeBuilder, name: String, sb: SettableBuilder): SBinaryValue = {
    val s = new SJavaBytesSettable(sb.newSettable[Array[Byte]](s"${name}_javabytearray"))
    s.store(cb, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String): SBinaryValue = memoizeWithBuilder(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String): SBinaryValue = memoizeWithBuilder(cb, name, cb.fieldBuilder)

  override def loadBytes(): Code[Array[Byte]] = bytes
}

class SJavaBytesValue(val bytes: Value[Array[Byte]]) extends SBinaryValue {
  override def st: SBinary = SJavaBytes

  override lazy val valueTuple: IndexedSeq[Value[_]] = FastIndexedSeq(bytes)

  override def get: SJavaBytesCode = new SJavaBytesCode(bytes)

  override def loadLength(): Code[Int] = bytes.length()

  override def loadByte(i: Code[Int]): Code[Byte] = bytes(i)

  override def loadBytes(): Code[Array[Byte]] = bytes
}

object SJavaBytesSettable {
  def apply(sb: SettableBuilder, name: String): SJavaBytesSettable = {
    new SJavaBytesSettable(sb.newSettable[Array[Byte]](s"${ name }_bytes"))
  }
}

final class SJavaBytesSettable(override val bytes: Settable[Array[Byte]]) extends SJavaBytesValue(bytes) with SSettable {
  override def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(bytes)

  override def store(cb: EmitCodeBuilder, v: SCode): Unit = {
    cb.assign(bytes, v.asInstanceOf[SJavaBytesCode].bytes)
  }
}
