package is.hail.types.physical.stypes.concrete

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SValue}
import is.hail.types.physical.{PCanonicalBinary, PCanonicalString, PString, PType}
import is.hail.types.physical.stypes.interfaces.{SBinary, SBinaryCode, SBinaryValue, SString, SStringValue}
import is.hail.types.virtual._
import is.hail.utils.FastIndexedSeq

case object SJavaBytes extends SBinary {
  val virtualType: TBinary.type = TBinary

  override def canonicalPType(): PType = PCanonicalBinary(false)

  override def castRename(t: Type): SType = this

  def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SJavaBytesCode = {
    value.st match {
      case SJavaBytes => value.asInstanceOf[SJavaBytesCode]
      case _ => new SJavaBytesCode(value.asBinary.loadBytes())
    }
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(arrayInfo[Byte])

  def fromSettables(settables: IndexedSeq[Settable[_]]): SJavaBytesSettable = {
    val IndexedSeq(b: Settable[Array[Byte]@unchecked]) = settables
    new SJavaBytesSettable(b)
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): SJavaBytesCode = {
    val IndexedSeq(b: Settable[Array[Byte]@unchecked]) = codes
    new SJavaBytesCode(b)
  }
}


class SJavaBytesCode(val bytes: Code[Array[Byte]]) extends SBinaryCode {
  def st: SBinary = SJavaBytes

  def makeCodeTuple(cb: EmitCodeBuilder): IndexedSeq[Code[_]] = FastIndexedSeq(bytes)

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

object SJavaBytesSettable {
  def apply(sb: SettableBuilder, name: String): SJavaBytesSettable = {
    new SJavaBytesSettable(sb.newSettable[Array[Byte]](s"${ name }_bytes"))
  }
}

class SJavaBytesSettable(val bytes: Settable[Array[Byte]]) extends SBinaryValue with SSettable {
  def st: SBinary = SJavaBytes

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(bytes)

  def get: SJavaBytesCode = new SJavaBytesCode(bytes.load())

  def store(cb: EmitCodeBuilder, v: SCode): Unit = {
    cb.assign(bytes, v.asInstanceOf[SJavaBytesCode].bytes)
  }

  override def loadLength(): Code[Int] = bytes.length()

  override def loadByte(i: Code[Int]): Code[Byte] = bytes(i)

  override def loadBytes(): Code[Array[Byte]] = bytes
}
