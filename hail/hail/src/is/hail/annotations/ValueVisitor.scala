package is.hail.annotations

import is.hail.types.physical._

trait ValueVisitor {
  def visitMissing(t: PType): Unit

  def visitBoolean(b: Boolean): Unit

  def visitInt32(i: Int): Unit

  def visitInt64(l: Long): Unit

  def visitFloat32(f: Float): Unit

  def visitFloat64(d: Double): Unit

  def visitString(s: String): Unit

  def visitBinary(b: Array[Byte]): Unit

  def enterStruct(t: PStruct): Unit

  def enterField(f: PField): Unit

  def leaveField(): Unit

  def leaveStruct(): Unit

  def enterArray(t: PContainer, length: Int): Unit

  def leaveArray(): Unit

  def enterElement(i: Int): Unit

  def leaveElement(): Unit

  def enterTuple(t: PTuple): Unit

  def leaveTuple(): Unit
}

final class PrettyVisitor extends ValueVisitor {
  val sb = new StringBuilder()

  @inline def result(): String = sb.result()

  @inline override def visitMissing(t: PType): Unit =
    sb.append("NA")

  @inline override def visitBoolean(b: Boolean): Unit =
    sb.append(b)

  @inline override def visitInt32(i: Int): Unit =
    sb.append(i)

  @inline override def visitInt64(l: Long): Unit =
    sb.append(l)

  @inline override def visitFloat32(f: Float): Unit =
    sb.append(f)

  @inline override def visitFloat64(d: Double): Unit =
    sb.append(d)

  @inline override def visitBinary(a: Array[Byte]): Unit =
    sb.append("bytes...")

  @inline override def visitString(s: String): Unit =
    sb.append(s)

  @inline override def enterStruct(t: PStruct): Unit =
    sb.append("{")

  override def enterField(f: PField): Unit = {
    if (f.index > 0) sb += ','
    sb += ' ' ++= f.name ++= ": ": Unit
  }

  @inline override def leaveField(): Unit = {}

  @inline override def leaveStruct(): Unit =
    sb.append(" }")

  @inline override def enterTuple(t: PTuple): Unit =
    sb.append('(')

  @inline override def leaveTuple(): Unit =
    sb.append(')')

  override def enterArray(t: PContainer, length: Int): Unit = {
    t match {
      case _: PSet => sb.append("Set")
      case _: PDict => sb.append("Dict")
      case _ =>
    }

    sb += '[' ++= length.toString += ';': Unit
  }

  @inline override def leaveArray(): Unit =
    sb.append("]")

  @inline override def enterElement(i: Int): Unit = {
    if (i > 0) sb += ','
    sb += ' '
  }

  @inline override def leaveElement(): Unit = {}
}
