package is.hail.annotations

import is.hail.macros.void
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

  @inline def visitMissing(t: PType): Unit =
    void(sb.append("NA"))

  @inline def visitBoolean(b: Boolean): Unit =
    void(sb.append(b))

  @inline def visitInt32(i: Int): Unit =
    void(sb.append(i))

  @inline def visitInt64(l: Long): Unit =
    void(sb.append(l))

  @inline def visitFloat32(f: Float): Unit =
    void(sb.append(f))

  @inline def visitFloat64(d: Double): Unit =
    void(sb.append(d))

  @inline def visitBinary(a: Array[Byte]): Unit =
    void(sb.append("bytes..."))

  @inline def visitString(s: String): Unit =
    void(sb.append(s))

  @inline def enterStruct(t: PStruct): Unit =
    void(sb.append("{"))

  def enterField(f: PField): Unit = {
    if (f.index > 0) sb += ','
    void(sb += ' ' ++= f.name ++= ": ")
  }

  @inline def leaveField(): Unit = {}

  @inline def leaveStruct(): Unit =
    void(sb.append(" }"))

  @inline def enterTuple(t: PTuple): Unit =
    void(sb.append('('))

  @inline def leaveTuple(): Unit =
    void(sb.append(')'))

  def enterArray(t: PContainer, length: Int): Unit = {
    t match {
      case _: PSet => sb.append("Set")
      case _: PDict => sb.append("Dict")
      case _ =>
    }

    void(sb += '[' ++= length.toString += ';')
  }

  @inline def leaveArray(): Unit =
    void(sb.append("]"))

  @inline def enterElement(i: Int): Unit = {
    if (i > 0) sb += ','
    void(sb += ' ')
  }

  @inline def leaveElement(): Unit = {}
}
