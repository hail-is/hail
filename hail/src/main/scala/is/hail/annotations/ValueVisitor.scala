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

  def result(): String = sb.result()

  def visitMissing(t: PType): Unit = {
    sb.append("NA")
  }

  def visitBoolean(b: Boolean): Unit = {
    sb.append(b)
  }

  def visitInt32(i: Int): Unit = {
    sb.append(i)
  }

  def visitInt64(l: Long): Unit = {
    sb.append(l)
  }

  def visitFloat32(f: Float): Unit = {
    sb.append(f)
  }

  def visitFloat64(d: Double): Unit = {
    sb.append(d)
  }

  def visitBinary(a: Array[Byte]): Unit = {
    sb.append("bytes...")
  }

  def visitString(s: String): Unit = {
    sb.append(s)
  }

  def enterStruct(t: PStruct): Unit = {
    sb.append("{")
  }

  def enterField(f: PField): Unit = {
    if (f.index > 0)
      sb.append(",")
    sb.append(" ")
    sb.append(f.name)
    sb.append(": ")
  }

  def leaveField(): Unit = {}

  def leaveStruct(): Unit = {
    sb.append(" }")
  }

  def enterTuple(t: PTuple): Unit = {
    sb.append('(')
  }

  def leaveTuple(): Unit = {
    sb.append(')')
  }

  def enterArray(t: PContainer, length: Int): Unit = {
    t match {
      case _: PSet =>
        sb.append("Set")
      case _: PDict =>
        sb.append("Dict")
      case _ =>
    }
    sb.append("[")
    sb.append(length)
    sb.append(";")
  }

  def leaveArray(): Unit = {
    sb.append("]")
  }

  def enterElement(i: Int): Unit = {
    if (i > 0)
      sb.append(",")
    sb.append(" ")
  }

  def leaveElement(): Unit = {}
}
