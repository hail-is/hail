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

  def visitMissing(t: PType) {
    sb.append("NA")
  }

  def visitBoolean(b: Boolean) {
    sb.append(b)
  }

  def visitInt32(i: Int) {
    sb.append(i)
  }

  def visitInt64(l: Long) {
    sb.append(l)
  }

  def visitFloat32(f: Float) {
    sb.append(f)
  }

  def visitFloat64(d: Double) {
    sb.append(d)
  }

  def visitBinary(a: Array[Byte]) {
    sb.append("bytes...")
  }

  def visitString(s: String) {
    sb.append(s)
  }

  def enterStruct(t: PStruct) {
    sb.append("{")
  }

  def enterField(f: PField) {
    if (f.index > 0)
      sb.append(",")
    sb.append(" ")
    sb.append(f.name)
    sb.append(": ")
  }

  def leaveField() {}

  def leaveStruct() {
    sb.append(" }")
  }

  def enterTuple(t: PTuple) {
    sb.append('(')
  }

  def leaveTuple() {
    sb.append(')')
  }

  def enterArray(t: PContainer, length: Int) {
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

  def leaveArray() {
    sb.append("]")
  }

  def enterElement(i: Int) {
    if (i > 0)
      sb.append(",")
    sb.append(" ")
  }

  def leaveElement() {}
}
