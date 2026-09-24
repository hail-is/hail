package is.hail.asm4s.implicits

import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.PrefixCoder

class RichCodePrefixCoder(val pc: Value[PrefixCoder]) {
  def toByteArray(cb: EmitCodeBuilder): Value[Array[Byte]] =
    cb.memoize(pc.invoke[Array[Byte]]("toByteArray"))

  def encodeBool(cb: EmitCodeBuilder, b: Code[Boolean]) =
    cb += pc.invoke[Boolean, Unit]("encodeBool", b)

  def encodeInt(cb: EmitCodeBuilder, v: Code[Int]) = cb += pc.invoke[Int, Unit]("encodeInt", v)

  def encodeLong(cb: EmitCodeBuilder, v: Code[Long]) = cb += pc.invoke[Long, Unit]("encodeLong", v)

  def encodeFloat(cb: EmitCodeBuilder, v: Code[Float]) =
    cb += pc.invoke[Float, Unit]("encodeFloat", v)

  def encodeDouble(cb: EmitCodeBuilder, v: Code[Double]) =
    cb += pc.invoke[Double, Unit]("encodeDouble", v)

  def encodeMissing(cb: EmitCodeBuilder) = cb += pc.invoke[Unit]("encodeMissing")
  def encodePresent(cb: EmitCodeBuilder) = cb += pc.invoke[Unit]("encodePresent")

  def encodeTerminator(cb: EmitCodeBuilder) = cb += pc.invoke[Unit]("encodeTerminator")
  def encodeContinuation(cb: EmitCodeBuilder) = cb += pc.invoke[Unit]("encodeContinuation")

  def writeBytes(cb: EmitCodeBuilder, bs: Code[Array[Byte]]) =
    cb += pc.invoke[Array[Byte], Unit]("writeBytes", bs)
}
