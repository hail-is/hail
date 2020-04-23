package is.hail.shuffler

import java.util._

import is.hail.asm4s._

object Base64Code {
  def getEncoder(): Code[Base64.Encoder] =
    Code.invokeStatic0[Base64, Base64.Encoder]("getEncoder")
}

class Base64EncoderValue(
  code: Value[Base64.Encoder]
) extends Value[Base64.Encoder] {
  def get: Code[Base64.Encoder] = code.get

  def encodeToString(b: Code[Array[Byte]]): Code[String] =
    code.invoke[Array[Byte], String]("encodeToString", b)
}

class Base64EncoderCode(
  code: Code[Base64.Encoder]
) {
  def encodeToString(b: Code[Array[Byte]]): Code[String] =
    code.invoke[Array[Byte], String]("encodeToString", b)
}
