package is.hail.shuffler

import java.io._
import java.net.Socket

import is.hail.asm4s._

class SocketCode(
  s: Value[Socket]
) extends Value[Socket] {
  def get: Code[Socket] = s.get

  def getInputStream(): Code[InputStream] =
    s.invoke[InputStream]("getInputStream")

  def getOutputStream(): Code[OutputStream] =
    s.invoke[OutputStream]("getOutputStream")

  def close(): Code[Unit] =
    s.invoke[Unit]("close")
}
