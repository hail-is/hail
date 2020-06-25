package is.hail.services.shuffler

import java.io._
import java.net.Socket

import is.hail.asm4s._

class SocketValue (
  val s: Value[Socket]
) extends AnyVal {
  def getInputStream(): Code[InputStream] =
    s.invoke[InputStream]("getInputStream")

  def getOutputStream(): Code[OutputStream] =
    s.invoke[OutputStream]("getOutputStream")

  def close(): Code[Unit] =
    s.invoke[Unit]("close")
}
