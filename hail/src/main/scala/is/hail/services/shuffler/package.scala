package is.hail.services

import java.io._
import java.net.Socket
import java.security.KeyStore
import java.util.Base64

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.types.physical._
import is.hail.io._
import is.hail.utils._
import org.apache.log4j.Logger
import javax.net.ssl._;
import scala.language.implicitConversions

package object shuffler {
  val shuffleBufferSpec = BufferSpec.unblockedUncompressed

  def rvstr(pt: PType, off: Long): String =
    UnsafeRow.read(pt, null, off).toString

  def writeRegionValueArray(
    encoder: Encoder,
    values: Array[Long]
  ): Unit = {
    var i = 0
    while (i < values.length) {
      encoder.writeByte(1)
      encoder.writeRegionValue(values(i))
      i += 1
    }
    encoder.writeByte(0)
  }

  def readRegionValueArray(
    region: Region,
    decoder: Decoder,
    sizeHint: Int = ArrayBuilder.defaultInitialCapacity
  ): Array[Long] = {
    val ab = new ArrayBuilder[Long](sizeHint)

    var hasNext = decoder.readByte()
    while (hasNext == 1) {
      ab += decoder.readRegionValue(region)
      hasNext = decoder.readByte()
    }
    assert(hasNext == 0, hasNext)

    ab.result()
  }

  private[this] val b64encoder = Base64.getEncoder()

  def uuidToString(uuid: Array[Byte]): String =
    b64encoder.encodeToString(uuid)

  def uuidToString(uuid: Code[Array[Byte]]): Code[String] =
    Code.invokeScalaObject1[Array[Byte], String](getClass, "uuidToString", uuid)
}
