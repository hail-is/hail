package is.hail.annotations

import java.io._
import is.hail.asm4s.HailClassLoader
import is.hail.backend.ExecuteContext
import is.hail.types.physical.PType
import is.hail.utils.{RestartableByteArrayInputStream, using}
import is.hail.io._
import sun.reflect.generics.reflectiveObjects.NotImplementedException

object RegionValue {
  def apply(): RegionValue = new RegionValue(null, 0)

  def apply(region: Region): RegionValue = new RegionValue(region, 0)

  def apply(region: Region, offset: Long) = new RegionValue(region, offset)

  def fromBytes(
    theHailClassLoader: HailClassLoader,
    makeDec: (InputStream, HailClassLoader) => Decoder,
    r: Region,
    byteses: Iterator[Array[Byte]]
  ): Iterator[Long] = {
    val bad = new ByteArrayDecoder(theHailClassLoader, makeDec)
    byteses.map(bad.regionValueFromBytes(r, _))
  }

  def pointerFromBytes(
    theHailClassLoader: HailClassLoader,
    makeDec: (InputStream, HailClassLoader) => Decoder,
    r: Region,
    byteses: Iterator[Array[Byte]]
  ): Iterator[Long] = {
    val bad = new ByteArrayDecoder(theHailClassLoader, makeDec)
    byteses.map { bytes =>
      bad.regionValueFromBytes(r, bytes)
    }
  }

  def toBytes(
    ctx: ExecuteContext,
    makeEnc: (OutputStream, ExecuteContext) => Encoder,
    rvs: Iterator[Long]
  ): Iterator[Array[Byte]] = {
    val bae = new ByteArrayEncoder(ctx, makeEnc)
    rvs.map(bae.regionValueToBytes)
  }
}

final class RegionValue(
  var region: Region,
  var offset: Long
) extends UnKryoSerializable {
  def getOffset: Long = offset

  def set(newRegion: Region, newOffset: Long) {
    region = newRegion
    offset = newOffset
  }

  def setRegion(newRegion: Region) {
    region = newRegion
  }

  def setOffset(newOffset: Long) {
    offset = newOffset
  }

  def pretty(t: PType): String = Region.pretty(t, offset)

  private def writeObject(s: ObjectOutputStream): Unit = {
    throw new NotImplementedException()
  }

  private def readObject(s: ObjectInputStream): Unit = {
    throw new NotImplementedException()
  }
}
