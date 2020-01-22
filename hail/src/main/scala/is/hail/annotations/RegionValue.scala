package is.hail.annotations

import java.io._

import is.hail.expr.types.physical.PType
import is.hail.utils.{using, RestartableByteArrayInputStream}
import is.hail.io._
import sun.reflect.generics.reflectiveObjects.NotImplementedException

object RegionValue {
  def apply(): RegionValue = new RegionValue(null, 0)

  def apply(region: Region): RegionValue = new RegionValue(region, 0)

  def apply(region: Region, offset: Long) = new RegionValue(region, offset)

  def fromBytes(
    makeDec: InputStream => Decoder,
    r: Region,
    byteses: Iterator[Array[Byte]]
  ): Iterator[RegionValue] = {
    val rv = RegionValue(r)
    val bad = new ByteArrayDecoder(makeDec)
    byteses.map { bytes =>
      rv.setOffset(bad.regionValueFromBytes(r, bytes))
      rv
    }
  }

  def pointerFromBytes(
    makeDec: InputStream => Decoder,
    r: Region,
    byteses: Iterator[Array[Byte]]
  ): Iterator[Long] = {
    val bad = new ByteArrayDecoder(makeDec)
    byteses.map { bytes =>
      bad.regionValueFromBytes(r, bytes)
    }
  }

  def toBytes(makeEnc: OutputStream => Encoder, rvs: Iterator[RegionValue]): Iterator[Array[Byte]] = {
    val bae = new ByteArrayEncoder(makeEnc)
    rvs.map(rv => bae.regionValueToBytes(rv.region, rv.offset))
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
