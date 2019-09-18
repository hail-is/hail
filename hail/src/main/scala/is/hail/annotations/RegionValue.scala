package is.hail.annotations

import java.io._

import is.hail.expr.types.physical.PType
import is.hail.utils.{using, RestartableByteArrayInputStream}
import is.hail.io.{Decoder, Encoder}
import sun.reflect.generics.reflectiveObjects.NotImplementedException

object RegionValue {
  def apply(): RegionValue = new RegionValue(null, 0)

  def apply(region: Region): RegionValue = new RegionValue(region, 0)

  def apply(region: Region, offset: Long) = new RegionValue(region, offset)

  def fromBytes(
    makeDec: InputStream => Decoder,
    r: Region,
    carrierRv: RegionValue
  )(bytes: Iterator[Array[Byte]]
  ): Iterator[RegionValue] = {
    val bais = new RestartableByteArrayInputStream()
    val dec = makeDec(bais)
    bytes.map { bytes =>
      bais.restart(bytes)
      carrierRv.setOffset(dec.readRegionValue(r))
      carrierRv
    }
  }

  def fromBytes(
    makeDec: InputStream => Decoder,
    r: Region,
    bytes: Iterator[Array[Byte]]
  ): Iterator[Long] = {
    val bais = new RestartableByteArrayInputStream(null)
    val dec = makeDec(bais)
    bytes.map { bytes =>
      bais.restart(bytes)
      dec.readRegionValue(r)
    }
  }

  def toBytes(makeEnc: OutputStream => Encoder, rvs: Iterator[RegionValue]): Iterator[Array[Byte]] = {
    val baos = new ByteArrayOutputStream()
    val enc = makeEnc(baos)
    rvs.map { rv =>
      baos.reset()
      enc.writeRegionValue(rv.region, rv.offset)
      enc.flush()
      baos.toByteArray
    }
  }
}

final class RegionValue(
  var region: Region,
  var offset: Long
) extends UnKryoSerializable {
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

  def pretty(t: PType): String = region.pretty(t, offset)

  private def writeObject(s: ObjectOutputStream): Unit = {
    throw new NotImplementedException()
  }

  private def readObject(s: ObjectInputStream): Unit = {
    throw new NotImplementedException()
  }
}
