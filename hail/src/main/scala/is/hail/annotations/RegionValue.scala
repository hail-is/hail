package is.hail.annotations

import java.io._

import is.hail.expr.types.physical.PType
import is.hail.utils.using
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
  )(bytes: Array[Byte]
  ): RegionValue =
    using(new ByteArrayInputStream(bytes)) { bais =>
      using(makeDec(bais)) { dec =>
        carrierRv.setOffset(dec.readRegionValue(r))
        carrierRv
      }
    }

  def fromBytes(
    makeDec: InputStream => Decoder, r: Region, bytes: Array[Byte]): Long =
    using(new ByteArrayInputStream(bytes)) { bais =>
      using(makeDec(bais)) { dec =>
        dec.readRegionValue(r)
      }
    }

  def toBytes(makeEnc: OutputStream => Encoder, r: Region, off: Long): Array[Byte] =
    using(new ByteArrayOutputStream()) { baos =>
      using(makeEnc(baos)) { enc =>
        enc.writeRegionValue(r, off)
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

  def toBytes(makeEnc: OutputStream => Encoder): Array[Byte] =
    using(new ByteArrayOutputStream()) { baos =>
      using(makeEnc(baos)) { enc =>
        enc.writeRegionValue(region, offset)
        enc.flush()
        baos.toByteArray
      }
    }

  private def writeObject(s: ObjectOutputStream): Unit = {
    throw new NotImplementedException()
  }

  private def readObject(s: ObjectInputStream): Unit = {
    throw new NotImplementedException()
  }
}
