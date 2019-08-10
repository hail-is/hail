package is.hail.nativecode

import is.hail.expr.types.physical._
import is.hail.io._
import is.hail.annotations.Region

final class PackEncoder(rowType: PType, out: OutputBuffer) extends Encoder {
  def flush() {
    out.flush()
  }

  def close() {
    out.close()
  }

  def writeByte(b: Byte): Unit = out.writeByte(b)

  def writeBinary(region: Region, offset: Long) {
    val boff = region.loadAddress(offset)
    val length = region.loadInt(boff)
    out.writeInt(length)
    out.writeBytes(region, boff + 4, length)
  }

  def writeArray(t: PArray, region: Region, aoff: Long) {
    val length = region.loadInt(aoff)

    out.writeInt(length)
    if (!t.elementType.required) {
      val nMissingBytes = (length + 7) >>> 3
      out.writeBytes(region, aoff + 4, nMissingBytes)
    }

    val elemsOff = aoff + t.elementsOffset(length)
    val elemSize = t.elementByteSize
    if (t.elementType.isInstanceOf[PInt32]) { // fast case
      var i = 0
      while (i < length) {
        if (t.isElementDefined(region, aoff, i)) {
          val off = elemsOff + i * elemSize
          out.writeInt(region.loadInt(off))
        }
        i += 1
      }
    } else {
      var i = 0
      while (i < length) {
        if (t.isElementDefined(region, aoff, i)) {
          val off = elemsOff + i * elemSize
          t.elementType match {
            case t2: PBaseStruct => writeBaseStruct(t2, region, off)
            case t2: PArray => writeArray(t2, region, region.loadAddress(off))
            case _: PBoolean => out.writeBoolean(region.loadByte(off) != 0)
            case _: PInt64 => out.writeLong(region.loadLong(off))
            case _: PFloat32 => out.writeFloat(region.loadFloat(off))
            case _: PFloat64 => out.writeDouble(region.loadDouble(off))
            case _: PBinary => writeBinary(region, off)
          }
        }

        i += 1
      }
    }
  }

  def writeBaseStruct(t: PBaseStruct, region: Region, offset: Long) {
    val nMissingBytes = t.nMissingBytes
    out.writeBytes(region, offset, nMissingBytes)

    var i = 0
    while (i < t.size) {
      if (t.isFieldDefined(region, offset, i)) {
        val off = offset + t.byteOffsets(i)
        t.types(i) match {
          case t2: PBaseStruct => writeBaseStruct(t2, region, off)
          case t2: PArray => writeArray(t2, region, region.loadAddress(off))
          case _: PBoolean => out.writeBoolean(region.loadByte(off) != 0)
          case _: PInt32 => out.writeInt(region.loadInt(off))
          case _: PInt64 => out.writeLong(region.loadLong(off))
          case _: PFloat32 => out.writeFloat(region.loadFloat(off))
          case _: PFloat64 => out.writeDouble(region.loadDouble(off))
          case _: PBinary => writeBinary(region, off)
        }
      }

      i += 1
    }
  }

  def writeRegionValue(region: Region, offset: Long) {
    (rowType.fundamentalType: @unchecked) match {
      case t: PBaseStruct =>
        writeBaseStruct(t, region, offset)
      case t: PArray =>
        writeArray(t, region, offset)
    }
  }

  def indexOffset(): Long = out.indexOffset()
}
