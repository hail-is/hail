package is.hail.annotations

import is.hail.expr._
import is.hail.utils.Interval
import is.hail.variant.{AltAllele, GenericGenotype, Locus, Variant}
import org.apache.spark.sql.Row

class UnsafeIndexedSeqAnnotation(val region: MemoryBuffer,
  arrayTTBc: BroadcastTypeTree,
  elemSize: Int, val offset: Int, elemOffset: Int,
  val length: Int) extends IndexedSeq[Annotation] {
  def apply(i: Int): Annotation = {
    if (i < 0 || i >= length)
      throw new IndexOutOfBoundsException(i.toString)
    assert(i >= 0 && i < length)
    if (region.loadBit(offset + 4, i))
      null
    else {
      UnsafeRow.read(region, elemOffset + i * elemSize,
        arrayTTBc.value.typ.asInstanceOf[TContainer].elementType,
        arrayTTBc.value.subtree(0))
    }
  }
}

object UnsafeRow {
  def readBinary(region: MemoryBuffer, offset: Int): Array[Byte] = {
    val start = region.loadInt(offset)
    assert(offset > 0 && (offset & 0x3) == 0, s"invalid binary start: $offset")
    val binLength = region.loadInt(start)
    region.loadBytes(start + 4, binLength)
  }

  def readArray(region: MemoryBuffer, offset: Int, elemType: Type, arrayTTBc: BroadcastTypeTree): IndexedSeq[Any] = {
    val aoff = region.loadInt(offset)

    val length = region.loadInt(aoff)
    val elemOffset = arrayTTBc.value.typ.asInstanceOf[TContainer].elementsOffset(length)
    val elemSize = UnsafeUtils.arrayElementSize(elemType)

    new UnsafeIndexedSeqAnnotation(region, arrayTTBc, elemSize, aoff, aoff + elemOffset, length)
  }

  def readStruct(region: MemoryBuffer, offset: Int, ttBc: BroadcastTypeTree): UnsafeRow = {
    new UnsafeRow(ttBc, region, offset)
  }

  def readString(region: MemoryBuffer, offset: Int): String =
    new String(readBinary(region, offset))

  def readLocus(region: MemoryBuffer, offset: Int): Locus = {
    val ft = TLocus.fundamentalType.asInstanceOf[TStruct]
    Locus(
      readString(region, offset + ft.byteOffsets(0)),
      region.loadInt(offset + ft.byteOffsets(1)))
  }

  def readAltAllele(region: MemoryBuffer, offset: Int): AltAllele = {
    val ft = TAltAllele.fundamentalType.asInstanceOf[TStruct]
    AltAllele(
      readString(region, offset + ft.byteOffsets(0)),
      readString(region, offset + ft.byteOffsets(1)))
  }

  def readArrayAltAllele(region: MemoryBuffer, offset: Int): Array[AltAllele] = {
    val elemType = TAltAllele
    val t = TArray(elemType)

    val aoff = region.loadInt(offset)

    val length = region.loadInt(aoff)
    val elemOffset = t.elementsOffset(length)
    val elemSize = UnsafeUtils.arrayElementSize(elemType)

    val a = new Array[AltAllele](length)
    var i = 0
    while (i < length) {
      a(i) = readAltAllele(region, aoff + elemOffset + i * elemSize)
      i += 1
    }
    a
  }

  def readArrayInt(region: MemoryBuffer, offset: Int): Array[Int] = {
    val elemType = TInt32
    val t = TArray(elemType)

    val aoff = region.loadInt(offset)

    val length = region.loadInt(aoff)
    val elemOffset = t.elementsOffset(length)
    val elemSize = UnsafeUtils.arrayElementSize(elemType)

    val a = new Array[Int](length)
    var i = 0
    while (i < length) {
      a(i) = region.loadInt(aoff + elemOffset + i * elemSize)
      i += 1
    }
    a
  }

  def read(region: MemoryBuffer, offset: Int, t: Type, ttBc: BroadcastTypeTree): Any = {
    t match {
      case TBoolean =>
        val b = region.loadByte(offset)
        assert(b == 0 || b == 1, s"invalid boolean byte $b from offset $offset")
        b == 1
      case TInt32 | TCall => region.loadInt(offset)
      case TInt64 => region.loadLong(offset)
      case TFloat32 => region.loadFloat(offset)
      case TFloat64 => region.loadDouble(offset)
      case TArray(elementType) =>
        readArray(region, offset, elementType, ttBc)
      case TSet(elementType) =>
        readArray(region, offset, elementType, ttBc).toSet
      case TString => readString(region, offset)
      case td: TDict =>
        readArray(region, offset, td.elementType, ttBc).asInstanceOf[IndexedSeq[Row]].map(r => (r.get(0), r.get(1))).toMap
      case struct: TStruct =>
        readStruct(region, offset, ttBc)

      case TVariant =>
        val ft = TVariant.fundamentalType.asInstanceOf[TStruct]
        Variant(
          readString(region, offset + ft.byteOffsets(0)),
          region.loadInt(offset + ft.byteOffsets(1)),
          readString(region, offset + ft.byteOffsets(2)),
          readArrayAltAllele(region, offset + ft.byteOffsets(3)))
      case TLocus => readLocus(region, offset)
      case TAltAllele => readAltAllele(region, offset)
      case TInterval =>
        val ft = TInterval.fundamentalType.asInstanceOf[TStruct]
        Interval[Locus](
          readLocus(region, offset + ft.byteOffsets(0)),
          readLocus(region, offset + ft.byteOffsets(1)))
      case TGenotype =>
        val ft = TGenotype.fundamentalType.asInstanceOf[TStruct]
        val gt: Int =
          if (region.loadBit(offset, 0))
            -1
          else
            region.loadInt(offset + ft.byteOffsets(0))
        val ad =
          if (region.loadBit(offset, 1))
            null
          else
            readArrayInt(region, offset + ft.byteOffsets(1))
        val dp: Int =
          if (region.loadBit(offset, 2))
            -1
          else
            region.loadInt(offset + ft.byteOffsets(2))
        val gq: Int =
          if (region.loadBit(offset, 3))
            -1
          else
            region.loadInt(offset + ft.byteOffsets(3))
        val px =
          if (region.loadBit(offset, 4))
            null
          else
            readArrayInt(region, offset + ft.byteOffsets(4))
        val fakeRef = region.loadByte(offset + ft.byteOffsets(5)) != 0
        val isLinearScale = region.loadByte(offset + ft.byteOffsets(6)) != 0

        new GenericGenotype(gt, ad, dp, gq, px, fakeRef, isLinearScale)
    }
  }
}

class UnsafeRow(val ttBc: BroadcastTypeTree,
  val region: MemoryBuffer, var offset: Int) extends Row {

  def t: TStruct = ttBc.value.typ.asInstanceOf[TStruct]

  def length: Int = t.size

  private def assertDefined(i: Int) {
    if (isNullAt(i))
      throw new NullPointerException(s"null value at index $i")
  }

  def get(i: Int): Any = {
    if (isNullAt(i))
      null
    else
      UnsafeRow.read(region, offset + t.byteOffsets(i), t.fields(i).typ, ttBc.value.subtree(i))
  }

  def copy(): Row = new UnsafeRow(ttBc, region, offset)

  override def getInt(i: Int): Int = {
    assertDefined(i)
    region.loadInt(offset + t.byteOffsets(i))
  }

  override def getLong(i: Int): Long = {
    assertDefined(i)
    region.loadLong(offset + t.byteOffsets(i))
  }

  override def getFloat(i: Int): Float = {
    assertDefined(i)
    region.loadFloat(offset + t.byteOffsets(i))
  }

  override def getDouble(i: Int): Double = {
    assertDefined(i)
    region.loadDouble(offset + t.byteOffsets(i))
  }

  override def getBoolean(i: Int): Boolean = {
    assertDefined(i)
    getByte(i) == 1
  }

  override def getByte(i: Int): Byte = {
    assertDefined(i)
    region.loadByte(offset + t.byteOffsets(i))
  }

  override def isNullAt(i: Int): Boolean = {
    if (i < 0 || i >= t.size)
      throw new IndexOutOfBoundsException(i.toString)
    region.loadBit(offset, i)
  }
}
