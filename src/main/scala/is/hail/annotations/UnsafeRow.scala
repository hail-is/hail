package is.hail.annotations

import java.io.{DataInputStream, DataOutputStream}

import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import is.hail.expr._
import is.hail.utils.Interval
import is.hail.variant.{AltAllele, GenericGenotype, Locus, Variant}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.Row

class UnsafeIndexedSeqAnnotation(
  var ttBc: Broadcast[TypeTree],
  var region: MemoryBuffer, var aoff: Long, var elemsOffset: Long,
  var length: Int,
  var elemSize: Long) extends IndexedSeq[Annotation] with KryoSerializable {
  def apply(i: Int): Annotation = {
    if (i < 0 || i >= length)
      throw new IndexOutOfBoundsException(i.toString)
    assert(i >= 0 && i < length)
    if (region.loadBit(aoff + 4, i))
      null
    else {
      UnsafeRow.read(region, elemsOffset + i * elemSize,
        ttBc.value.typ.asInstanceOf[TContainer].elementType,
        ttBc.value.subtree(0))
    }
  }

  override def write(kryo: Kryo, output: Output) {
    kryo.writeObject(output, ttBc)
    val enc = new Encoder(new DataOutputStream(output))
    enc.writeRegionValue(ttBc.value.typ, region, aoff)
  }

  override def read(kryo: Kryo, input: Input) {
    val cTorrentBroadcast = Class.forName("org.apache.spark.broadcast.TorrentBroadcast")
    ttBc = kryo.readObject(input, cTorrentBroadcast).asInstanceOf[Broadcast[TypeTree]]
    val t = ttBc.value.typ.asInstanceOf[TArray]

    val dec = new Decoder(new DataInputStream(input))
    region = MemoryBuffer()
    aoff = dec.readRegionValue(t, region)

    length = region.loadInt(aoff)

    elemSize = UnsafeUtils.arrayElementSize(t.elementType)
    elemsOffset = aoff + t.elementsOffset(length)
  }
}

object UnsafeRow {
  def readBinary(region: MemoryBuffer, offset: Long): Array[Byte] = {
    val start = region.loadAddress(offset)
    assert(offset > 0 && (offset & 0x3) == 0, s"invalid binary start: $offset")
    val binLength = region.loadInt(start)
    region.loadBytes(start + 4, binLength)
  }

  def readArray(region: MemoryBuffer, offset: Long, elemType: Type, arrayTTBc: Broadcast[TypeTree]): IndexedSeq[Any] = {
    val aoff = region.loadAddress(offset)

    val length = region.loadInt(aoff)
    val elemsOffset = arrayTTBc.value.typ.asInstanceOf[TContainer].elementsOffset(length)
    val elemSize = UnsafeUtils.arrayElementSize(elemType)

    new UnsafeIndexedSeqAnnotation(arrayTTBc, region, aoff, aoff + elemsOffset, length, elemSize)
  }

  def readStruct(region: MemoryBuffer, offset: Long, ttBc: Broadcast[TypeTree]): UnsafeRow = {
    new UnsafeRow(ttBc, region, offset)
  }

  def readString(region: MemoryBuffer, offset: Long): String =
    new String(readBinary(region, offset))

  def readLocus(region: MemoryBuffer, offset: Long): Locus = {
    val ft = TLocus.fundamentalType.asInstanceOf[TStruct]
    Locus(
      readString(region, offset + ft.byteOffsets(0)),
      region.loadInt(offset + ft.byteOffsets(1)))
  }

  def readAltAllele(region: MemoryBuffer, offset: Long): AltAllele = {
    val ft = TAltAllele.fundamentalType.asInstanceOf[TStruct]
    AltAllele(
      readString(region, offset + ft.byteOffsets(0)),
      readString(region, offset + ft.byteOffsets(1)))
  }

  def readArrayAltAllele(region: MemoryBuffer, offset: Long): Array[AltAllele] = {
    val elemType = TAltAllele
    val t = TArray(elemType)

    val aoff = region.loadAddress(offset)

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

  def readArrayInt(region: MemoryBuffer, offset: Long): Array[Int] = {
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

  def read(region: MemoryBuffer, offset: Long, t: Type, ttBc: Broadcast[TypeTree]): Any = {
    t match {
      case TBoolean =>
        region.loadBoolean(offset)
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
        val a = readArray(region, offset, td.elementType, ttBc)
        a.asInstanceOf[IndexedSeq[Row]].map(r => (r.get(0), r.get(1))).toMap
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

class UnsafeRow(var ttBc: Broadcast[TypeTree],
  var region: MemoryBuffer, var offset: Long) extends Row with KryoSerializable {

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

  override def write(kryo: Kryo, output: Output) {
    kryo.writeObject(output, ttBc)
    val enc = new Encoder(new DataOutputStream(output))
    enc.writeRegionValue(ttBc.value.typ, region, offset)
  }

  override def read(kryo: Kryo, input: Input) {
    val cTorrentBroadcast = Class.forName("org.apache.spark.broadcast.TorrentBroadcast")
    ttBc = kryo.readObject(input, cTorrentBroadcast).asInstanceOf[Broadcast[TypeTree]]
    val dec = new Decoder(new DataInputStream(input))
    region = MemoryBuffer()
    offset = dec.readRegionValue(ttBc.value.typ, region)
  }
}
