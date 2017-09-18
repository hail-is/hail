package is.hail.annotations

import java.io.{DataInputStream, DataOutputStream, ObjectInputStream, ObjectOutputStream}

import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import is.hail.expr._
import is.hail.utils.Interval
import is.hail.variant.{AltAllele, GenericGenotype, GenomeReference, Locus, Variant}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.Row

object UnsafeIndexedSeq {
  def apply(sc: SparkContext, elementType: Type, elements: Array[RegionValue]): UnsafeIndexedSeq = {
    val t = TArray(elementType)

    val region = MemoryBuffer()
    val rvb = new RegionValueBuilder(region)
    rvb.start(t)
    rvb.startArray(elements.length)
    var i = 0
    while (i < elements.length) {
      rvb.addRegionValue(elementType, elements(i))
      i += 1
    }
    rvb.endArray()

    UnsafeIndexedSeq(sc, region, t, rvb.end(), elements.length)
  }

  def apply(sc: SparkContext, region: MemoryBuffer, t: TArray, aoff: Long, length: Int): UnsafeIndexedSeq = {
    new UnsafeIndexedSeq(region, BroadcastTypeTree(sc, t), t.elementByteSize, aoff, aoff + t.elementsOffset(length), length)
  }

  def apply(ttBc: Broadcast[TypeTree], region: MemoryBuffer, aoff: Long): UnsafeIndexedSeq = {
    val t = ttBc.value.typ.asInstanceOf[TContainer]
    val length = region.loadInt(aoff)
    new UnsafeIndexedSeq(region, ttBc, t.elementByteSize, aoff, aoff + t.elementsOffset(length), length)
  }

  def apply(sc: SparkContext, t: TArray, a: IndexedSeq[Annotation]): UnsafeIndexedSeq = {
    val region = MemoryBuffer()
    val rvb = new RegionValueBuilder(region)
    rvb.start(t)
    rvb.startArray(a.length)
    var i = 0
    while (i < a.length) {
      rvb.addAnnotation(t.elementType, a(i))
      i += 1
    }
    rvb.endArray()
    UnsafeIndexedSeq(sc, region, t, rvb.end(), a.length)
  }

  def empty(sc: SparkContext, t: TArray): UnsafeIndexedSeq = {
    val region = MemoryBuffer()
    val rvb = new RegionValueBuilder(region)
    rvb.start(t)
    rvb.startArray(0)
    rvb.endArray()
    UnsafeIndexedSeq(sc, region, t, rvb.end(), 0)
  }
}

class UnsafeIndexedSeq(
  var region: MemoryBuffer,
  var ttBc: Broadcast[TypeTree],
  var elemSize: Long, var aoff: Long, var elemsOffset: Long,
  var length: Int) extends IndexedSeq[Annotation] with KryoSerializable with Serializable {
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

    val enc = new Encoder()
    enc.writeRegionValue(ttBc.value.typ, region, aoff)

    assert(enc.outOff <= Int.MaxValue)
    val smallOutOff = enc.outOff.toInt
    output.writeInt(smallOutOff)
    output.write(enc.outMem, 0, smallOutOff)
  }

  private def writeObject(out: ObjectOutputStream) {
    out.writeObject(ttBc)

    val enc = new Encoder()
    enc.writeRegionValue(ttBc.value.typ, region, aoff)

    assert(enc.outOff <= Int.MaxValue)
    val smallOutOff = enc.outOff.toInt
    out.writeInt(smallOutOff)
    out.write(enc.outMem, 0, smallOutOff)
  }

  override def read(kryo: Kryo, input: Input) {
    val cTorrentBroadcast = Class.forName("org.apache.spark.broadcast.TorrentBroadcast")
    ttBc = kryo.readObject(input, cTorrentBroadcast).asInstanceOf[Broadcast[TypeTree]]
    val t = ttBc.value.typ.asInstanceOf[TArray]

    val smallInOff = input.readInt()
    val inMem = new Array[Byte](smallInOff)

    var off = 0
    while (off < smallInOff) {
      val n = input.read(inMem, off, smallInOff - off)
      off += n
    }

    val dec = new Decoder()
    dec.set(inMem)

    region = MemoryBuffer()
    aoff = dec.readRegionValue(t, region)

    length = region.loadInt(aoff)

    elemSize = UnsafeUtils.arrayElementSize(t.elementType)
    elemsOffset = aoff + t.elementsOffset(length)
  }

  private def readObject(in: ObjectInputStream) {
    ttBc = in.readObject().asInstanceOf[Broadcast[TypeTree]]
    val t = ttBc.value.typ.asInstanceOf[TArray]

    val smallInOff = in.readInt()
    val inMem = new Array[Byte](smallInOff)
    in.readFully(inMem, 0, smallInOff)

    val dec = new Decoder()
    dec.set(inMem)

    region = MemoryBuffer()
    aoff = dec.readRegionValue(ttBc.value.typ, region)

    length = region.loadInt(aoff)

    elemSize = UnsafeUtils.arrayElementSize(t.elementType)
    elemsOffset = aoff + t.elementsOffset(length)
  }
}

object UnsafeRow {
  def apply(ttBc: Broadcast[TypeTree], r: Row): UnsafeRow = {
    val region = MemoryBuffer()
    val rvb = new RegionValueBuilder(region)
    val t = ttBc.value.typ
    rvb.start(t)
    rvb.addAnnotation(t, r)
    new UnsafeRow(ttBc, region, rvb.end())
  }

  def apply(ttBc: Broadcast[TypeTree], args: Annotation*): UnsafeRow = {
    val region = MemoryBuffer()
    val rvb = new RegionValueBuilder(region)
    val t = ttBc.value.typ.asInstanceOf[TStruct]
    assert(t.size == args.length)
    rvb.start(t)
    rvb.startStruct()
    var i = 0
    while (i < args.length) {
      rvb.addAnnotation(t.fields(i).typ, args(i))
      i += 1
    }
    rvb.endStruct()
    new UnsafeRow(ttBc, region, rvb.end())
  }

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

    new UnsafeIndexedSeq(region, arrayTTBc, elemSize, aoff, aoff + elemsOffset, length)
  }

  def readStruct(region: MemoryBuffer, offset: Long, ttBc: Broadcast[TypeTree]): UnsafeRow = {
    new UnsafeRow(ttBc, region, offset)
  }

  def readString(region: MemoryBuffer, offset: Long): String =
    new String(readBinary(region, offset))

  def readLocus(region: MemoryBuffer, offset: Long): Locus = {
    val ft = TLocus(GenomeReference.GRCh37).fundamentalType.asInstanceOf[TStruct]
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

  private val tArrayAltAllele = TArray(TAltAllele)

  def readArrayAltAllele(region: MemoryBuffer, offset: Long): Array[AltAllele] = {
    val t = tArrayAltAllele

    val aoff = region.loadAddress(offset)

    val length = region.loadInt(aoff)
    val elemOffset = t.elementsOffset(length)
    val elemSize = t.elementByteSize

    val a = new Array[AltAllele](length)
    var i = 0
    while (i < length) {
      a(i) = readAltAllele(region, aoff + elemOffset + i * elemSize)
      i += 1
    }
    a
  }

  private val tArrayInt32 = TArray(TInt32)

  def readArrayInt(region: MemoryBuffer, offset: Long): Array[Int] = {
    val t = tArrayInt32

    val aoff = region.loadInt(offset)

    val length = region.loadInt(aoff)
    val elemOffset = t.elementsOffset(length)
    val elemSize = t.elementByteSize

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

      case x: TVariant =>
        val ft = x.fundamentalType.asInstanceOf[TStruct]
        Variant(
          readString(region, offset + ft.byteOffsets(0)),
          region.loadInt(offset + ft.byteOffsets(1)),
          readString(region, offset + ft.byteOffsets(2)),
          readArrayAltAllele(region, offset + ft.byteOffsets(3)))
      case x: TLocus => readLocus(region, offset)
      case TAltAllele => readAltAllele(region, offset)
      case x: TInterval =>
        val ft = x.fundamentalType.asInstanceOf[TStruct]
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

  def this() = this(null, null, 0L)

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

    val enc = new Encoder()
    enc.writeRegionValue(ttBc.value.typ, region, offset)

    assert(enc.outOff <= Int.MaxValue)
    val smallOutOff = enc.outOff.toInt
    output.writeInt(smallOutOff)
    output.write(enc.outMem, 0, smallOutOff)
  }

  private def writeObject(out: ObjectOutputStream) {
    out.writeObject(ttBc)

    val enc = new Encoder()
    enc.writeRegionValue(ttBc.value.typ, region, offset)

    assert(enc.outOff <= Int.MaxValue)
    val smallOutOff = enc.outOff.toInt
    out.writeInt(smallOutOff)
    out.write(enc.outMem, 0, smallOutOff)
  }

  override def read(kryo: Kryo, input: Input) {
    val cTorrentBroadcast = Class.forName("org.apache.spark.broadcast.TorrentBroadcast")
    ttBc = kryo.readObject(input, cTorrentBroadcast).asInstanceOf[Broadcast[TypeTree]]

    val smallInOff = input.readInt()
    val inMem = new Array[Byte](smallInOff)

    var off = 0
    while (off < smallInOff) {
      val n = input.read(inMem, off, smallInOff - off)
      off += n
    }

    val dec = new Decoder()
    dec.set(inMem)

    region = MemoryBuffer()
    offset = dec.readRegionValue(ttBc.value.typ, region)
  }

  private def readObject(in: ObjectInputStream) {
    ttBc = in.readObject().asInstanceOf[Broadcast[TypeTree]]

    val smallInOff = in.readInt()
    val inMem = new Array[Byte](smallInOff)
    in.readFully(inMem, 0, smallInOff)

    val dec = new Decoder()
    dec.set(inMem)

    region = MemoryBuffer()
    offset = dec.readRegionValue(ttBc.value.typ, region)
  }
}
