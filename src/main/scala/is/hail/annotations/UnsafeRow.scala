package is.hail.annotations

import java.io.{DataInputStream, DataOutputStream, ObjectInputStream, ObjectOutputStream}

import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import is.hail.expr._
import is.hail.utils._
import is.hail.variant.{AltAllele, GRBase, GenericGenotype, Locus, Variant}
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
      val t = ttBc.value.typ.asInstanceOf[TContainer]
      UnsafeRow.read(region, t.loadElement(region, aoff, length, i),
        t.elementType, ttBc.value.subtree(0))
    }
  }

  override def write(kryo: Kryo, output: Output) {
    kryo.writeObject(output, ttBc)

    val aos = new ArrayOutputStream()
    val enc = new Encoder(new LZ4OutputBuffer(aos))
    enc.writeRegionValue(ttBc.value.typ, region, aoff)
    enc.flush()

    output.writeInt(aos.off)
    output.write(aos.a, 0, aos.off)
  }

  private def writeObject(out: ObjectOutputStream) {
    out.writeObject(ttBc)

    val aos = new ArrayOutputStream()
    val enc = new Encoder(new LZ4OutputBuffer(aos))
    enc.writeRegionValue(ttBc.value.typ, region, aoff)
    enc.flush()

    out.writeInt(aos.off)
    out.write(aos.a, 0, aos.off)
  }

  override def read(kryo: Kryo, input: Input) {
    val cTorrentBroadcast = Class.forName("org.apache.spark.broadcast.TorrentBroadcast")
    ttBc = kryo.readObject(input, cTorrentBroadcast).asInstanceOf[Broadcast[TypeTree]]
    val t = ttBc.value.typ.asInstanceOf[TArray]

    val smallInOff = input.readInt()
    val a = new Array[Byte](smallInOff)
    input.readFully(a, 0, smallInOff)
    val dec = new Decoder(
      new LZ4InputBuffer(
        new ArrayInputStream(a)))

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
    val a = new Array[Byte](smallInOff)
    in.readFully(a, 0, smallInOff)
    val dec = new Decoder(
      new LZ4InputBuffer(
        new ArrayInputStream(a)))

    region = MemoryBuffer()
    aoff = dec.readRegionValue(ttBc.value.typ, region)

    length = region.loadInt(aoff)

    elemSize = UnsafeUtils.arrayElementSize(t.elementType)
    elemsOffset = aoff + t.elementsOffset(length)
  }
}

object UnsafeRow {
  def readBinary(region: MemoryBuffer, boff: Long): Array[Byte] = {
    val length = TBinary.loadLength(region, boff)
    region.loadBytes(TBinary.bytesOffset(boff), length)
  }

  def readArray(region: MemoryBuffer, aoff: Long, elemType: Type, arrayTTBc: Broadcast[TypeTree]): IndexedSeq[Any] = {
    val length = region.loadInt(aoff)
    val elemsOffset = arrayTTBc.value.typ.asInstanceOf[TContainer].elementsOffset(length)
    val elemSize = UnsafeUtils.arrayElementSize(elemType)

    new UnsafeIndexedSeqAnnotation(arrayTTBc, region, aoff, aoff + elemsOffset, length, elemSize)
  }

  def readStruct(region: MemoryBuffer, offset: Long, ttBc: Broadcast[TypeTree]): UnsafeRow = {
    new UnsafeRow(ttBc, region, offset)
  }

  def readString(region: MemoryBuffer, boff: Long): String =
    new String(readBinary(region, boff))

  def readLocus(region: MemoryBuffer, offset: Long, gr: GRBase): Locus = {
    val ft = TLocus(gr).fundamentalType.asInstanceOf[TStruct]
    Locus(
      readString(region, ft.loadField(region, offset, 0)),
      region.loadInt(ft.loadField(region, offset, 1)))
  }

  def readAltAllele(region: MemoryBuffer, offset: Long): AltAllele = {
    val ft = TAltAllele.fundamentalType.asInstanceOf[TStruct]
    AltAllele(
      readString(region, ft.loadField(region, offset, 0)),
      readString(region, ft.loadField(region, offset, 1)))
  }

  def readArrayAltAllele(region: MemoryBuffer, aoff: Long): Array[AltAllele] = {
    val elemType = TAltAllele
    val t = TArray(elemType)

    val length = region.loadInt(aoff)
    val elemOffset = t.elementsOffset(length)
    val elemSize = UnsafeUtils.arrayElementSize(elemType)

    val a = new Array[AltAllele](length)
    var i = 0
    while (i < length) {
      a(i) = readAltAllele(region, t.loadElement(region, aoff, length, i))
      i += 1
    }
    a
  }

  def readArrayInt(region: MemoryBuffer, aoff: Long): Array[Int] = {
    val elemType = TInt32
    val t = TArray(elemType)

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

      case x: TVariant =>
        val ft = x.fundamentalType.asInstanceOf[TStruct]
        Variant(
          readString(region, ft.loadField(region, offset, 0)),
          region.loadInt(ft.loadField(region, offset, 1)),
          readString(region, ft.loadField(region, offset, 2)),
          readArrayAltAllele(region, ft.loadField(region, offset, 3)))
      case x: TLocus => readLocus(region, offset, x.gr)
      case TAltAllele => readAltAllele(region, offset)
      case x: TInterval =>
        val ft = x.fundamentalType.asInstanceOf[TStruct]
        Interval[Locus](
          readLocus(region, ft.loadField(region, offset, 0), x.gr),
          readLocus(region, ft.loadField(region, offset, 1), x.gr))
      case TGenotype =>
        val ft = TGenotype.fundamentalType.asInstanceOf[TStruct]
        val gt: Int =
          if (ft.isFieldDefined(region, offset, 0))
            region.loadInt(ft.loadField(region, offset, 0))
          else
            -1
        val ad =
          if (ft.isFieldDefined(region, offset, 1))
            readArrayInt(region, ft.loadField(region, offset, 1))
          else
            null
        val dp: Int =
          if (ft.isFieldDefined(region, offset, 2))
            region.loadInt(ft.loadField(region, offset, 2))
          else
            -1
        val gq: Int =
          if (ft.isFieldDefined(region, offset, 3))
            region.loadInt(ft.loadField(region, offset, 3))
          else
            -1
        val px =
          if (ft.isFieldDefined(region, offset, 4))
            readArrayInt(region, ft.loadField(region, offset, 4))
          else
            null
        val fakeRef = region.loadByte(ft.loadField(region, offset, 5)) != 0
        val isLinearScale = region.loadByte(ft.loadField(region, offset, 6)) != 0

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
      UnsafeRow.read(region, t.loadField(region, offset, i), t.fields(i).typ, ttBc.value.subtree(i))
  }

  def copy(): Row = new UnsafeRow(ttBc, region, offset)

  override def getInt(i: Int): Int = {
    assertDefined(i)
    region.loadInt(t.loadField(region, offset, i))
  }

  override def getLong(i: Int): Long = {
    assertDefined(i)
    region.loadLong(t.loadField(region, offset, i))
  }

  override def getFloat(i: Int): Float = {
    assertDefined(i)
    region.loadFloat(t.loadField(region, offset, i))
  }

  override def getDouble(i: Int): Double = {
    assertDefined(i)
    region.loadDouble(t.loadField(region, offset, i))
  }

  override def getBoolean(i: Int): Boolean = {
    assertDefined(i)
    region.loadBoolean(t.loadField(region, offset, i))
  }

  override def getByte(i: Int): Byte = {
    assertDefined(i)
    region.loadByte(t.loadField(region, offset, i))
  }

  override def isNullAt(i: Int): Boolean = {
    if (i < 0 || i >= t.size)
      throw new IndexOutOfBoundsException(i.toString)
    !t.isFieldDefined(region, offset, i)
  }

  override def write(kryo: Kryo, output: Output) {
    kryo.writeObject(output, ttBc)

    val aos = new ArrayOutputStream()
    val enc = new Encoder(new LZ4OutputBuffer(aos))
    enc.writeRegionValue(t, region, offset)
    enc.flush()

    output.writeInt(aos.off)
    output.write(aos.a, 0, aos.off)
  }

  private def writeObject(out: ObjectOutputStream) {
    out.writeObject(ttBc)

    val aos = new ArrayOutputStream()
    val enc = new Encoder(new LZ4OutputBuffer(aos))
    enc.writeRegionValue(t, region, offset)
    enc.flush()

    out.writeInt(aos.off)
    out.write(aos.a, 0, aos.off)
  }

  override def read(kryo: Kryo, input: Input) {
    val cTorrentBroadcast = Class.forName("org.apache.spark.broadcast.TorrentBroadcast")
    ttBc = kryo.readObject(input, cTorrentBroadcast).asInstanceOf[Broadcast[TypeTree]]

    val smallInOff = input.readInt()
    val a = new Array[Byte](smallInOff)
    input.readFully(a, 0, smallInOff)
    val dec = new Decoder(
      new LZ4InputBuffer(
        new ArrayInputStream(a)))

    region = MemoryBuffer()
    offset = dec.readRegionValue(ttBc.value.typ, region)
  }

  private def readObject(in: ObjectInputStream) {
    ttBc = in.readObject().asInstanceOf[Broadcast[TypeTree]]

    val smallInOff = in.readInt()
    val a = new Array[Byte](smallInOff)
    in.readFully(a, 0, smallInOff)
    val dec = new Decoder(
      new LZ4InputBuffer(
        new ArrayInputStream(a)))

    region = MemoryBuffer()
    offset = dec.readRegionValue(ttBc.value.typ, region)
  }
}
