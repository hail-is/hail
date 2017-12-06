package is.hail.annotations

import java.io.{ObjectInputStream, ObjectOutputStream}

import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import is.hail.expr._
import is.hail.io._
import is.hail.utils._
import is.hail.variant.{AltAllele, GRBase, Locus, Variant}
import org.apache.spark.sql.Row

object UnsafeIndexedSeq {
  def apply(t: TArray, elements: Array[RegionValue]): UnsafeIndexedSeq = {
    val region = MemoryBuffer()
    val rvb = new RegionValueBuilder(region)
    rvb.start(t)
    rvb.startArray(elements.length)
    var i = 0
    while (i < elements.length) {
      rvb.addRegionValue(t.elementType, elements(i))
      i += 1
    }
    rvb.endArray()

    new UnsafeIndexedSeq(t, region, rvb.end())
  }

  def apply(t: TArray, a: IndexedSeq[Annotation]): UnsafeIndexedSeq = {
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
    new UnsafeIndexedSeq(t, region, rvb.end())
  }

  def empty(t: TArray): UnsafeIndexedSeq = {
    val region = MemoryBuffer()
    val rvb = new RegionValueBuilder(region)
    rvb.start(t)
    rvb.startArray(0)
    rvb.endArray()
    new UnsafeIndexedSeq(t, region, rvb.end())
  }
}

class UnsafeIndexedSeq(
  var t: TContainer,
  var region: MemoryBuffer, var aoff: Long) extends IndexedSeq[Annotation] with KryoSerializable with Serializable {

  var length: Int = t.loadLength(region, aoff)

  def apply(i: Int): Annotation = {
    if (i < 0 || i >= length)
      throw new IndexOutOfBoundsException(i.toString)
    if (t.isElementDefined(region, aoff, i)) {
      UnsafeRow.read(t.elementType, region, t.loadElement(region, aoff, length, i))
    } else
      null
  }

  override def write(kryo: Kryo, output: Output) {
    kryo.writeObject(output, t)

    val aos = new ArrayOutputStream()
    val enc = new Encoder(new LZ4OutputBuffer(aos))
    enc.writeRegionValue(t, region, aoff)
    enc.flush()

    output.writeInt(aos.off)
    output.write(aos.a, 0, aos.off)
  }

  private def writeObject(out: ObjectOutputStream) {
    out.writeObject(t)

    val aos = new ArrayOutputStream()
    val enc = new Encoder(new LZ4OutputBuffer(aos))
    enc.writeRegionValue(t, region, aoff)
    enc.flush()

    out.writeInt(aos.off)
    out.write(aos.a, 0, aos.off)
  }

  override def read(kryo: Kryo, input: Input) {
    t = kryo.readObject(input, classOf[TArray])

    val smallInOff = input.readInt()
    val a = new Array[Byte](smallInOff)
    input.readFully(a, 0, smallInOff)
    val dec = new Decoder(
      new LZ4InputBuffer(
        new ArrayInputStream(a)))

    region = MemoryBuffer()
    aoff = dec.readRegionValue(t, region)

    length = region.loadInt(aoff)
  }

  private def readObject(in: ObjectInputStream) {
    t = in.readObject().asInstanceOf[TArray]

    val smallInOff = in.readInt()
    val a = new Array[Byte](smallInOff)
    in.readFully(a, 0, smallInOff)
    val dec = new Decoder(
      new LZ4InputBuffer(
        new ArrayInputStream(a)))

    region = MemoryBuffer()
    aoff = dec.readRegionValue(t, region)

    length = region.loadInt(aoff)
  }
}

object UnsafeRow {
  def readBinary(region: MemoryBuffer, boff: Long): Array[Byte] = {
    val binLength = TBinary.loadLength(region, boff)
    region.loadBytes(TBinary.bytesOffset(boff), binLength)
  }

  def readArray(t: TContainer, region: MemoryBuffer, aoff: Long): IndexedSeq[Any] =
    new UnsafeIndexedSeq(t, region, aoff)

  def readStruct(t: TStruct, region: MemoryBuffer, offset: Long): UnsafeRow =
    new UnsafeRow(t, region, offset)

  def readString(region: MemoryBuffer, boff: Long): String =
    new String(readBinary(region, boff))

  def readLocus(region: MemoryBuffer, offset: Long, gr: GRBase): Locus = {
    val ft = gr.locus.fundamentalType.asInstanceOf[TStruct]
    Locus(
      readString(region, ft.loadField(region, offset, 0)),
      region.loadInt(ft.loadField(region, offset, 1)))
  }

  def readAltAllele(region: MemoryBuffer, offset: Long): AltAllele = {
    val ft = TAltAllele().fundamentalType.asInstanceOf[TStruct]
    AltAllele(
      readString(region, ft.loadField(region, offset, 0)),
      readString(region, ft.loadField(region, offset, 1)))
  }

  private val tArrayAltAllele = TArray(TAltAllele())

  def readArrayAltAllele(region: MemoryBuffer, aoff: Long): Array[AltAllele] = {
    val t = tArrayAltAllele

    val length = region.loadInt(aoff)
    val a = new Array[AltAllele](length)
    var i = 0
    while (i < length) {
      a(i) = readAltAllele(region, t.loadElement(region, aoff, length, i))
      i += 1
    }
    a
  }

  private val tArrayInt32 = TArray(!TInt32())

  def readArrayInt(region: MemoryBuffer, aoff: Long): Array[Int] = {
    val t = tArrayInt32

    val length = region.loadInt(aoff)
    val a = new Array[Int](length)
    var i = 0
    while (i < length) {
      a(i) = region.loadInt(t.loadElement(region, aoff, length, i))
      i += 1
    }
    a
  }

  def read(t: Type, region: MemoryBuffer, offset: Long): Any = {
    t match {
      case _: TBoolean =>
        region.loadBoolean(offset)
      case _: TInt32 | _: TCall => region.loadInt(offset)
      case _: TInt64 => region.loadLong(offset)
      case _: TFloat32 => region.loadFloat(offset)
      case _: TFloat64 => region.loadDouble(offset)
      case t: TArray =>
        readArray(t, region, offset)
      case t: TSet =>
        readArray(t, region, offset).toSet
      case _: TString => readString(region, offset)
      case _: TBinary => readBinary(region, offset)
      case td: TDict =>
        val a = readArray(td, region, offset)
        a.asInstanceOf[IndexedSeq[Row]].map(r => (r.get(0), r.get(1))).toMap
      case t: TStruct =>
        readStruct(t, region, offset)

      case x: TVariant =>
        val ft = x.fundamentalType.asInstanceOf[TStruct]
        Variant(
          readString(region, ft.loadField(region, offset, 0)),
          region.loadInt(ft.loadField(region, offset, 1)),
          readString(region, ft.loadField(region, offset, 2)),
          readArrayAltAllele(region, ft.loadField(region, offset, 3)))
      case x: TLocus => readLocus(region, offset, x.gr)
      case _: TAltAllele => readAltAllele(region, offset)
      case x: TInterval =>
        val ft = x.fundamentalType.asInstanceOf[TStruct]
        Interval[Locus](
          readLocus(region, ft.loadField(region, offset, 0), x.gr),
          readLocus(region, ft.loadField(region, offset, 1), x.gr))(x.locusOrdering)
    }
  }
}

class UnsafeRow(var t: TStruct,
  var region: MemoryBuffer, var offset: Long) extends Row with KryoSerializable {

  def this(t: TStruct, rv: RegionValue) = this(t, rv.region, rv.offset)
  def this(t: TStruct) = this(t, null, 0)
  def this() = this(null, null, 0)

  def set(newRegion: MemoryBuffer, newOffset: Long) {
    region = newRegion
    offset = newOffset
  }

  def set(rv: RegionValue): Unit = set(rv.region, rv.offset)

  def length: Int = t.size

  private def assertDefined(i: Int) {
    if (isNullAt(i))
      throw new NullPointerException(s"null value at index $i")
  }

  def get(i: Int): Any = {
    if (isNullAt(i))
      null
    else
      UnsafeRow.read(t.fieldType(i), region, t.loadField(region, offset, i))
  }

  def copy(): Row = new UnsafeRow(t, region, offset)

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
    kryo.writeObject(output, t)

    val aos = new ArrayOutputStream()
    val enc = new Encoder(new LZ4OutputBuffer(aos))
    enc.writeRegionValue(t, region, offset)
    enc.flush()

    output.writeInt(aos.off)
    output.write(aos.a, 0, aos.off)
  }

  private def writeObject(out: ObjectOutputStream) {
    out.writeObject(t)

    val aos = new ArrayOutputStream()
    val enc = new Encoder(new LZ4OutputBuffer(aos))
    enc.writeRegionValue(t, region, offset)
    enc.flush()

    out.writeInt(aos.off)
    out.write(aos.a, 0, aos.off)
  }

  override def read(kryo: Kryo, input: Input) {
    t = kryo.readObject(input, classOf[TStruct])

    val smallInOff = input.readInt()
    val a = new Array[Byte](smallInOff)
    input.readFully(a, 0, smallInOff)
    val dec = new Decoder(
      new LZ4InputBuffer(
        new ArrayInputStream(a)))

    region = MemoryBuffer()
    offset = dec.readRegionValue(t, region)
  }

  private def readObject(in: ObjectInputStream) {
    t = in.readObject().asInstanceOf[TStruct]

    val smallInOff = in.readInt()
    val a = new Array[Byte](smallInOff)
    in.readFully(a, 0, smallInOff)
    val dec = new Decoder(
      new LZ4InputBuffer(
        new ArrayInputStream(a)))

    region = MemoryBuffer()
    offset = dec.readRegionValue(t, region)
  }
}
