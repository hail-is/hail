package is.hail.annotations

import java.io.{ObjectInputStream, ObjectOutputStream}

import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import com.esotericsoftware.kryo.io.{Input, Output}
import is.hail.expr.types._
import is.hail.utils._
import is.hail.variant.{Locus, RGBase}
import org.apache.spark.sql.Row
import sun.reflect.generics.reflectiveObjects.NotImplementedException

trait UnKryoSerializable extends KryoSerializable {
  def write(kryo: Kryo, output: Output): Unit = {
    throw new NotImplementedException()
  }

  def read(kryo: Kryo, input: Input): Unit = {
    throw new NotImplementedException()
  }
}

object UnsafeIndexedSeq {
  def apply(t: TArray, elements: Array[RegionValue]): UnsafeIndexedSeq = {
    val region = Region()
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
    val region = Region()
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
    val region = Region()
    val rvb = new RegionValueBuilder(region)
    rvb.start(t)
    rvb.startArray(0)
    rvb.endArray()
    new UnsafeIndexedSeq(t, region, rvb.end())
  }
}

class UnsafeIndexedSeq(
  var t: TContainer,
  var region: Region, var aoff: Long) extends IndexedSeq[Annotation] with UnKryoSerializable {

  var length: Int = t.loadLength(region, aoff)

  def apply(i: Int): Annotation = {
    if (i < 0 || i >= length)
      throw new IndexOutOfBoundsException(i.toString)
    if (t.isElementDefined(region, aoff, i)) {
      UnsafeRow.read(t.elementType, region, t.loadElement(region, aoff, length, i))
    } else
      null
  }
}

object UnsafeRow {
  def readBinary(region: Region, boff: Long): Array[Byte] = {
    val binLength = TBinary.loadLength(region, boff)
    region.loadBytes(TBinary.bytesOffset(boff), binLength)
  }

  def readArray(t: TContainer, region: Region, aoff: Long): IndexedSeq[Any] =
    new UnsafeIndexedSeq(t, region, aoff)

  def readBaseStruct(t: TBaseStruct, region: Region, offset: Long): UnsafeRow =
    new UnsafeRow(t, region, offset)

  def readString(region: Region, boff: Long): String =
    new String(readBinary(region, boff))

  def readLocus(region: Region, offset: Long, rg: RGBase): Locus = {
    val ft = rg.locusType.fundamentalType.asInstanceOf[TStruct]
    Locus(
      readString(region, ft.loadField(region, offset, 0)),
      region.loadInt(ft.loadField(region, offset, 1)))
  }

  def read(t: Type, region: Region, offset: Long): Any = {
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
      case t: TBaseStruct => readBaseStruct(t, region, offset)
      case x: TLocus => readLocus(region, offset, x.rg)
      case x: TInterval =>
        val ft = x.fundamentalType.asInstanceOf[TStruct]
        val start: Annotation =
          if (ft.isFieldDefined(region, offset, 0))
            read(x.pointType, region, ft.loadField(region, offset, 0))
          else
            null
        val end =
          if (ft.isFieldDefined(region, offset, 1))
            read(x.pointType, region, ft.loadField(region, offset, 1))
          else
            null
        val includesStart = read(TBooleanRequired, region, ft.loadField(region, offset, 2)).asInstanceOf[Boolean]
        val includesEnd = read(TBooleanRequired, region, ft.loadField(region, offset, 3)).asInstanceOf[Boolean]
        Interval(start, end, includesStart, includesEnd)
    }
  }
}

class UnsafeRow(var t: TBaseStruct,
  var region: Region, var offset: Long) extends Row with UnKryoSerializable {

  def this(t: TBaseStruct, rv: RegionValue) = this(t, rv.region, rv.offset)

  def this(t: TBaseStruct) = this(t, null, 0)

  def this() = this(null, null, 0)

  def set(newRegion: Region, newOffset: Long) {
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
      UnsafeRow.read(t.types(i), region, t.loadField(region, offset, i))
  }

  def copy(): Row = new UnsafeRow(t, region, offset)

  def pretty(): String = region.pretty(t, offset)

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

  private def writeObject(s: ObjectOutputStream): Unit = {
    throw new NotImplementedException()
  }

  private def readObject(s: ObjectInputStream): Unit = {
    throw new NotImplementedException()
  }
}

object SafeRow {
  def apply(t: TBaseStruct, region: Region, off: Long): Row = {
    Annotation.copy(t, new UnsafeRow(t, region, off)).asInstanceOf[Row]
  }

  def apply(t: TBaseStruct, rv: RegionValue): Row = SafeRow(t, rv.region, rv.offset)

  def selectFields(t: TBaseStruct, region: Region, off: Long)(selectIdx: Array[Int]): Row = {
    val fullRow = new UnsafeRow(t, region, off)
    Row(selectIdx.map(i => Annotation.copy(t.types(i), fullRow.get(i))): _*)
  }

  def selectFields(t: TBaseStruct, rv: RegionValue)(selectIdx: Array[Int]): Row =
    SafeRow.selectFields(t, rv.region, rv.offset)(selectIdx)

  def read(t: Type, region: Region, off: Long): Annotation =
    Annotation.copy(t, UnsafeRow.read(t, region, off))

  def read(t: Type, rv: RegionValue): Annotation =
    read(t, rv.region, rv.offset)
}

object SafeIndexedSeq {
  def apply(t: TArray, region: Region, off: Long): IndexedSeq[Annotation] =
    Annotation.copy(t, new UnsafeIndexedSeq(t, region, off))
      .asInstanceOf[IndexedSeq[Annotation]]

  def apply(t: TArray, rv: RegionValue): IndexedSeq[Annotation] =
    apply(t, rv.region, rv.offset)
}

class KeyedRow(var row: Row, keyFields: Array[Int]) extends Row {
  def length: Int = row.size
  def get(i: Int): Any = row.get(keyFields(i))
  def copy(): Row = new KeyedRow(row, keyFields)
  def set(newRow: Row): KeyedRow = {
    row = newRow
    this
  }
}
