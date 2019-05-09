package is.hail.annotations

import java.io.{ObjectInputStream, ObjectOutputStream}

import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import com.esotericsoftware.kryo.io.{Input, Output}
import is.hail.expr.types.physical._
import is.hail.utils._
import is.hail.variant.{Locus, RGBase}
import org.apache.spark.sql.Row
import sun.reflect.generics.reflectiveObjects.NotImplementedException

import scala.collection.immutable.TreeMap

trait UnKryoSerializable extends KryoSerializable {
  def write(kryo: Kryo, output: Output): Unit = {
    throw new NotImplementedException()
  }

  def read(kryo: Kryo, input: Input): Unit = {
    throw new NotImplementedException()
  }
}

class UnsafeIndexedSeq(
  var t: PContainer,
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

  override def toString: String = s"[${this.mkString(",")}]"
}

object UnsafeRow {
  def readBinary(region: Region, boff: Long): Array[Byte] = {
    val binLength = PBinary.loadLength(region, boff)
    region.loadBytes(PBinary.bytesOffset(boff), binLength)
  }

  def readArray(t: PContainer, region: Region, aoff: Long): IndexedSeq[Any] =
    new UnsafeIndexedSeq(t, region, aoff)

  def readBaseStruct(t: PBaseStruct, region: Region, offset: Long): UnsafeRow =
    new UnsafeRow(t, region, offset)

  def readString(region: Region, boff: Long): String =
    new String(readBinary(region, boff))

  def readLocus(region: Region, offset: Long, rg: RGBase): Locus = {
    val ft = rg.locusType.physicalType.fundamentalType.asInstanceOf[PStruct]
    Locus(
      readString(region, ft.loadField(region, offset, 0)),
      region.loadInt(ft.loadField(region, offset, 1)))
  }

  def read(t: PType, region: Region, offset: Long): Any = {
    t match {
      case _: PBoolean =>
        region.loadBoolean(offset)
      case _: PInt32 | _: PCall => region.loadInt(offset)
      case _: PInt64 => region.loadLong(offset)
      case _: PFloat32 => region.loadFloat(offset)
      case _: PFloat64 => region.loadDouble(offset)
      case t: PArray =>
        readArray(t, region, offset)
      case t: PSet =>
        t.virtualType.toLiteral(readArray(t, region, offset))
      case _: PString => readString(region, offset)
      case _: PBinary => readBinary(region, offset)
      case td: PDict =>
        val a = readArray(td, region, offset)
        TreeMap(
          a.asInstanceOf[IndexedSeq[Row]].map(r => (r.get(0), r.get(1))): _*)(td.keyType.virtualType.ordering.toTotalOrdering)
      case t: PBaseStruct => readBaseStruct(t, region, offset)
      case x: PLocus => readLocus(region, offset, x.rg)
      case x: PInterval =>
        val start: Annotation =
          if (x.startDefined(region, offset))
            read(x.pointType, region, x.loadStart(region, offset))
          else
            null
        val end =
          if (x.endDefined(region, offset))
            read(x.pointType, region, x.loadEnd(region, offset))
          else
            null
        val includesStart = x.includesStart(region, offset)
        val includesEnd = x.includesEnd(region, offset)
        Interval(start, end, includesStart, includesEnd)
    }
  }
}

class UnsafeRow(var t: PBaseStruct,
  var region: Region, var offset: Long) extends Row with UnKryoSerializable {

  def this(t: PBaseStruct, rv: RegionValue) = this(t, rv.region, rv.offset)

  def this(t: PBaseStruct) = this(t, null, 0)

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
  def apply(t: PBaseStruct, region: Region, off: Long): Row = {
    Annotation.copy(t.virtualType, new UnsafeRow(t, region, off)).asInstanceOf[Row]
  }

  def apply(t: PBaseStruct, rv: RegionValue): Row = SafeRow(t, rv.region, rv.offset)

  def selectFields(t: PBaseStruct, region: Region, off: Long)(selectIdx: Array[Int]): Row = {
    val fullRow = new UnsafeRow(t, region, off)
    Row.fromSeq(selectIdx.map(i => Annotation.copy(t.types(i).virtualType, fullRow.get(i))))
  }

  def selectFields(t: PBaseStruct, rv: RegionValue)(selectIdx: Array[Int]): Row =
    SafeRow.selectFields(t, rv.region, rv.offset)(selectIdx)

  def read(t: PType, region: Region, off: Long): Annotation =
    Annotation.copy(t.virtualType, UnsafeRow.read(t, region, off))

  def read(t: PType, rv: RegionValue): Annotation =
    read(t, rv.region, rv.offset)
}

object SafeIndexedSeq {
  def apply(t: PArray, region: Region, off: Long): IndexedSeq[Annotation] =
    Annotation.copy(t.virtualType, new UnsafeIndexedSeq(t, region, off))
      .asInstanceOf[IndexedSeq[Annotation]]

  def apply(t: PArray, rv: RegionValue): IndexedSeq[Annotation] =
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
