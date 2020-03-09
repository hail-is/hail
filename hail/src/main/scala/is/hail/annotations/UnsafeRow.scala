package is.hail.annotations

import java.io.{ObjectInputStream, ObjectOutputStream}

import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import com.esotericsoftware.kryo.io.{Input, Output}
import is.hail.expr.types.physical._
import is.hail.utils._
import is.hail.variant.Locus
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

class UnsafeIndexedSeq(
  var t: PContainer,
  var region: Region, var aoff: Long) extends IndexedSeq[Annotation] with UnKryoSerializable {

  var length: Int = t.loadLength(aoff)

  def apply(i: Int): Annotation = {
    if (i < 0 || i >= length)
      throw new IndexOutOfBoundsException(i.toString)
    if (t.isElementDefined(aoff, i)) {
      UnsafeRow.read(t.elementType, region, t.loadElement(aoff, length, i))
    } else
      null
  }

  override def toString: String = s"[${this.mkString(",")}]"
}

object UnsafeRow {
  def readBinary(region: Region, boff: Long, t: PBinary): Array[Byte] =
    t.loadBytes(boff)

  def readArray(t: PContainer, region: Region, aoff: Long): IndexedSeq[Any] =
    new UnsafeIndexedSeq(t, region, aoff)

  def readBaseStruct(t: PBaseStruct, region: Region, offset: Long): UnsafeRow =
    new UnsafeRow(t, region, offset)

  def readString(region: Region, boff: Long, t: PString): String =
    new String(readBinary(region, boff, t.fundamentalType))

  def readLocus(region: Region, offset: Long, t: PLocus): Locus = {
    val ft = t.representation.asInstanceOf[PStruct]
    Locus(
      readString(region, ft.loadField(offset, 0), t.contigType),
      Region.loadInt(ft.loadField(offset, 1)))
  }

  def readAnyRef(t: PType, region: Region, offset: Long): AnyRef = read(t, region, offset).asInstanceOf[AnyRef]

  def read(t: PType, region: Region, offset: Long): Any = {
    t match {
      case _: PBoolean =>
        Region.loadBoolean(offset)
      case _: PInt32 | _: PCall => Region.loadInt(offset)
      case _: PInt64 => Region.loadLong(offset)
      case _: PFloat32 => Region.loadFloat(offset)
      case _: PFloat64 => Region.loadDouble(offset)
      case t: PArray =>
        readArray(t, region, offset)
      case t: PSet =>
        readArray(t, region, offset).toSet
      case t: PString => readString(region, offset, t)
      case t: PBinary => readBinary(region, offset, t)
      case td: PDict =>
        val a = readArray(td, region, offset)
        a.asInstanceOf[IndexedSeq[Row]].map(r => (r.get(0), r.get(1))).toMap
      case t: PBaseStruct => readBaseStruct(t, region, offset)
      case x: PLocus => readLocus(region, offset, x)
      case x: PInterval =>
        val start: Annotation =
          if (x.startDefined(offset))
            read(x.pointType, region, x.loadStart(offset))
          else
            null
        val end =
          if (x.endDefined(offset))
            read(x.pointType, region, x.loadEnd(offset))
          else
            null
        val includesStart = x.includesStart(offset)
        val includesEnd = x.includesEnd(offset)
        Interval(start, end, includesStart, includesEnd)
      case nd: PNDArray => UnsafeRow.read(nd.representation, region, offset)
    }
  }
}

class UnsafeRow(var t: PBaseStruct,
  var region: Region, var offset: Long) extends Row with UnKryoSerializable {

  override def toString: String = {
    if (t.isInstanceOf[PStruct]) {
      val sb = new StringBuilder()
      var i = 0
      sb += '{'
      while (i < t.size) {
        if (i != 0) {
          sb ++= ", "
        }
        sb ++= t.fieldNames(i)
        sb ++= ": "
        val x = get(i)
        sb ++= (if (x == null) "null" else x.toString())
        i += 1
      }
      sb += '}'
      sb.toString
    } else if (t.isInstanceOf[PTuple]) {
      val sb = new StringBuilder()
      var i = 0
      sb += '('
      while (i < t.size) {
        if (i != 0) {
          sb ++= ", "
        }
        val x = get(i)
        sb ++= (if (x == null) "null" else x.toString())
        i += 1
      }
      sb += ')'
      sb.toString
    } else {
      super.toString
    }
  }

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
      UnsafeRow.read(t.types(i), region, t.loadField(offset, i))
  }

  def copy(): Row = new UnsafeRow(t, region, offset)

  def pretty(): String = Region.pretty(t, offset)

  override def getInt(i: Int): Int = {
    assertDefined(i)
    Region.loadInt(t.loadField(offset, i))
  }

  override def getLong(i: Int): Long = {
    assertDefined(i)
    Region.loadLong(t.loadField(offset, i))
  }

  override def getFloat(i: Int): Float = {
    assertDefined(i)
    Region.loadFloat(t.loadField(offset, i))
  }

  override def getDouble(i: Int): Double = {
    assertDefined(i)
    Region.loadDouble(t.loadField(offset, i))
  }

  override def getBoolean(i: Int): Boolean = {
    assertDefined(i)
    Region.loadBoolean(t.loadField(offset, i))
  }

  override def getByte(i: Int): Byte = {
    assertDefined(i)
    Region.loadByte(t.loadField(offset, i))
  }

  override def isNullAt(i: Int): Boolean = {
    if (i < 0 || i >= t.size)
      throw new IndexOutOfBoundsException(i.toString)
    !t.isFieldDefined(offset, i)
  }

  private def writeObject(s: ObjectOutputStream): Unit = {
    throw new NotImplementedException()
  }

  private def readObject(s: ObjectInputStream): Unit = {
    throw new NotImplementedException()
  }
}

object SafeRow {
  def apply(t: PBaseStruct, off: Long): Row = {
    Annotation.copy(t.virtualType, new UnsafeRow(t, null, off)).asInstanceOf[Row]
  }

  def apply(t: PBaseStruct, rv: RegionValue): Row = SafeRow(t, rv.offset)

  def selectFields(t: PBaseStruct, region: Region, off: Long)(selectIdx: Array[Int]): Row = {
    val fullRow = new UnsafeRow(t, region, off)
    Row.fromSeq(selectIdx.map(i => Annotation.copy(t.types(i).virtualType, fullRow.get(i))))
  }

  def selectFields(t: PBaseStruct, rv: RegionValue)(selectIdx: Array[Int]): Row =
    SafeRow.selectFields(t, rv.region, rv.offset)(selectIdx)

  def read(t: PType, off: Long): Annotation =
    Annotation.copy(t.virtualType, UnsafeRow.read(t, null, off))

  def read(t: PType, rv: RegionValue): Annotation =
    read(t, rv.offset)
}

object SafeIndexedSeq {
  def apply(t: PArray, off: Long): IndexedSeq[Annotation] =
    Annotation.copy(t.virtualType, new UnsafeIndexedSeq(t, null, off))
      .asInstanceOf[IndexedSeq[Annotation]]

  def apply(t: PArray, rv: RegionValue): IndexedSeq[Annotation] =
    apply(t, rv.offset)
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
