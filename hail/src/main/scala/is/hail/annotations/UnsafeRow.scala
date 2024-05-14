package is.hail.annotations

import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.Locus

import java.io.{ObjectInputStream, ObjectOutputStream}

import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import com.esotericsoftware.kryo.io.{Input, Output}
import org.apache.spark.sql.Row
import sun.reflect.generics.reflectiveObjects.NotImplementedException

trait UnKryoSerializable extends KryoSerializable {
  def write(kryo: Kryo, output: Output): Unit =
    throw new NotImplementedException()

  def read(kryo: Kryo, input: Input): Unit =
    throw new NotImplementedException()
}

class UnsafeIndexedSeq(
  val t: PContainer,
  val region: Region,
  val aoff: Long,
) extends IndexedSeq[Annotation] with UnKryoSerializable {

  val length: Int = t.loadLength(aoff)

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
  def readBinary(boff: Long, t: PBinary): Array[Byte] =
    t.loadBytes(boff)

  def readArray(t: PContainer, region: Region, aoff: Long): IndexedSeq[Any] =
    new UnsafeIndexedSeq(t, region, aoff)

  def readBaseStruct(t: PBaseStruct, region: Region, offset: Long): UnsafeRow =
    new UnsafeRow(t, region, offset)

  def readString(boff: Long, t: PString): String =
    new String(readBinary(boff, t.binaryRepresentation))

  def readLocus(offset: Long, t: PLocus): Locus =
    Locus(
      t.contig(offset),
      t.position(offset),
    )

  def readNDArray(offset: Long, region: Region, nd: PNDArray): UnsafeNDArray =
    new UnsafeNDArray(nd, region, offset)

  def readAnyRef(t: PType, region: Region, offset: Long): AnyRef =
    read(t, region, offset).asInstanceOf[AnyRef]

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
      case t: PString => readString(offset, t)
      case t: PBinary => readBinary(offset, t)
      case td: PDict =>
        val a = readArray(td, region, offset)
        a.asInstanceOf[IndexedSeq[Row]].map(r => (r.get(0), r.get(1))).toMap
      case t: PBaseStruct => readBaseStruct(t, region, offset)
      case x: PLocus => readLocus(offset, x)
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
      case nd: PNDArray =>
        readNDArray(offset, region, nd)
    }
  }
}

class UnsafeRow(val t: PBaseStruct, var region: Region, var offset: Long)
    extends Row with UnKryoSerializable {

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

  def set(newRegion: Region, newOffset: Long): Unit = {
    region = newRegion
    offset = newOffset
  }

  def set(rv: RegionValue): Unit = set(rv.region, rv.offset)

  def length: Int = t.size

  private def assertDefined(i: Int): Unit =
    if (isNullAt(i))
      throw new NullPointerException(s"null value at index $i")

  def get(i: Int): Any =
    if (isNullAt(i))
      null
    else
      UnsafeRow.read(t.types(i), region, t.loadField(offset, i))

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

  private def writeObject(s: ObjectOutputStream): Unit =
    throw new NotImplementedException()

  private def readObject(s: ObjectInputStream): Unit =
    throw new NotImplementedException()
}

object SafeRow {
  def apply(t: PBaseStruct, off: Long): Row =
    Annotation.copy(t.virtualType, new UnsafeRow(t, null, off)).asInstanceOf[Row]

  def apply(t: PBaseStruct, rv: RegionValue): Row = SafeRow(t, rv.offset)

  def selectFields(t: PBaseStruct, region: Region, off: Long)(selectIdx: Array[Int]): Row = {
    val fullRow = new UnsafeRow(t, region, off)
    Row.fromSeq(selectIdx.map(i => Annotation.copy(t.types(i).virtualType, fullRow.get(i))))
  }

  def selectFields(t: PBaseStruct, rv: RegionValue)(selectIdx: Array[Int]): Row =
    SafeRow.selectFields(t, rv.region, rv.offset)(selectIdx)

  def read(t: PType, off: Long): Annotation =
    Annotation.copy(t.virtualType, UnsafeRow.read(t, null, off))

  def readAnyRef(t: PType, region: Region, offset: Long): AnyRef =
    read(t, offset).asInstanceOf[AnyRef]

  def read(t: PType, rv: RegionValue): Annotation =
    read(t, rv.offset)

  def isSafe(a: Any): Boolean = {
    a match {
      case _: UnsafeRow => false
      case _: UnsafeIndexedSeq => false
      case _: UnsafeNDArray => false

      case r: Row =>
        r.toSeq.forall(isSafe)
      case a: IndexedSeq[_] =>
        a.forall(isSafe)
      case i: Interval =>
        isSafe(i.start) && isSafe(i.end)
      case nd: NDArray =>
        nd.getRowMajorElements().forall(isSafe)

      case _ => true
    }
  }
}

object SafeIndexedSeq {
  def apply(t: PArray, off: Long): IndexedSeq[Annotation] =
    Annotation.copy(t.virtualType, new UnsafeIndexedSeq(t, null, off))
      .asInstanceOf[IndexedSeq[Annotation]]

  def apply(t: PArray, rv: RegionValue): IndexedSeq[Annotation] =
    apply(t, rv.offset)
}

class SelectFieldsRow(
  private[this] var old: Row,
  private[this] val fieldMapping: Array[Int],
) extends Row {
  def this(
    old: Row,
    oldPType: TStruct,
    newPType: TStruct,
  ) = this(old, newPType.fieldNames.map(name => oldPType.fieldIdx(name)))

  def this(
    old: Row,
    oldPType: PStruct,
    newPType: PStruct,
  ) =
    this(
      old,
      (require(
        oldPType.fields.length <= old.length &&
          newPType.fields.length <= old.length,
        s"$oldPType, $newPType ${old.length} $old",
      )
        ->
          newPType.fieldNames.map(name => oldPType.fieldIdx(name)))._2,
    )

  require(fieldMapping.forall(x => x < old.length), s"${fieldMapping.toSeq}, ${old.length} $old")

  override def length = fieldMapping.length
  override def get(i: Int) = old.get(fieldMapping(i))
  override def isNullAt(i: Int) = old.isNullAt(fieldMapping(i))
  override def copy(): Row = new SelectFieldsRow(old.copy(), fieldMapping)

  def set(newRow: Row): SelectFieldsRow = {
    old = newRow
    this
  }
}

trait NDArray {
  val shape: IndexedSeq[Long]
  lazy val numElements = shape.foldLeft(1L)(_ * _)
  def lookupElement(indices: IndexedSeq[Long]): Annotation
  def getRowMajorElements(): IndexedSeq[Annotation]
  def forall(pred: Annotation => Boolean): Boolean

  override def equals(that: Any): Boolean = {
    if (that.isInstanceOf[NDArray]) {
      val thatNd = that.asInstanceOf[NDArray]
      this.shape == thatNd.shape && this.getRowMajorElements() == thatNd.getRowMajorElements()
    } else {
      false
    }
  }
}

class UnsafeNDArray(val pnd: PNDArray, val region: Region, val ndAddr: Long) extends NDArray {
  val shape: IndexedSeq[Long] = (0 until pnd.nDims).map(i => pnd.loadShape(ndAddr, i))
  val elementType = pnd.elementType.virtualType
  val coordStorageArray = new Array[Long](shape.size)

  def lookupElement(indices: IndexedSeq[Long]): Annotation = {
    val elementAddress = pnd.getElementAddress(indices, ndAddr)
    UnsafeRow.read(pnd.elementType, region, pnd.elementType.unstagedLoadFromNested(elementAddress))
  }

  def getRowMajorElements(): IndexedSeq[Annotation] = {
    val indices = (0 until pnd.nDims).map(_ => 0L).toArray
    var curIdx = indices.size - 1
    var idxIntoFlat = 0
    val flat = new Array[Annotation](numElements.toInt)

    if (numElements > Int.MaxValue) {
      throw new IllegalArgumentException(
        s"Cannot make an UnsafeNDArray with greater than Int.MaxValue entries. Shape was $shape"
      )
    }

    while (idxIntoFlat < numElements) {
      flat(idxIntoFlat) = lookupElement(indices)
      while (curIdx >= 0L && indices(curIdx) >= shape(curIdx) - 1) {
        indices(curIdx) = 0L
        curIdx -= 1
      }
      // found the index that needs incrementing, so long as we haven't run out of room
      if (curIdx >= 0) {
        indices(curIdx) += 1L
        curIdx = indices.size - 1
      }
      idxIntoFlat += 1
    }

    flat
  }

  override def forall(pred: Annotation => Boolean): Boolean = {
    val indices = (0 until pnd.nDims).map(_ => 0L).toArray
    var curIdx = indices.size - 1
    var idxIntoFlat = 0

    while (idxIntoFlat < numElements) {
      if (!pred(lookupElement(indices))) {
        return false
      }

      while (curIdx >= 0L && indices(curIdx) >= shape(curIdx) - 1) {
        indices(curIdx) = 0L
        curIdx -= 1
      }
      // found the index that needs incrementing, so long as we haven't run out of room
      if (curIdx >= 0) {
        indices(curIdx) += 1L
        curIdx = indices.size - 1
      }
      idxIntoFlat += 1
    }

    true
  }

  override def toString: String =
    s"UnsafeNDArray of shape (${shape.mkString(", ")}) with elements ${getRowMajorElements()}"
}

case class SafeNDArray(val shape: IndexedSeq[Long], rowMajorElements: IndexedSeq[Annotation])
    extends NDArray {
  assert(shape.foldLeft(1L)(_ * _) == rowMajorElements.size)
  override def getRowMajorElements: IndexedSeq[Annotation] = rowMajorElements

  override def lookupElement(indices: IndexedSeq[Long]): Annotation = {
    val flatIdx = indices.zip(shape).foldLeft(0L) { case (flatIdx, (index, shape)) =>
      flatIdx + index * shape
    }
    rowMajorElements(flatIdx.toInt)
  }

  override def forall(pred: Annotation => Boolean): Boolean = this.rowMajorElements.forall(pred)
}
