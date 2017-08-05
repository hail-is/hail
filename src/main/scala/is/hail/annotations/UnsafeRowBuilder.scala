package is.hail.annotations

import is.hail.expr._
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.sql.Row

class UnsafeRowBuilder(t: TStruct, sizeHint: Int = 128) {
  private var buffer: MemoryBuffer = MemoryBuffer(sizeHint)
  buffer.allocate(t.byteSize)

  def setAll(r: Row) {
    require(r != null, "cannot convert null row")

    r match {
      case ur: UnsafeRow =>
        var i = 0
        while (i < t.size) {
          setFromUnsafe(i, i, ur)
          i += 1
        }
      case _ =>
        var i = 0
        while (i < r.size) {
          setFromRow(i, i, r)
          i += 1
        }
    }
  }

  def clear() {
    buffer.clear()
    buffer.allocate(t.byteSize)
  }

  def result(sc: SparkContext): UnsafeRow = {
    new UnsafeRow(BroadcastTypeTree(sc, t), buffer, 0)
  }

  def setFromUnsafe(index: Int, oldIndex: Int, u: UnsafeRow) {
    val fieldType = t.fields(index).typ
    assert(fieldType == u.t.fields(oldIndex).typ)

    val o1 = t.byteOffsets(index)
    if (u.isNullAt(oldIndex)) {
      setMissing(index)
    } else {
      copyFromUnsafe(fieldType, o1, o1 + u.offset, u.region)
    }
  }

  def setFromRow(index: Int, oldIndex: Int, r: Row) {
    if (r.isNullAt(oldIndex))
      setMissing(index)
    else
      put(r.get(oldIndex), t.byteOffsets(index), t.fields(index).typ)
  }

  private def putBinary(value: Array[Byte], offset: Int) {
    assert(offset % 4 == 0, s"invalid binary offset: $offset")

    buffer.align(4)
    buffer.storeInt(offset, buffer.offset)
    buffer.appendInt(value.length)
    buffer.appendBytes(value)
  }

  private def putArray(value: Iterable[_], offset: Int, elementType: Type) {
    assert(offset % 4 == 0, s"invalid array offset: $offset")

    val eltSize = UnsafeUtils.arrayElementSize(elementType)

    buffer.align(4)
    buffer.storeInt(offset, buffer.offset)

    val nElements = value.size
    buffer.appendInt(nElements)

    val nMissingBytes = (value.size + 7) / 8
    val missingBytesStart = buffer.allocate(nMissingBytes)

    var i = 0
    while (i < nMissingBytes) {
      buffer.storeByte(i + missingBytesStart, 0)
      i += 1
    }

    buffer.align(elementType.alignment)
    val elementStart = buffer.allocate(nElements * eltSize)

    i = 0
    value.foreach { elt =>
      if (elt == null) {
        val byteIndex = missingBytesStart + (i / 8)
        val shift = i & 0x7
        val oldByte = buffer.loadByte(byteIndex)
        buffer.storeByte(byteIndex, (oldByte | (0x1 << (i & 0x7))).toByte)
      } else {
        put(elt, i * eltSize + elementStart, elementType)
      }
      i += 1
    }
  }

  private def putStruct(value: Row, offset: Int, struct: TStruct) {
    var i = 0
    val nMissingBytes = (struct.size + 7) / 8
    while (i < nMissingBytes) {
      buffer.storeByte(offset + i, 0)
      i += 1
    }
    i = 0
    while (i < struct.size) {
      if (value.isNullAt(i)) {
        val byteIndex = offset + (i / 8)
        val shift = i & 0x7
        val oldByte = buffer.loadByte(byteIndex)
        buffer.storeByte(byteIndex, (oldByte | (0x1 << (i & 0x7))).toByte)
      } else {
        put(value.get(i), struct.byteOffsets(i) + offset, struct.fields(i).typ)
      }
      i += 1
    }
  }

  private def put(value: Annotation, offset: Int, elementType: Type) {
    assert(value != null, s"got a null value of type ${ elementType } at offset $offset")
    elementType match {
      case TInt32 | TCall => buffer.storeInt(offset, value.asInstanceOf[Int])
      case TInt64 => buffer.storeLong(offset, value.asInstanceOf[Long])
      case TFloat32 => buffer.storeFloat(offset, value.asInstanceOf[Float])
      case TFloat64 => buffer.storeDouble(offset, value.asInstanceOf[Double])
      case TBoolean => buffer.storeByte(offset, value.asInstanceOf[Boolean].toByte)
      case TString => putBinary(value.asInstanceOf[String].getBytes(), offset)
      case t: TArray => putArray(value.asInstanceOf[Iterable[_]], offset, t.elementType)
      case t: TSet => putArray(value.asInstanceOf[Iterable[_]], offset, t.elementType)
      case t: TDict =>
        val m = value.asInstanceOf[Map[Any, Any]]
        val arr = m.keys.view.map(k => Row(k, m(k)))
        putArray(arr, offset, t.elementType)
      case struct: TStruct =>
        putStruct(value.asInstanceOf[Row], offset, struct)
      case TAltAllele | TVariant | TGenotype | TLocus | TInterval =>
        val expandedType = Annotation.expandType(elementType).asInstanceOf[TStruct]
        val expandedAnnotation = Annotation.expandAnnotation(value, elementType).asInstanceOf[Row]
        putStruct(expandedAnnotation, offset, expandedType)
    }
  }

  private def setMissing(index: Int) {
    val byteIndex = index >> 3
    val bitIndex = index & 0x7

    val oldBits = buffer.loadByte(byteIndex)
    buffer.storeByte(byteIndex, (oldBits | (0x1 << bitIndex)).toByte)
  }

  private def copyStructFromUnsafe(st: TStruct, o1: Int, o2: Int, mb: MemoryBuffer) {
    val missingBytes = (st.size + 7) / 8
    buffer.copyFrom(mb, o1, o2, missingBytes)
    var i = 0
    var b = 0
    while (i < st.size) {
      if ((i & 0x7) == 0)
        b = mb.loadByte((i >> 3) + o2)
      val isMissing = (b & (0x1 << (i & 0x7))) != 0
      if (!isMissing) {
        val offset = st.byteOffsets(i)
        val fieldType = st.fields(i).typ
        copyFromUnsafe(fieldType, o1 + offset, o2 + offset, mb)
      }
      i += 1
    }
  }

  private def copyArrayFromMem(et: Type, o1: Int, o2: Int, mb: MemoryBuffer) {
    buffer.align(4)
    val leftStart = buffer.allocate(4)
    buffer.storeInt(o1, leftStart)

    val rightStart = mb.loadInt(o2)
    val arrSize = mb.loadInt(rightStart)
    buffer.storeInt(leftStart, arrSize)
    val nMissingBytes = (arrSize + 7) / 8
    buffer.allocate(nMissingBytes)
    buffer.copyFrom(mb, leftStart + 4, rightStart + 4, nMissingBytes)
    buffer.align(et.alignment)
    val arrElementSize = UnsafeUtils.arrayElementSize(et)
    val arrStart = buffer.allocate(arrSize * arrElementSize)
    val rightArrStart = UnsafeUtils.roundUpAlignment(rightStart + 4 + nMissingBytes, et.alignment)

    assert(arrSize >= 0)
    var i = 0
    var b = 0
    while (i < arrSize) {
      if ((i & 0x7) == 0)
        b = mb.loadByte(leftStart + 4 + (i >> 3))

      val isMissing = (b & (0x1 << (i & 0x7))) != 0
      if (!isMissing) {
        val elementOffset = i * arrElementSize
        copyFromUnsafe(et, arrStart + elementOffset, rightArrStart + elementOffset, mb)
      }
      i += 1
    }
  }

  private def copyFromUnsafe(ft: Type, o1: Int, o2: Int, mb: MemoryBuffer) {
    ft match {
      case TInt32 =>
        val value = mb.loadInt(o2)
        buffer.storeInt(o1, value)
      case TFloat32 =>
        val value = mb.loadFloat(o2)
        buffer.storeFloat(o1, value)
      case TInt64 =>
        val value = mb.loadLong(o2)
        buffer.storeLong(o1, value)
      case TFloat64 =>
        val value = mb.loadDouble(o2)
        buffer.storeDouble(o1, value)
      case TBoolean =>
        val value = mb.loadByte(o2)
        buffer.storeByte(o1, value)
      case TString =>
        val abs = mb.loadInt(o2)
        val size = mb.loadInt(abs)
        buffer.align(4)
        val start = buffer.allocate(4 + size)
        buffer.storeInt(o1, start)
        buffer.storeInt(start, size)
        buffer.copyFrom(mb, start + 4, abs + 4, size)
      case st: TStruct =>
        if (st.size > 0)
          copyStructFromUnsafe(st, o1, o2, mb)
      case ct: ComplexType => copyFromUnsafe(ct.representation, o1, o2, mb)
      case coll: TContainer => copyArrayFromMem(coll.elementType, o1, o2, mb)
    }
  }
}

