package is.hail.annotations

import java.io.{ObjectInputStream, ObjectOutputStream}

import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import is.hail.expr._
import is.hail.utils._
import is.hail.variant.{AltAllele, Genotype, Locus, Variant}
import org.apache.spark.sql.Row

import scala.collection.mutable

object MemoryBuffer {
  def apply(sizeHint: Long = 128): MemoryBuffer = {
    new MemoryBuffer(Memory.malloc(sizeHint), sizeHint)
  }
}

final class MemoryBuffer(var mem: Long, var length: Long, var offset: Long = 0) extends KryoSerializable with Serializable {
  def size: Long = offset

  // TODO rename offset => end, length => capacity
  def end: Long = offset

  def copyFrom(other: MemoryBuffer, readStart: Long, writeStart: Long, n: Long) {
    assert(size <= length)
    assert(other.size <= other.length)
    assert(n >= 0)
    assert(readStart >= 0 && readStart + n <= other.size)
    assert(writeStart >= 0 && writeStart + n <= size)
    Memory.memcpy(mem + writeStart, other.mem + readStart, n)
  }

  def loadInt(off: Long): Int = {
    assert(size <= length)
    assert(off >= 0 && off + 4 <= size)
    Memory.loadInt(mem + off)
  }

  def loadLong(off: Long): Long = {
    assert(size <= length)
    assert(off >= 0 && off + 8 <= size)
    Memory.loadLong(mem + off)
  }

  def loadFloat(off: Long): Float = {
    assert(size <= length)
    assert(off >= 0 && off + 4 <= size)
    Memory.loadFloat(mem + off)
  }

  def loadDouble(off: Long): Double = {
    assert(size <= length)
    assert(off >= 0 && off + 8 <= size)
    Memory.loadDouble(mem + off)
  }

  def loadAddress(off: Long): Long = {
    assert(size <= length)
    assert(off >= 0 && off + 8 <= size)
    Memory.loadAddress(mem + off)
  }

  def loadByte(off: Long): Byte = {
    assert(size <= length)
    assert(off >= 0 && off + 1 <= size)
    Memory.loadByte(mem + off)
  }

  def loadBytes(off: Long, n: Int): Array[Byte] = {
    assert(size <= length)
    assert(off >= 0 && off + n <= size)
    val a = new Array[Byte](n)
    Memory.copyToArray(a, 0, mem + off, n)
    a
  }

  def loadBytes(off: Long, n: Long, dst: Array[Byte]) {
    assert(size <= length)
    assert(off >= 0 && off + n <= size)
    assert(n <= dst.length)
    Memory.copyToArray(dst, 0, mem + off, n)
  }

  def storeInt(off: Long, i: Int) {
    assert(size <= length)
    assert(off >= 0 && off + 4 <= size)
    Memory.storeInt(mem + off, i)
  }

  def storeLong(off: Long, l: Long) {
    assert(size <= length)
    assert(off >= 0 && off + 8 <= size)
    Memory.storeLong(mem + off, l)
  }

  def storeFloat(off: Long, f: Float) {
    assert(size <= length)
    assert(off >= 0 && off + 4 <= size)
    Memory.storeFloat(mem + off, f)
  }

  def storeDouble(off: Long, d: Double) {
    assert(size <= length)
    assert(off >= 0 && off + 8 <= size)
    Memory.storeDouble(mem + off, d)
  }

  def storeAddress(off: Long, a: Long) {
    assert(size <= length)
    assert(off >= 0 && off + 8 <= size)
    Memory.storeAddress(mem + off, a)
  }

  def storeByte(off: Long, b: Byte) {
    assert(size <= length)
    assert(off >= 0 && off + 1 <= size)
    Memory.storeByte(mem + off, b)
  }

  def storeBytes(off: Long, bytes: Array[Byte]) {
    storeBytes(off, bytes, 0, bytes.length)
  }

  def storeBytes(off: Long, bytes: Array[Byte], bytesOff: Long, n: Int) {
    assert(size <= length)
    assert(off >= 0 && off + n <= size)
    assert(bytesOff + n <= bytes.length)
    Memory.copyFromArray(mem + off, bytes, bytesOff, n)
  }

  def ensure(n: Long) {
    val required = size + n
    if (length < required) {
      val newLength = (length * 2).max(required)
      mem = Memory.realloc(mem, newLength)
      length = newLength
    }
  }

  def align(alignment: Long) {
    assert(alignment > 0)
    assert((alignment & (alignment - 1)) == 0) // power of 2
    offset = (offset + (alignment - 1)) & ~(alignment - 1)
  }

  def allocate(n: Long): Long = {
    assert(n >= 0)
    val off = offset
    ensure(n)
    offset += n
    off
  }

  def alignAndAllocate(n: Long): Long = {
    align(n)
    allocate(n)
  }

  def loadBoolean(off: Long): Boolean = {
    val b = loadByte(off)
    assert(b == 1 || b == 0)
    b != 0
  }

  def loadBit(byteOff: Long, bitOff: Long): Boolean = {
    val b = byteOff + (bitOff >> 3)
    (loadByte(b) & (1 << (bitOff & 7))) != 0
  }

  def setBit(byteOff: Long, bitOff: Long) {
    val b = byteOff + (bitOff >> 3)
    storeByte(b,
      (loadByte(b) | (1 << (bitOff & 7))).toByte)
  }

  def clearBit(byteOff: Long, bitOff: Long) {
    val b = byteOff + (bitOff >> 3)
    storeByte(b,
      (loadByte(b) & ~(1 << (bitOff & 7))).toByte)
  }

  def storeBit(byteOff: Long, bitOff: Long, b: Boolean) {
    if (b)
      setBit(byteOff, bitOff)
    else
      clearBit(byteOff, bitOff)
  }

  def appendInt(i: Int) {
    storeInt(alignAndAllocate(4), i)
  }

  def appendLong(l: Long) {
    storeLong(alignAndAllocate(8), l)
  }

  def appendFloat(f: Float) {
    storeFloat(alignAndAllocate(4), f)
  }

  def appendDouble(d: Double) {
    storeDouble(alignAndAllocate(8), d)
  }

  def appendByte(b: Byte) {
    storeByte(allocate(1), b)
  }

  def appendBytes(bytes: Array[Byte]) {
    storeBytes(allocate(bytes.length), bytes)
  }

  def appendBytes(bytes: Array[Byte], bytesOff: Long, n: Int) {
    assert(bytesOff + n <= bytes.length)
    storeBytes(allocate(n), bytes, bytesOff, n)
  }

  def clear(newEnd: Long) {
    assert(newEnd <= end)
    offset = newEnd
  }

  def clear() {
    offset = 0
  }

  def setFrom(from: MemoryBuffer) {
    if (from.offset > length) {
      Memory.free(mem)

      val newLength = math.max((length * 3) / 2, from.offset)
      mem = Memory.malloc(newLength)
      length = newLength
    }
    Memory.memcpy(mem, from.mem, from.offset)
    offset = from.offset
  }

  def copy(): MemoryBuffer = {
    val newMem = Memory.malloc(offset)
    Memory.memcpy(newMem, mem, offset)
    new MemoryBuffer(newMem, offset, offset)
  }

  override def finalize() {
    Memory.free(mem)
  }

  override def write(kryo: Kryo, output: Output) {
    output.writeLong(offset)

    assert(offset <= Int.MaxValue)
    val smallOffset = offset.toInt
    val a = new Array[Byte](smallOffset)

    Memory.memcpy(a, 0, mem, offset)

    output.write(a)
  }

  override def read(kryo: Kryo, input: Input) {
    offset = input.readLong()
    assert(offset <= Int.MaxValue)
    val smallOffset = offset.toInt
    val inMem = new Array[Byte](smallOffset)
    input.read(inMem)

    mem = Memory.malloc(offset)
    length = offset

    Memory.memcpy(mem, inMem, 0, offset)
  }

  private def writeObject(out: ObjectOutputStream) {
    out.writeLong(offset)

    assert(offset <= Int.MaxValue)
    val smallOffset = offset.toInt
    val a = new Array[Byte](smallOffset)

    Memory.memcpy(a, 0, mem, offset)

    out.write(a)
  }

  private def readObject(in: ObjectInputStream) {
    offset = in.readLong()
    assert(offset <= Int.MaxValue)
    val smallOffset = offset.toInt
    val inMem = new Array[Byte](smallOffset)
    in.read(inMem)

    mem = Memory.malloc(offset)
    length = offset

    Memory.memcpy(mem, inMem, 0, offset)
  }

  def visit(t: Type, off: Long, v: ValueVisitor) {
    t match {
      case _: TBoolean => v.visitBoolean(loadBoolean(off))
      case _: TInt32 => v.visitInt32(loadInt(off))
      case _: TInt64 => v.visitInt64(loadLong(off))
      case _: TFloat32 => v.visitFloat32(loadFloat(off))
      case _: TFloat64 => v.visitFloat64(loadDouble(off))
      case _: TString =>
        val boff = off
        v.visitString(TString.loadString(this, boff))
      case _: TBinary =>
        val boff = off
        val length = TBinary.loadLength(this, boff)
        val b = loadBytes(TBinary.bytesOffset(boff), length)
        v.visitBinary(b)
      case t: TContainer =>
        val aoff = off
        val length = t.loadLength(this, aoff)
        v.enterArray(t, length)
        var i = 0
        while (i < length) {
          v.enterElement(i)
          if (t.isElementDefined(this, aoff, i))
            visit(t.elementType, t.loadElement(this, aoff, length, i), v)
          else
            v.visitMissing(t.elementType)
          i += 1
        }
        v.leaveArray()
      case t: TStruct =>
        v.enterStruct(t)
        var i = 0
        while (i < t.size) {
          val f = t.fields(i)
          v.enterField(f)
          if (t.isFieldDefined(this, off, i))
            visit(f.typ, t.loadField(this, off, i), v)
          else
            v.visitMissing(f.typ)
          v.leaveField()
          i += 1
        }
        v.leaveStruct()
      case t: ComplexType =>
        visit(t.representation, off, v)
    }
  }

  def pretty(t: Type, off: Long): String = {
    val v = new PrettyVisitor()
    visit(t, off, v)
    v.result()
  }
}

trait ValueVisitor {
  def visitMissing(t: Type): Unit

  def visitBoolean(b: Boolean): Unit

  def visitInt32(i: Int): Unit

  def visitInt64(l: Long): Unit

  def visitFloat32(f: Float): Unit

  def visitFloat64(d: Double): Unit

  def visitString(s: String): Unit

  def visitBinary(b: Array[Byte]): Unit

  def enterStruct(t: TStruct): Unit

  def enterField(f: Field): Unit

  def leaveField(): Unit

  def leaveStruct(): Unit

  def enterArray(t: TContainer, length: Int): Unit

  def leaveArray(): Unit

  def enterElement(i: Int): Unit

  def leaveElement(): Unit
}

final class PrettyVisitor extends ValueVisitor {
  val sb = new StringBuilder()

  def result(): String = sb.result()

  def visitMissing(t: Type) {
    sb.append("NA")
  }

  def visitBoolean(b: Boolean) {
    sb.append(b)
  }

  def visitInt32(i: Int) {
    sb.append(i)
  }

  def visitInt64(l: Long) {
    sb.append(l)
  }

  def visitFloat32(f: Float) {
    sb.append(f)
  }

  def visitFloat64(d: Double) {
    sb.append(d)
  }

  def visitBinary(a: Array[Byte]) {
    sb.append("bytes...")
  }

  def visitString(s: String) {
    sb.append(s)
  }

  def enterStruct(t: TStruct) {
    sb.append("{")
  }

  def enterField(f: Field) {
    if (f.index > 0)
      sb.append(",")
    sb.append(" ")
    sb.append(f.name)
    sb.append(": ")
  }

  def leaveField() {}

  def leaveStruct() {
    sb.append(" }")
  }

  def enterArray(t: TContainer, length: Int) {
    t match {
      case t: TSet =>
        sb.append("Set")
      case t: TDict =>
        sb.append("Dict")
      case _ =>
    }
    sb.append("[")
    sb.append(length)
    sb.append(";")
  }

  def leaveArray() {
    sb.append("]")
  }

  def enterElement(i: Int) {
    if (i > 0)
      sb.append(",")
    sb.append(" ")
  }

  def leaveElement() {}
}

object RegionValue {
  def apply(): RegionValue = new RegionValue(null, 0)

  def apply(region: MemoryBuffer): RegionValue = new RegionValue(region, 0)

  def apply(region: MemoryBuffer, offset: Long) = new RegionValue(region, offset)
}

final class RegionValue(var region: MemoryBuffer,
  var offset: Long) extends Serializable {
  def set(newRegion: MemoryBuffer, newOffset: Long) {
    region = newRegion
    offset = newOffset
  }

  def setRegion(newRegion: MemoryBuffer) {
    region = newRegion
  }

  def setOffset(newOffset: Long) {
    offset = newOffset
  }

  def pretty(t: Type): String = region.pretty(t, offset)
}

class RegionValueBuilder(var region: MemoryBuffer) {
  def this() = this(null)

  var start: Long = _
  var root: Type = _

  val typestk = new ArrayStack[Type]()
  val indexstk = new ArrayStack[Int]()
  val offsetstk = new ArrayStack[Long]()
  val elementsOffsetstk = new ArrayStack[Long]()

  def inactive: Boolean = root == null && typestk.isEmpty && offsetstk.isEmpty && elementsOffsetstk.isEmpty && indexstk.isEmpty

  def clear(): Unit = {
    root = null
    typestk.clear()
    offsetstk.clear()
    elementsOffsetstk.clear()
    indexstk.clear()
  }

  def set(newRegion: MemoryBuffer) {
    assert(inactive)
    region = newRegion
  }

  def currentOffset(): Long = {
    if (typestk.isEmpty)
      start
    else {
      val i = indexstk.top
      typestk.top match {
        case t: TStruct =>
          offsetstk.top + t.byteOffsets(i)
        case t: TArray =>
          elementsOffsetstk.top + i * t.elementByteSize
      }
    }
  }

  def currentType(): Type = {
    if (typestk.isEmpty)
      root
    else {
      typestk.top match {
        case t: TStruct =>
          val i = indexstk.top
          t.fields(i).typ
        case t: TArray =>
          t.elementType
      }
    }
  }

  def start(newRoot: Type) {
    assert(inactive)
    root = newRoot.fundamentalType
  }

  def allocateRoot() {
    assert(typestk.isEmpty)
    root match {
      case t: TArray =>
      case _: TBinary =>
      case _ =>
        region.align(root.alignment)
        start = region.allocate(root.byteSize)
    }
  }

  def end(): Long = {
    assert(root != null)
    root = null
    assert(inactive)
    start
  }

  def advance() {
    if (indexstk.nonEmpty)
      indexstk(0) = indexstk(0) + 1
  }

  def startStruct(init: Boolean = true) {
    if (typestk.isEmpty)
      allocateRoot()

    val t = currentType().asInstanceOf[TStruct]
    val off = currentOffset()
    typestk.push(t)
    offsetstk.push(off)
    indexstk.push(0)

    if (init)
      t.clearMissingBits(region, off)
  }

  def endStruct() {
    typestk.top match {
      case t: TStruct =>
        typestk.pop()
        offsetstk.pop()
        val last = indexstk.pop()
        assert(last == t.size)

        advance()
    }
  }

  def startArray(length: Int, init: Boolean = true) {
    val t = currentType().asInstanceOf[TArray]
    region.align(t.contentsAlignment)
    val aoff = region.allocate(t.contentsByteSize(length))

    if (typestk.nonEmpty) {
      val off = currentOffset()
      region.storeAddress(off, aoff)
    } else
      start = aoff

    typestk.push(t)
    elementsOffsetstk.push(aoff + t.elementsOffset(length))
    indexstk.push(0)
    offsetstk.push(aoff)

    if (init)
      t.initialize(region, aoff, length)
  }

  def endArray() {
    val t = typestk.top.asInstanceOf[TArray]
    val aoff = offsetstk.top
    val length = t.loadLength(region, aoff)
    assert(length == indexstk.top)

    typestk.pop()
    offsetstk.pop()
    elementsOffsetstk.pop()
    indexstk.pop()

    advance()
  }

  def setFieldIndex(newI: Int) {
    assert(typestk.top.isInstanceOf[TStruct])
    indexstk(0) = newI
  }

  def setMissing() {
    val i = indexstk.top
    typestk.top match {
      case t: TStruct =>
        if (t.fieldType(i).required)
          fatal(s"cannot set missing field for required type ${ t.fieldType(i) }")
        t.setFieldMissing(region, offsetstk.top, i)
      case t: TArray =>
        if (t.elementType.required)
          fatal(s"cannot set missing field for required type ${ t.elementType }")
        t.setElementMissing(region, offsetstk.top, i)
    }
    advance()
  }

  def addBoolean(b: Boolean) {
    assert(currentType().isInstanceOf[TBoolean])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    region.storeByte(off, b.toByte)
    advance()
  }

  def addInt(i: Int) {
    assert(currentType().isInstanceOf[TInt32])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    region.storeInt(off, i)
    advance()
  }

  def addLong(l: Long) {
    assert(currentType().isInstanceOf[TInt64])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    region.storeLong(off, l)
    advance()
  }

  def addFloat(f: Float) {
    assert(currentType().isInstanceOf[TFloat32])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    region.storeFloat(off, f)
    advance()
  }

  def addDouble(d: Double) {
    assert(currentType().isInstanceOf[TFloat64])
    if (typestk.isEmpty)
      allocateRoot()
    val off = currentOffset()
    region.storeDouble(off, d)
    advance()
  }

  def addBinary(bytes: Array[Byte]) {
    assert(currentType().isInstanceOf[TBinary])

    region.align(TBinary.contentAlignment)
    val boff = region.offset
    region.appendInt(bytes.length)
    region.appendBytes(bytes)

    if (typestk.nonEmpty) {
      val off = currentOffset()
      region.storeAddress(off, boff)
    } else
      start = boff

    advance()
  }

  def addString(s: String) {
    addBinary(s.getBytes)
  }

  def addRow(t: TStruct, r: Row) {
    assert(r != null)

    startStruct()
    var i = 0
    while (i < t.size) {
      addAnnotation(t.fields(i).typ, r.get(i))
      i += 1
    }
    endStruct()
  }

  def fixupBinary(fromRegion: MemoryBuffer, fromBOff: Long): Long = {
    val length = TBinary.loadLength(fromRegion, fromBOff)
    region.align(TBinary.contentAlignment)
    val toBOff = TBinary.allocate(region, length)
    region.copyFrom(fromRegion, fromBOff, toBOff, TBinary.contentByteSize(length))
    toBOff
  }

  def requiresFixup(t: Type): Boolean = {
    t match {
      case t: TStruct => t.fields.exists(f => requiresFixup(f.typ))
      case _: TArray | _: TBinary => true
      case _ => false
    }
  }

  def fixupArray(t: TArray, fromRegion: MemoryBuffer, fromAOff: Long): Long = {
    val length = t.loadLength(fromRegion, fromAOff)
    region.align(t.contentsAlignment)
    val toAOff = t.allocate(region, length)

    region.copyFrom(fromRegion, fromAOff, toAOff, t.contentsByteSize(length))

    if (region.ne(fromRegion) && requiresFixup(t.elementType)) {
      var i = 0
      while (i < length) {
        if (t.isElementDefined(fromRegion, fromAOff, i)) {
          t.elementType match {
            case t2: TStruct =>
              fixupStruct(t2, t.elementOffset(toAOff, length, i), fromRegion, t.elementOffset(fromAOff, length, i))

            case t2: TArray =>
              val toAOff2 = fixupArray(t2, fromRegion, t.loadElement(fromRegion, fromAOff, length, i))
              region.storeAddress(t.elementOffset(toAOff, length, i), toAOff2)

            case _: TBinary =>
              val toBOff = fixupBinary(fromRegion, t.loadElement(fromRegion, fromAOff, length, i))
              region.storeAddress(t.elementOffset(toAOff, length, i), toBOff)

            case _ =>
          }
        }
        i += 1
      }
    }

    toAOff
  }

  def fixupStruct(t: TStruct, toOff: Long, fromRegion: MemoryBuffer, fromOff: Long) {
    assert(region.ne(fromRegion))

    var i = 0
    while (i < t.size) {
      if (t.isFieldDefined(fromRegion, fromOff, i)) {
        t.fields(i).typ match {
          case t2: TStruct =>
            fixupStruct(t2, t.fieldOffset(toOff, i), fromRegion, t.fieldOffset(fromOff, i))

          case _: TBinary =>
            val toBOff = fixupBinary(fromRegion, t.loadField(fromRegion, fromOff, i))
            region.storeAddress(t.fieldOffset(toOff, i), toBOff)

          case t2: TArray =>
            val toAOff = fixupArray(t2, fromRegion, t.loadField(fromRegion, fromOff, i))
            region.storeAddress(t.fieldOffset(toOff, i), toAOff)

          case _ =>
        }
      }
      i += 1
    }
  }

  def addField(t: TStruct, fromRegion: MemoryBuffer, fromOff: Long, i: Int) {
    if (t.isFieldDefined(fromRegion, fromOff, i))
      addRegionValue(t.fieldType(i), fromRegion, t.loadField(fromRegion, fromOff, i))
    else
      setMissing()
  }

  def addField(t: TStruct, rv: RegionValue, i: Int) {
    addField(t, rv.region, rv.offset, i)
  }

  def addElement(t: TArray, fromRegion: MemoryBuffer, fromAOff: Long, i: Int) {
    if (t.isElementDefined(fromRegion, fromAOff, i))
      addRegionValue(t.elementType, fromRegion,
        t.elementOffsetInRegion(fromRegion, fromAOff, i))
    else
      setMissing()
  }

  def addElement(t: TArray, rv: RegionValue, i: Int) {
    addElement(t, rv.region, rv.offset, i)
  }

  def addRegionValue(t: Type, rv: RegionValue) {
    addRegionValue(t, rv.region, rv.offset)
  }

  def addRegionValue(t: Type, fromRegion: MemoryBuffer, fromOff: Long) {
    val toT = currentType()
    assert(toT == t.fundamentalType)

    if (typestk.isEmpty) {
      if (region.eq(fromRegion)) {
        start = fromOff
        advance()
        return
      }

      allocateRoot()
    }

    val toOff = currentOffset()
    assert(typestk.nonEmpty || toOff == start)

    t.fundamentalType match {
      case t: TStruct =>
        region.copyFrom(fromRegion, fromOff, toOff, t.byteSize)
        if (region.ne(fromRegion))
          fixupStruct(t, toOff, fromRegion, fromOff)
      case t: TArray =>
        if (region.eq(fromRegion)) {
          assert(!typestk.isEmpty)
          region.storeAddress(toOff, fromOff)
        } else {
          val toAOff = fixupArray(t, fromRegion, fromOff)
          if (typestk.nonEmpty)
            region.storeAddress(toOff, toAOff)
          else
            start = toAOff
        }
      case _: TBinary =>
        if (region.eq(fromRegion)) {
          assert(!typestk.isEmpty)
          region.storeAddress(toOff, fromOff)
        } else {
          val toBOff = fixupBinary(fromRegion, fromOff)
          if (typestk.nonEmpty)
            region.storeAddress(toOff, toBOff)
          else
            start = toBOff
        }
      case _ =>
        region.copyFrom(fromRegion, fromOff, toOff, t.byteSize)
    }
    advance()
  }

  def addUnsafeRow(t: TStruct, ur: UnsafeRow) {
    addRegionValue(t, ur.region, ur.offset)
  }

  def addUnsafeArray(t: TArray, uis: UnsafeIndexedSeq) {
    addRegionValue(t, uis.region, uis.aoff)
  }

  def addAnnotation(t: Type, a: Annotation) {
    if (a == null)
      setMissing()
    else
      t match {
        case _: TBoolean => addBoolean(a.asInstanceOf[Boolean])
        case _: TInt32 => addInt(a.asInstanceOf[Int])
        case _: TInt64 => addLong(a.asInstanceOf[Long])
        case _: TFloat32 => addFloat(a.asInstanceOf[Float])
        case _: TFloat64 => addDouble(a.asInstanceOf[Double])
        case _: TString => addString(a.asInstanceOf[String])
        case _: TBinary => addBinary(a.asInstanceOf[Array[Byte]])

        case t: TArray =>
          a match {
            case uis: UnsafeIndexedSeq =>
              addUnsafeArray(t, uis)

            case is: IndexedSeq[Annotation] =>
              startArray(is.length)
              var i = 0
              while (i < is.length) {
                addAnnotation(t.elementType, is(i))
                i += 1
              }
              endArray()
          }

        case t: TStruct =>
          a match {
            case ur: UnsafeRow =>
              addUnsafeRow(t, ur)
            case r: Row =>
              addRow(t, r)
          }

        case TSet(elementType, _) =>
          val s = a.asInstanceOf[Set[Annotation]]
            .toArray
            .sorted(elementType.ordering(true))
          startArray(s.length)
          s.foreach { x => addAnnotation(elementType, x) }
          endArray()

        case td: TDict =>
          val m = a.asInstanceOf[Map[Annotation, Annotation]]
            .map { case (k, v) => Row(k, v) }
            .toArray
            .sorted(td.elementType.ordering(true))
          startArray(m.length)
          m.foreach { case Row(k, v) =>
            startStruct()
            addAnnotation(td.keyType, k)
            addAnnotation(td.valueType, v)
            endStruct()
          }
          endArray()

        case t: TVariant =>
          val v = a.asInstanceOf[Variant]
          startStruct()
          addString(v.contig)
          addInt(v.start)
          addString(v.ref)
          startArray(v.altAlleles.length)
          var i = 0
          while (i < v.altAlleles.length) {
            addAnnotation(TAltAllele(), v.altAlleles()(i))
            i += 1
          }
          endArray()
          endStruct()

        case _: TAltAllele =>
          val aa = a.asInstanceOf[AltAllele]
          startStruct()
          addString(aa.ref)
          addString(aa.alt)
          endStruct()

        case _: TCall =>
          addInt(a.asInstanceOf[Int])

        case _: TGenotype =>
          val g = a.asInstanceOf[Genotype]
          startStruct()

          val unboxedGT = g._unboxedGT
          if (unboxedGT >= 0)
            addInt(unboxedGT)
          else
            setMissing()

          val unboxedAD = g._unboxedAD
          if (unboxedAD == null)
            setMissing()
          else {
            startArray(unboxedAD.length)
            var i = 0
            while (i < unboxedAD.length) {
              addInt(unboxedAD(i))
              i += 1
            }
            endArray()
          }

          val unboxedDP = g._unboxedDP
          if (unboxedDP >= 0)
            addInt(unboxedDP)
          else
            setMissing()

          val unboxedGQ = g._unboxedGQ
          if (unboxedGQ >= 0)
            addInt(unboxedGQ)
          else
            setMissing()

          val unboxedPX = g._unboxedPX
          if (unboxedPX == null)
            setMissing()
          else {
            startArray(unboxedPX.length)
            var i = 0
            while (i < unboxedPX.length) {
              addInt(unboxedPX(i))
              i += 1
            }
            endArray()
          }

          endStruct()

        case t: TLocus =>
          val l = a.asInstanceOf[Locus]
          startStruct()
          addString(l.contig)
          addInt(l.position)
          endStruct()

        case t: TInterval =>
          val i = a.asInstanceOf[Interval[Locus]]
          startStruct()
          addAnnotation(TLocus(t.gr), i.start)
          addAnnotation(TLocus(t.gr), i.end)
          endStruct()
      }

  }

  def result(): RegionValue = RegionValue(region, start)
}
