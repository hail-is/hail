package is.hail.annotations

import is.hail.asm4s.{AsmFunction2, Code, FunctionBuilder}
import is.hail.asm4s.Code._
import is.hail.asm4s._
import is.hail.expr._
import is.hail.utils._
import org.objectweb.asm.tree.{AbstractInsnNode, IincInsnNode}

import scala.collection.generic.Growable
import scala.reflect.ClassTag

class StagedRegionValueBuilder[T] private (val fb: FunctionBuilder[AsmFunction2[T, MemoryBuffer, Long]], val rowType: Type, var unstagedRegion: Code[MemoryBuffer], val pOffset: Code[Long])(implicit tti: TypeInfo[T]) {

  private def this(fb: FunctionBuilder[AsmFunction2[T, MemoryBuffer, Long]], rowType: Type, parent: StagedRegionValueBuilder[T])(implicit tti: TypeInfo[T]) = {
    this(fb, rowType, parent.unstagedRegion, parent.currentOffset)
    region = parent.region
    extraInt = parent.extraInt
    extraLong = parent.extraLong
  }
  def this(fb: FunctionBuilder[AsmFunction2[T, MemoryBuffer, Long]], rowType: Type)(implicit tti: TypeInfo[T]) = {
    this(fb, rowType, fb.getArg[MemoryBuffer](2), null)
    region = new StagedMemoryBuffer(fb.newLocal[Long],fb.newLocal[Long],fb.newLocal[Long])
    extraInt = fb.newLocal[Int]
    extraLong = fb.newLocal[Long]
    fb.emit(Code(
      region.mem := unstagedRegion.invoke[Long]("mem"),
      region.offset := unstagedRegion.invoke[Long]("offset"),
      region.length := unstagedRegion.invoke[Long]("length")
    ))
  }

  val input: LocalRef[T] = fb.getArg[T](1)
  var staticIdx: Int = 0
  var idx: LocalRef[Int] = _
  var elementsOffset: LocalRef[Long] = _

  var extraInt: LocalRef[Int] = _
  var extraLong: LocalRef[Long] = _

  var region: StagedMemoryBuffer = _

  var elementsOffsetArray: LocalRef[Array[Long]] = _

  rowType match {
    case t: TStruct => elementsOffset = fb.newLocal[Long]
    case t: TArray => {
      elementsOffset = fb.newLocal[Long]
      elementsOffsetArray = fb.newLocal[Array[Long]]
      idx = fb.newLocal[Int]
      var c = elementsOffsetArray := Code.newArray[Long](10)
      var i = 0
      while (i < 10) {
        c = Code(c, elementsOffsetArray.update(i, t._elementsOffset(i)))
        i += 1
      }
      fb.emit(c)
    }
    case _ =>
  }

  var transform: () => AsmFunction2[T, MemoryBuffer, Long] = _

  val startOffset: LocalRef[Long] = fb.newLocal[Long]
  def endOffset: Code[Long] = region.size

  def currentOffset: Code[Long] = {
    rowType match {
      case _: TStruct => elementsOffset
      case t: TArray => elementsOffset// + (idx.toL * t.elementByteSize)
      case _ => startOffset
    }
  }

  def start(): Code[Unit] = {
    assert(!rowType.isInstanceOf[TArray])
    rowType.fundamentalType match {
      case _: TStruct => start(true)
      case TBinary =>
        assert (pOffset == null)
        startOffset.store(endOffset)
      case _ => Code(
        region.align(rowType.alignment),
        startOffset.store(region.allocate(rowType.byteSize))
      )
    }
  }

  def start(length: Code[Int], init: Boolean = true): Code[Unit] = {
    val t = rowType.asInstanceOf[TArray]
    var c = Code(
        region.align(t.contentsAlignment),
        startOffset.store(region.allocate(t.contentsByteSize(length)))
    )
    if (pOffset != null) {
      c = Code(c, region.storeAddress(pOffset, startOffset))
    }
    if (init)
      c = Code(c, t.initialize(region, startOffset.load(), length, idx))
    c = Code(c,
      (length < 10).mux(
        elementsOffset.store(startOffset.load() + elementsOffsetArray(length)),
        elementsOffset.store(startOffset.load() + t.elementsOffset(length))
      )
    )
    Code(c, idx.store(0))
  }

  def start(init: Boolean): Code[Unit] = {
    val t = rowType.asInstanceOf[TStruct]
    var c = if (pOffset == null)
      Code(
        region.align(t.alignment),
        startOffset.store(region.allocate(t.byteSize))
      )
    else
      startOffset.store(pOffset)
    c = Code(c, elementsOffset.store(startOffset + t.byteOffsets(0)))
    if (init)
      c = Code(c,t.clearMissingBits(region, startOffset))
    c
  }

  def setMissing(): Code[Unit] = {
    rowType match {
      case t: TArray => t.setElementMissing(region, startOffset, idx)
      case t: TStruct => t.setFieldMissing(region, startOffset, staticIdx)
    }
  }

  def addInt32(v: Code[Int]): Code[Unit] = region.storeInt32(currentOffset, v)

  def addInt64(v: Code[Long]): Code[Unit] = region.storeInt64(currentOffset, v)

  def addFloat32(v: Code[Float]): Code[Unit] = region.storeFloat32(currentOffset, v)

  def addFloat64(v: Code[Double]): Code[Unit] = region.storeFloat64(currentOffset, v)

  def addBinary(bytes: Code[Array[Byte]]): Code[Unit] = {
    Code(
      region.align(TBinary.contentAlignment),
      rowType.fundamentalType match {
        case TBinary => _empty
        case _ =>
          region.storeAddress(currentOffset,endOffset)
      },
      region.appendInt32(bytes.length()),
      region.appendBytes(bytes)
    )
  }

  def addString(str: Code[String]): Code[Unit] = addBinary(str.invoke[Array[Byte]]("getBytes"))

  def addArray(t: TArray, f: (StagedRegionValueBuilder[T] => Code[Unit])): Code[Unit] = f(new StagedRegionValueBuilder[T](fb, t, this))

  def addStruct(t: TStruct, f: (StagedRegionValueBuilder[T] => Code[Unit]), init: LocalRef[Boolean] = null): Code[Unit] = f(new StagedRegionValueBuilder[T](fb, t, this))

  def advance(): Code[Unit] = {
    rowType match {
      case t: TArray => Code(
        elementsOffset := elementsOffset + t.elementByteSize,
        new Code[Unit] {
          def emit(il: Growable[AbstractInsnNode]):Unit = {
            il += new IincInsnNode(idx.i, 1)
          }
      }
      )
      case t: TStruct => {
        staticIdx += 1
        if (staticIdx < t.size)
          elementsOffset.store(startOffset + t.byteOffsets(staticIdx))
        else _empty
      }
      case _ => _empty
    }
  }

  def build() {
    emit(unstagedRegion.invoke[Long, Unit]("mem_$eq", region.mem))
    emit(unstagedRegion.invoke[Long, Unit]("offset_$eq", region.offset))
    emit(unstagedRegion.invoke[Long, Unit]("length_$eq", region.length))
    emit(_return(startOffset))
    transform = fb.result()
  }

  def emit(c: Code[_]) {fb.emit(c)}

  def emit(cs: Array[Code[_]]) {
    for (c <- cs) {
      fb.emit(c)
    }
  }
}

class StagedMemoryBuffer(val mem: LocalRef[Long], val offset: LocalRef[Long], val length: LocalRef[Long]) {

  def size: Code[Long] = offset

  def storeInt32(off: Code[Long], v: Code[Int]): Code[Unit] = invokeStatic[Memory, Long, Int, Unit]("storeInt",mem + off, v)

  def storeInt64(off: Code[Long], v: Code[Long]): Code[Unit] = invokeStatic[Memory, Long, Long, Unit]("storeLong",mem + off, v)

  def storeFloat32(off: Code[Long], v: Code[Float]): Code[Unit] = invokeStatic[Memory, Long, Float, Unit]("storeFloat",mem + off, v)

  def storeFloat64(off: Code[Long], v: Code[Double]): Code[Unit] = invokeStatic[Memory, Long, Double, Unit]("storeDouble",mem + off, v)

  def storeAddress(off: Code[Long], a: Code[Long]): Code[Unit] = invokeStatic[Memory, Long, Long, Unit]("storeAddress",mem + off, a)

  def storeByte(off: Code[Long], b: Code[Byte]): Code[Unit] = invokeStatic[Memory, Long, Byte, Unit]("storeByte",mem + off, b)

  def storeBytes(off: Code[Long], bytes: Code[Array[Byte]], bytesOff: Code[Long], n: Code[Int]): Code[Unit] = invokeStatic[Memory, Long, Array[Byte], Long, Long, Unit]("copyFromArray", mem + off, bytes, bytesOff, n.toL) // [Memory, Long, Array[Byte], Long, Int, Unit]
  def storeBytes(off: Code[Long], bytes: Code[Array[Byte]]): Code[Unit] = storeBytes(off, bytes, 0L, bytes.length()) // [Memory, Long, Array[Byte], Long, Int, Unit]

  def ensure(n: Code[Long]):Code[Unit] = {
    val required = offset + n
    (length < required).mux(
      Code(
        length := invokeStatic[Math, Long, Long, Long]("max", length + length, required),
        mem := invokeStatic[Memory, Long, Long, Long]("realloc", mem, length)
      ),
      _empty
    )
  }

  def align(alignment: Code[Long]): Code[Unit] = offset := (offset + (alignment - 1L)) & alignment.negate()//& ((alignment - 1L) xor 0xffffffffL)

  def allocate(n: Code[Long]): Code[Long] = {
    Code(
      offset := offset + n,
      ensure(n),
      offset - n
    )
  }

  def alignAndAllocate(n: Code[Long]): Code[Long] = Code(align(n), allocate(n))

  def loadBoolean(off: Code[Long]): Code[Boolean] = loadByte(off).cne(0)

  def loadByte(off: Code[Long]): Code[Byte] =
    invokeStatic[Memory, Long, Byte]("loadByte",mem + off)

  def loadInt32(off: Code[Long]): Code[Int] =
    invokeStatic[Memory, Long, Int]("loadInt",mem + off)

  def loadBit(byteOff: Code[Long], bitOff: Code[Long]): Code[Boolean] =
    (loadByte(byteOff + (bitOff >> 3)) & (const(1) << (bitOff & 7L).toI)).cne(0)

  def setBit(byteOff: Code[Long], bitOff: Code[Long]): Code[Unit] = {
    val b = byteOff + (bitOff >> 3)
    storeByte(b,
      (loadByte(b) | (const(1) << (bitOff & 7L).toI)).toB)
  }

  def clearBit(byteOff: Code[Long], bitOff: Code[Long]): Code[Unit] = {
    val b = byteOff + (bitOff >> 3)
    storeByte(b,
      (loadByte(b) & (const(1) << (bitOff & 7L).toI xor 0xffff)).toB)
  }

  def storeBit(byteOff: Code[Long], bitOff: Code[Long], b: Code[Boolean]): Code[Unit] = b.mux(setBit(byteOff, bitOff), clearBit(byteOff, bitOff))

  def appendInt32(i: Code[Int]): Code[Unit] = {
    storeInt32(alignAndAllocate(4L), i)
  }

  def appendInt64(l: Code[Long]): Code[Unit] = {
    storeInt64(alignAndAllocate(8L), l)
  }

  def appendFloat32(f: Code[Float]): Code[Unit] = {
    storeFloat32(alignAndAllocate(4L), f)
  }

  def appendFloat64(d: Code[Double]): Code[Unit] = {
    storeFloat64(alignAndAllocate(8L), d)
  }

  def appendByte(b: Code[Byte]): Code[Unit] = {
    storeByte(allocate(1L), b)
  }

  def appendBytes(bytes: Code[Array[Byte]]): Code[Unit] = {
    storeBytes(allocate(bytes.length().toL), bytes)
  }

  def appendBytes(bytes: Code[Array[Byte]], bytesOff: Code[Long], n: Code[Int]): Code[Unit] = {
    storeBytes(allocate(n.toL), bytes, bytesOff, n)
  }

  def clear(): Code[Unit] = {
    offset := 0L
  }
}