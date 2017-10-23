package is.hail.annotations

import is.hail.asm4s.{AsmFunction2, Code, FunctionBuilder}
import is.hail.asm4s.Code._
import is.hail.asm4s._
import is.hail.expr._
import is.hail.utils._
import org.objectweb.asm.tree.{AbstractInsnNode, IincInsnNode}

import scala.collection.generic.Growable
import scala.reflect.ClassTag

class StagedRegionValueBuilder[T] private (val fb: FunctionBuilder[AsmFunction2[T, MemoryBuffer, Long]], val rowType: Type, var region: Code[MemoryBuffer], val pOffset: Code[Long])(implicit tti: TypeInfo[T]) {

  private def this(fb: FunctionBuilder[AsmFunction2[T, MemoryBuffer, Long]], rowType: Type, parent: StagedRegionValueBuilder[T])(implicit tti: TypeInfo[T]) = {
    this(fb, rowType, parent.region, parent.currentOffset)
  }
  def this(fb: FunctionBuilder[AsmFunction2[T, MemoryBuffer, Long]], rowType: Type)(implicit tti: TypeInfo[T]) = {
    this(fb, rowType, fb.getArg[MemoryBuffer](2), null)
  }

  val input: LocalRef[T] = fb.getArg[T](1)
  var staticIdx: Int = 0
  var idx: LocalRef[Int] = _
  var elementsOffset: LocalRef[Long] = _

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