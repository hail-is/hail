package is.hail.annotations

import is.hail.asm4s.{AsmFunction2, Code, FunctionBuilder}
import is.hail.asm4s.Code._
import is.hail.asm4s._
import is.hail.expr._
import is.hail.utils._
import org.objectweb.asm.tree.{AbstractInsnNode, IincInsnNode}

import scala.collection.generic.Growable
import scala.reflect.ClassTag
import scala.language.postfixOps

class StagedRegionValueBuilder private(val fb: FunctionBuilder[_], val typ: Type, var region: Code[MemoryBuffer], val pOffset: Code[Long]) {

  private def this(fb: FunctionBuilder[_], typ: Type, parent: StagedRegionValueBuilder) = {
    this(fb, typ, parent.region, parent.currentOffset)
  }

  def this(fb: FunctionBuilder[_], rowType: Type) = {
    this(fb, rowType, fb.getArg[MemoryBuffer](1), null)
  }

  private var staticIdx: Int = 0
  private var idx: LocalRef[Int] = _
  private var elementsOffset: LocalRef[Long] = _
  private val startOffset: LocalRef[Long] = fb.newLocal[Long]

  typ match {
    case t: TStruct => elementsOffset = fb.newLocal[Long]
    case t: TArray => {
      elementsOffset = fb.newLocal[Long]
      idx = fb.newLocal[Int]
    }
    case _ =>
  }

  def offset: Code[Long] = startOffset.load()

  def endOffset: Code[Long] = region.size

  def arrayIdx: Code[Int] = idx.load()

  def currentOffset: Code[Long] = {
    typ match {
      case _: TStruct => elementsOffset
      case _: TArray => elementsOffset
      case _ => startOffset
    }
  }

  def start(): Code[Unit] = {
    assert(!typ.isInstanceOf[TArray])
    typ.fundamentalType match {
      case _: TStruct => start(true)
      case TBinary =>
        assert(pOffset == null)
        startOffset.store(endOffset)
      case _ => Code(
        region.align(typ.alignment),
        startOffset.store(region.allocate(typ.byteSize))
      )
    }
  }

  def start(length: Code[Int], init: Boolean = true): Code[Unit] = {
    val t = typ.asInstanceOf[TArray]
    var c = Code(
      region.align(t.contentsAlignment),
      startOffset.store(region.allocate(t.contentsByteSize(length)))
    )
    if (pOffset != null) {
      c = Code(c, region.storeAddress(pOffset, startOffset))
    }
    if (init)
      c = Code(c, t.initialize(region, startOffset.load(), length, idx))
    c = Code(c, elementsOffset.store(startOffset.load() + t.elementsOffset(length)))
    Code(c, idx.store(0))
  }

  def start(init: Boolean): Code[Unit] = {
    val t = typ.asInstanceOf[TStruct]
    var c = if (pOffset == null)
      Code(
        region.align(t.alignment),
        startOffset.store(region.allocate(t.byteSize))
      )
    else
      startOffset.store(pOffset)
    c = Code(c, elementsOffset.store(startOffset + t.byteOffsets(0)))
    if (init)
      c = Code(c, t.clearMissingBits(region, startOffset))
    c
  }

  def setMissing(): Code[Unit] = {
    typ match {
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
      typ.fundamentalType match {
        case TBinary => _empty
        case _ =>
          region.storeAddress(currentOffset, endOffset)
      },
      region.appendInt32(bytes.length()),
      region.appendBytes(bytes)
    )
  }

  def addString(str: Code[String]): Code[Unit] = addBinary(str.invoke[Array[Byte]]("getBytes"))

  def addArray(t: TArray, f: (StagedRegionValueBuilder => Code[Unit])): Code[Unit] = f(new StagedRegionValueBuilder(fb, t, this))

  def addStruct(t: TStruct, f: (StagedRegionValueBuilder => Code[Unit]), init: LocalRef[Boolean] = null): Code[Unit] = f(new StagedRegionValueBuilder(fb, t, this))

  def advance(): Code[Unit] = {
    typ match {
      case t: TArray => Code(
        elementsOffset := elementsOffset + t.elementByteSize,
        idx ++
      )
      case t: TStruct =>
        staticIdx += 1
        if (staticIdx < t.size)
          elementsOffset.store(startOffset + t.byteOffsets(staticIdx))
        else _empty
    }
  }

  def returnStart(): Code[Unit] = _return(startOffset)
}
