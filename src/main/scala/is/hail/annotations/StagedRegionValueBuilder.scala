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

class StagedRegionValueBuilder private(val fb: FunctionBuilder[_], val typ: Type, var region: Code[Region], val pOffset: Code[Long]) {

  private def this(fb: FunctionBuilder[_], typ: Type, parent: StagedRegionValueBuilder) = {
    this(fb, typ, parent.region, parent.currentOffset)
  }

  def this(fb: FunctionBuilder[_], rowType: Type) = {
    this(fb, rowType, fb.getArg[Region](1), null)
  }

  private var staticIdx: Int = 0
  private var idx: LocalRef[Int] = _
  private var elementsOffset: LocalRef[Long] = _
  private val startOffset: LocalRef[Long] = fb.newLocal[Long]

  typ match {
    case t: TStruct => elementsOffset = fb.newLocal[Long]
    case t: TArray =>
      elementsOffset = fb.newLocal[Long]
      idx = fb.newLocal[Int]
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
      case _: TBinary =>
        assert(pOffset == null)
        startOffset.store(endOffset)
      case _ =>
        startOffset.store(region.allocate(typ.alignment, typ.byteSize))
    }
  }

  def start(length: Code[Int], init: Boolean = true): Code[Unit] = {
    val t = typ.asInstanceOf[TArray]
    var c = startOffset.store(region.allocate(t.contentsAlignment, t.contentsByteSize(length)))
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
      startOffset.store(region.allocate(t.alignment, t.byteSize))
    else
      startOffset.store(pOffset)
    assert(staticIdx == 0)
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

  def addBoolean(v: Code[Boolean]): Code[Unit] = region.storeByte(currentOffset, v.toI.toB)

  def addInt(v: Code[Int]): Code[Unit] = region.storeInt(currentOffset, v)

  def addLong(v: Code[Long]): Code[Unit] = region.storeLong(currentOffset, v)

  def addFloat(v: Code[Float]): Code[Unit] = region.storeFloat(currentOffset, v)

  def addDouble(v: Code[Double]): Code[Unit] = region.storeDouble(currentOffset, v)

  def addBinary(bytes: Code[Array[Byte]]): Code[Unit] = {
    val boff = fb.newLocal[Long]
    Code(
      boff := region.appendInt(bytes.length()),
      toUnit(region.appendBytes(bytes)),
      typ.fundamentalType match {
        case _: TBinary => _empty
        case _ =>
          region.storeAddress(currentOffset, boff)
      })
  }

  def addAddress(v: Code[Long]): Code[Unit] = region.storeAddress(currentOffset, v)

  def addString(str: Code[String]): Code[Unit] = addBinary(str.invoke[Array[Byte]]("getBytes"))

  def addArray(t: TArray, f: (StagedRegionValueBuilder => Code[Unit])): Code[Unit] = f(new StagedRegionValueBuilder(fb, t, this))

  def addStruct(t: TStruct, f: (StagedRegionValueBuilder => Code[Unit]), init: LocalRef[Boolean] = null): Code[Unit] = f(new StagedRegionValueBuilder(fb, t, this))

  def addIRIntermediate(t: Type): (Code[_]) => Code[Unit] = t.fundamentalType match {
    case _: TBoolean => v => addBoolean(v.asInstanceOf[Code[Boolean]])
    case _: TInt32 => v => addInt(v.asInstanceOf[Code[Int]])
    case _: TInt64 => v => addLong(v.asInstanceOf[Code[Long]])
    case _: TFloat32 => v => addFloat(v.asInstanceOf[Code[Float]])
    case _: TFloat64 => v => addDouble(v.asInstanceOf[Code[Double]])
    case _: TStruct => v =>
      region.copyFrom(region, v.asInstanceOf[Code[Long]], currentOffset, t.byteSize)
    case _: TArray => v => addAddress(v.asInstanceOf[Code[Long]])
    case _: TBinary => v => addAddress(v.asInstanceOf[Code[Long]])
    case ft => throw new UnsupportedOperationException("Unknown fundamental type: " + ft)
  }

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

  // FIXME, remove this?
  def returnStart(): Code[Unit] = _return(end())

  def end(): Code[Long] = startOffset
}
