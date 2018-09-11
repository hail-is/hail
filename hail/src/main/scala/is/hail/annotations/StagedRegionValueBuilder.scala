package is.hail.annotations

import is.hail.asm4s.{AsmFunction2, Code, FunctionBuilder}
import is.hail.asm4s.Code._
import is.hail.asm4s._
import is.hail.expr._
import is.hail.expr.types._
import is.hail.utils._
import org.objectweb.asm.tree.{AbstractInsnNode, IincInsnNode}

import scala.collection.generic.Growable
import scala.reflect.ClassTag
import scala.language.postfixOps

class StagedRegionValueBuilder private(val mb: MethodBuilder, val typ: Type, var region: Code[Region], val pOffset: Code[Long]) {

  private def this(mb: MethodBuilder, typ: Type, parent: StagedRegionValueBuilder) = {
    this(mb, typ, parent.region, parent.currentOffset)
  }

  def this(fb: FunctionBuilder[_], rowType: Type) = {
    this(fb.apply_method, rowType, fb.apply_method.getArg[Region](1), null)
  }

  def this(mb: MethodBuilder, rowType: Type) = {
    this(mb, rowType, mb.getArg[Region](1), null)
  }

  private val ftype = typ.fundamentalType

  private var staticIdx: Int = 0
  private var idx: ClassFieldRef[Int] = _
  private var elementsOffset: ClassFieldRef[Long] = _
  private val startOffset: ClassFieldRef[Long] = mb.newField[Long]

  ftype match {
    case t: TBaseStruct => elementsOffset = mb.newField[Long]
    case t: TArray =>
      elementsOffset = mb.newField[Long]
      idx = mb.newField[Int]
    case _ =>
  }

  def offset: Code[Long] = startOffset

  def arrayIdx: Code[Int] = idx

  def currentOffset: Code[Long] = {
    ftype match {
      case _: TBaseStruct => elementsOffset
      case _: TArray => elementsOffset
      case _ => startOffset
    }
  }

  def start(): Code[Unit] = {
    assert(!ftype.isInstanceOf[TArray])
    ftype match {
      case _: TBaseStruct => start(true)
      case _: TBinary =>
        assert(pOffset == null)
        startOffset := -1L
      case _ =>
        startOffset := region.allocate(ftype.alignment, ftype.byteSize)
    }
  }

  def start(length: Code[Int], init: Boolean = true): Code[Unit] = {
    val t = ftype.asInstanceOf[TArray]
    var c = startOffset.store(region.allocate(t.contentsAlignment, t.contentsByteSize(length)))
    if (pOffset != null) {
      c = Code(c, region.storeAddress(pOffset, startOffset))
    }
    if (init)
      c = Code(c, t.initialize(region, startOffset, length, idx))
    c = Code(c, elementsOffset.store(startOffset + t.elementsOffset(length)))
    Code(c, idx.store(0))
  }

  def start(init: Boolean): Code[Unit] = {
    val t = ftype.asInstanceOf[TBaseStruct]
    var c = if (pOffset == null)
      startOffset.store(region.allocate(t.alignment, t.byteSize))
    else
      startOffset.store(pOffset)
    assert(staticIdx == 0)
    if (t.size > 0)
      c = Code(c, elementsOffset := startOffset + t.byteOffsets(0))
    if (init)
      c = Code(c, t.clearMissingBits(region, startOffset))
    c
  }

  def setMissing(): Code[Unit] = {
    ftype match {
      case t: TArray => t.setElementMissing(region, startOffset, idx)
      case t: TBaseStruct =>
        if (t.fieldRequired(staticIdx))
          Code._fatal("Required field cannot be missing.")
        else
          t.setFieldMissing(region, startOffset, staticIdx)
    }
  }

  def addBoolean(v: Code[Boolean]): Code[Unit] = region.storeByte(currentOffset, v.toI.toB)

  def addInt(v: Code[Int]): Code[Unit] = region.storeInt(currentOffset, v)

  def addLong(v: Code[Long]): Code[Unit] = region.storeLong(currentOffset, v)

  def addFloat(v: Code[Float]): Code[Unit] = region.storeFloat(currentOffset, v)

  def addDouble(v: Code[Double]): Code[Unit] = region.storeDouble(currentOffset, v)

  def allocateBinary(n: Code[Int]): Code[Long] = {
    val boff = mb.newField[Long]
    Code(
      boff := TBinary.allocate(region, n),
      region.storeInt(boff, n),
      ftype match {
        case _: TBinary => _empty
        case _ =>
          region.storeAddress(currentOffset, boff)
      },
      boff)
  }

  def addBinary(bytes: Code[Array[Byte]]): Code[Unit] = {
    val boff = mb.newField[Long]
    Code(
      boff := region.appendBinary(bytes),
      ftype match {
        case _: TBinary =>
          startOffset := boff
        case _ =>
          region.storeAddress(currentOffset, boff)
      })
  }

  def addAddress(v: Code[Long]): Code[Unit] = region.storeAddress(currentOffset, v)

  def addString(str: Code[String]): Code[Unit] = addBinary(str.invoke[Array[Byte]]("getBytes"))

  def addArray(t: TArray, f: (StagedRegionValueBuilder => Code[Unit])): Code[Unit] = f(new StagedRegionValueBuilder(mb, t, this))

  def addBaseStruct(t: TBaseStruct, f: (StagedRegionValueBuilder => Code[Unit])): Code[Unit] = f(new StagedRegionValueBuilder(mb, t, this))

  def addIRIntermediate(t: Type): (Code[_]) => Code[Unit] = t.fundamentalType match {
    case _: TBoolean => v => addBoolean(v.asInstanceOf[Code[Boolean]])
    case _: TInt32 => v => addInt(v.asInstanceOf[Code[Int]])
    case _: TInt64 => v => addLong(v.asInstanceOf[Code[Long]])
    case _: TFloat32 => v => addFloat(v.asInstanceOf[Code[Float]])
    case _: TFloat64 => v => addDouble(v.asInstanceOf[Code[Double]])
    case _: TBaseStruct => v =>
      region.copyFrom(region, v.asInstanceOf[Code[Long]], currentOffset, t.byteSize)
    case _: TArray => v => addAddress(v.asInstanceOf[Code[Long]])
    case _: TBinary => v => addAddress(v.asInstanceOf[Code[Long]])
    case ft => throw new UnsupportedOperationException("Unknown fundamental type: " + ft)
  }

  def advance(): Code[Unit] = {
    ftype match {
      case t: TArray => Code(
        elementsOffset := elementsOffset + t.elementByteSize,
        idx := idx + 1
      )
      case t: TBaseStruct =>
        staticIdx += 1
        if (staticIdx < t.size)
          elementsOffset := elementsOffset + (t.byteOffsets(staticIdx) - t.byteOffsets(staticIdx - 1))
        else _empty
    }
  }

  def end(): Code[Long] = startOffset
}
