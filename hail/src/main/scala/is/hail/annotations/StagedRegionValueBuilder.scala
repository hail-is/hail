package is.hail.annotations

import is.hail.asm4s.Code._
import is.hail.asm4s.{Code, FunctionBuilder, _}
import is.hail.expr.ir
import is.hail.expr.ir.{EmitMethodBuilder, EmitRegion}
import is.hail.expr.types.physical._
import is.hail.utils._

object StagedRegionValueBuilder {
  def fixupStruct(er: EmitRegion, typ: PBaseStruct, value: Code[Long]): Code[Unit] = {
    coerce[Unit](Code(typ.fields.map { f =>
      if (f.typ.isPrimitive)
        Code._empty
      else {
        val fix = f.typ.fundamentalType match {
          case t@(_: PBinary | _: PArray) =>
            er.region.storeAddress(typ.fieldOffset(value, f.index),
              deepCopy(er, t, typ.loadField(er.region, value, f.index)))
          case t: PBaseStruct =>
              fixupStruct(er, t, typ.loadField(er.region, value, f.index))
        }
        typ.isFieldDefined(er.region, value, f.index).mux(fix, Code._empty)
      }
    }: _*))
  }

  def fixupArray(er: EmitRegion, typ: PArray, value: Code[Long]): Code[Unit] = {
    if (typ.elementType.isPrimitive)
      return Code._empty
    val region = er.region

    val i = er.mb.newField[Int]
    val len = er.mb.newField[Int]

    val perElt = typ.elementType.fundamentalType match {
      case t@(_: PBinary | _: PArray) =>
        region.storeAddress(typ.elementOffset(value, len, i),
          deepCopy(er, t, typ.loadElement(region, value, i)))
      case t: PBaseStruct =>
        val off = er.mb.newField[Long]
        Code(off := typ.elementOffset(value, len, i),
          fixupStruct(er, t, off))
    }

    Code(
      i := 0,
      len := typ.loadLength(region, value),
      Code.whileLoop(i < len,
        typ.isElementDefined(region, value, i).mux(perElt, Code._empty),
        i := i + 1))
  }

  def deepCopy(er: EmitRegion, typ: PType, src: Code[Long], dest: Code[Long]): Code[Unit] = {
    val region = er.region
    typ.fundamentalType match {
      case t if t.isPrimitive => region.copyFrom(region, src, dest, t.byteSize)
      case t@(_: PBinary | _: PArray) =>
        region.storeAddress(deepCopy(er, t, src), dest)
      case t: PBaseStruct =>
        Code(region.copyFrom(region, src, dest, t.byteSize),
          fixupStruct(er, t, dest))
      case t => fatal(s"unknown type $t")
    }
  }

  def deepCopy(er: EmitRegion, typ: PType, value: Code[Long]): Code[Long] = {
    val mb = er.mb
    val region = er.region
    val offset = mb.newField[Long]

    val copy = typ.fundamentalType match {
      case _: PBinary =>
        Code(
          offset := PBinary.allocate(region, PBinary.loadLength(region, value)),
          region.copyFrom(region, value, offset, PBinary.loadLength(region, value).toL + 4L))
      case t: PArray =>
        Code(
          offset := region.allocate(t.contentsAlignment, t.contentsByteSize(t.loadLength(region, value))),
          region.copyFrom(region, value, offset, t.contentsByteSize(t.loadLength(region, value))),
          fixupArray(er, t, offset))
      case t =>
        Code(
          offset := region.allocate(t.alignment, t.byteSize),
          deepCopy(er, t, value, offset))
    }
    Code(copy, offset)
  }
}

class StagedRegionValueBuilder private(val mb: MethodBuilder, val typ: PType, var region: Code[Region], val pOffset: Code[Long]) {

  private def this(mb: MethodBuilder, typ: PType, parent: StagedRegionValueBuilder) = {
    this(mb, typ, parent.region, parent.currentOffset)
  }

  def this(fb: FunctionBuilder[_], rowType: PType) = {
    this(fb.apply_method, rowType, fb.apply_method.getArg[Region](1), null)
  }

  def this(mb: MethodBuilder, rowType: PType) = {
    this(mb, rowType, mb.getArg[Region](1), null)
  }

  def this (er: ir.EmitRegion, rowType: PType) = {
    this(er.mb, rowType, er.region, null)
  }

  private val ftype = typ.fundamentalType

  private var staticIdx: Int = 0
  private var idx: ClassFieldRef[Int] = _
  private var elementsOffset: ClassFieldRef[Long] = _
  private val startOffset: ClassFieldRef[Long] = mb.newField[Long]

  ftype match {
    case t: PBaseStruct => elementsOffset = mb.newField[Long]
    case t: PArray =>
      elementsOffset = mb.newField[Long]
      idx = mb.newField[Int]
    case _ =>
  }

  def offset: Code[Long] = startOffset

  def arrayIdx: Code[Int] = idx

  def currentOffset: Code[Long] = {
    ftype match {
      case _: PBaseStruct => elementsOffset
      case _: PArray => elementsOffset
      case _ => startOffset
    }
  }

  def start(): Code[Unit] = {
    assert(!ftype.isInstanceOf[PArray])
    ftype match {
      case _: PBaseStruct => start(true)
      case _: PBinary =>
        assert(pOffset == null)
        startOffset := -1L
      case _ =>
        startOffset := region.allocate(ftype.alignment, ftype.byteSize)
    }
  }

  def start(length: Code[Int], init: Boolean = true): Code[Unit] = {
    val t = ftype.asInstanceOf[PArray]
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
    val t = ftype.asInstanceOf[PBaseStruct]
    var c = if (pOffset == null)
        startOffset.store(region.allocate(t.alignment, t.byteSize))
    else
      startOffset.store(pOffset)
    staticIdx = 0
    if (t.size > 0)
      c = Code(c, elementsOffset := startOffset + t.byteOffsets(0))
    if (init)
      c = Code(c, t.clearMissingBits(region, startOffset))
    c
  }

  def setMissing(): Code[Unit] = {
    ftype match {
      case t: PArray => t.setElementMissing(region, startOffset, idx)
      case t: PBaseStruct =>
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
      boff := PBinary.allocate(region, n),
      region.storeInt(boff, n),
      ftype match {
        case _: PBinary => _empty
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
        case _: PBinary =>
          startOffset := boff
        case _ =>
          region.storeAddress(currentOffset, boff)
      })
  }

  def addAddress(v: Code[Long]): Code[Unit] = region.storeAddress(currentOffset, v)

  def addString(str: Code[String]): Code[Unit] = addBinary(str.invoke[Array[Byte]]("getBytes"))

  def addArray(t: PArray, f: (StagedRegionValueBuilder => Code[Unit])): Code[Unit] = f(new StagedRegionValueBuilder(mb, t, this))

  def addBaseStruct(t: PBaseStruct, f: (StagedRegionValueBuilder => Code[Unit])): Code[Unit] = f(new StagedRegionValueBuilder(mb, t, this))

  def addIRIntermediate(t: PType): (Code[_]) => Code[Unit] = t.fundamentalType match {
    case _: PBoolean => v => addBoolean(v.asInstanceOf[Code[Boolean]])
    case _: PInt32 => v => addInt(v.asInstanceOf[Code[Int]])
    case _: PInt64 => v => addLong(v.asInstanceOf[Code[Long]])
    case _: PFloat32 => v => addFloat(v.asInstanceOf[Code[Float]])
    case _: PFloat64 => v => addDouble(v.asInstanceOf[Code[Double]])
    case _: PBaseStruct => v =>
      region.copyFrom(region, v.asInstanceOf[Code[Long]], currentOffset, t.byteSize)
    case _: PArray => v => addAddress(v.asInstanceOf[Code[Long]])
    case _: PBinary => v => addAddress(v.asInstanceOf[Code[Long]])
    case ft => throw new UnsupportedOperationException("Unknown fundamental type: " + ft)
  }

  def addWithDeepCopy(t: PType, src: Code[Long]): Code[Unit] = {
    StagedRegionValueBuilder.deepCopy(EmitRegion(mb.asInstanceOf[EmitMethodBuilder], region), t, src, currentOffset)
  }

  def advance(): Code[Unit] = {
    ftype match {
      case t: PArray => Code(
        elementsOffset := elementsOffset + t.elementByteSize,
        idx := idx + 1
      )
      case t: PBaseStruct =>
        staticIdx += 1
        if (staticIdx < t.size)
          elementsOffset := elementsOffset + (t.byteOffsets(staticIdx) - t.byteOffsets(staticIdx - 1))
        else _empty
    }
  }

  def end(): Code[Long] = {
    startOffset
  }
}
