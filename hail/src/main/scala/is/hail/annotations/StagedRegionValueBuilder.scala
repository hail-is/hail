package is.hail.annotations

import is.hail.asm4s.Code._
import is.hail.asm4s.{Code, FunctionBuilder, _}
import is.hail.expr.ir
import is.hail.expr.ir.{EmitFunctionBuilder, EmitMethodBuilder, EmitRegion}
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual.{TBoolean, TFloat32, TFloat64, TInt32, TInt64, Type}
import is.hail.utils._

object StagedRegionValueBuilder {
  def fixupStruct(fb: EmitFunctionBuilder[_], region: Code[Region], typ: PBaseStruct, value: Code[Long]): Code[Unit] = {
    coerce[Unit](Code(typ.fields.map { f =>
      if (f.typ.isPrimitive)
        Code._empty
      else {
        val fix = f.typ.fundamentalType match {
          case t@(_: PBinary | _: PArray) =>
            val off = fb.newField[Long]
            Code(
              off := typ.fieldOffset(value, f.index),
              Region.storeAddress(off, deepCopyFromOffset(fb, region, t, coerce[Long](Region.loadIRIntermediate(t)(off))))
            )
          case t: PBaseStruct =>
            val off = fb.newField[Long]
            Code(off := typ.fieldOffset(value, f.index),
              fixupStruct(fb, region, t, off))
        }
        typ.isFieldDefined(region, value, f.index).mux(fix, Code._empty)
      }
    }: _*))
  }

  def fixupArray(fb: EmitFunctionBuilder[_], region: Code[Region], typ: PArray, value: Code[Long]): Code[Unit] = {
    if (typ.elementType.isPrimitive)
      return Code._empty

    val i = fb.newField[Int]
    val len = fb.newField[Int]

    val perElt = typ.elementType.fundamentalType match {
      case t@(_: PBinary | _: PArray) =>
        val off = fb.newField[Long]
        Code(
          off := typ.elementOffset(value, len, i),
          Region.storeAddress(off, deepCopyFromOffset(fb, region, t, coerce[Long](Region.loadIRIntermediate(t)(off)))))
      case t: PBaseStruct =>
        val off = fb.newField[Long]
        Code(off := typ.elementOffset(value, len, i),
          fixupStruct(fb, region, t, off))
    }
    Code(
      i := 0,
      len := typ.loadLength(region, value),
      Code.whileLoop(i < len,
        typ.isElementDefined(region, value, i).mux(perElt, Code._empty),
        i := i + 1))
  }

  def deepCopy(fb: EmitFunctionBuilder[_], region: Code[Region], typ: PType, value: Code[_], dest: Code[Long]): Code[Unit] = {
    typ.fundamentalType match {
      case t if t.isPrimitive => Region.storePrimitive(t, dest)(value)
      case t@(_: PBinary | _: PArray) =>
        Region.storeAddress(dest, deepCopyFromOffset(fb, region, t, coerce[Long](value)))
      case t: PBaseStruct =>
        Code(Region.copyFrom(coerce[Long](value), dest, t.byteSize),
          fixupStruct(fb, region, t, dest))
      case t => fatal(s"unknown type $t")
    }
  }

  def deepCopyFromOffset(fb: EmitFunctionBuilder[_], region: Code[Region], typ: PType, value: Code[Long]): Code[Long] = {
    val offset = fb.newField[Long]

    val copy = typ.fundamentalType match {
      case _: PBinary =>
        Code(
          offset := PBinary.allocate(region, PBinary.loadLength(region, value)),
          Region.copyFrom(value, offset, PBinary.contentByteSize(PBinary.loadLength(region, value))))
      case t: PArray =>
        Code(
          offset := t.copyFrom(fb.apply_method, region, value),
          fixupArray(fb, region, t, offset))
      case t =>
        Code(
          offset := region.allocate(t.alignment, t.byteSize),
          deepCopy(fb, region, t, Region.getIRIntermediate(t)(value), offset))
    }

    Code(copy, offset)
  }

  def deepCopyFromOffset(er: EmitRegion, typ: PType, value: Code[Long]): Code[Long] =
    deepCopyFromOffset(er.mb.fb, er.region, typ, value)

  def deepCopy(er: EmitRegion, typ: PType, value: Code[_], dest: Code[Long]): Code[Unit] =
    deepCopy(er.mb.fb, er.region, typ, value, dest)
}

class StagedRegionValueBuilder private(val mb: MethodBuilder, val typ: PType, var region: Code[Region], val pOffset: Code[Long]) {
  def this(mb: MethodBuilder, typ: PType, parent: StagedRegionValueBuilder) = {
    this(mb, typ, parent.region, parent.currentOffset)
  }

  def this(fb: FunctionBuilder[_], rowType: PType) = {
    this(fb.apply_method, rowType, fb.apply_method.getArg[Region](1), null)
  }

  def this(fb: FunctionBuilder[_], rowType: PType, pOffset: Code[Long]) = {
    this(fb.apply_method, rowType, fb.apply_method.getArg[Region](1), pOffset)
  }

  def this(mb: MethodBuilder, rowType: PType) = {
    this(mb, rowType, mb.getArg[Region](1), null)
  }

  def this(mb: MethodBuilder, rowType: PType, r: Code[Region]) = {
    this(mb, rowType, r, null)
  }

  def this(er: ir.EmitRegion, rowType: PType) = {
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

  def init(): Code[Unit] = Code(
    startOffset := -1L,
    elementsOffset := -1L,
    if (idx != null) idx := -1 else Code._empty
  )

  def start(): Code[Unit] = {
    assert(!ftype.isInstanceOf[PArray]) // Need to use other start with length.
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
    var c = startOffset.store(t.allocate(region, length))
    if (pOffset != null) {
      c = Code(c, Region.storeAddress(pOffset, startOffset))
    }
    if (init)
      c = Code(c, t.stagedInitialize(startOffset, length))
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

  def checkType(knownType: Type): Unit = {
    val current = ftype match {
      case t: PArray => t.elementType.virtualType
      case t: PBaseStruct =>
        t.types(staticIdx).virtualType
      case t => t.virtualType
    }
    if (!current.isOfType(knownType))
      throw new RuntimeException(s"bad SRVB addition: expected $current, tried to add $knownType")
  }

  def addBoolean(v: Code[Boolean]): Code[Unit] = {
    checkType(TBoolean())
    Region.storeByte(currentOffset, v.toI.toB)
  }

  def addInt(v: Code[Int]): Code[Unit] = {
    checkType(TInt32())
    Region.storeInt(currentOffset, v)
  }

  def addLong(v: Code[Long]): Code[Unit] = {
    checkType(TInt64())
    Region.storeLong(currentOffset, v)
  }

  def addFloat(v: Code[Float]): Code[Unit] = {
    checkType(TFloat32())
    Region.storeFloat(currentOffset, v)
  }

  def addDouble(v: Code[Double]): Code[Unit] = {
    checkType(TFloat64())
    Region.storeDouble(currentOffset, v)
  }

  def allocateBinary(n: Code[Int]): Code[Long] = {
    val boff = mb.newField[Long]
    Code(
      boff := PBinary.allocate(region, n),
      Region.storeInt(boff, n),
      ftype match {
        case _: PBinary => startOffset := boff
        case _ =>
          Region.storeAddress(currentOffset, boff)
      },
      boff)
  }

  def addBinary(bytes: Code[Array[Byte]]): Code[Unit] = {
    val b = mb.newField[Array[Byte]]
    val boff = mb.newField[Long]
    Code(
      b := bytes,
      boff := PBinary.allocate(region, b.length()),
      ftype match {
        case _: PBinary => startOffset := boff
        case _ =>
          Region.storeAddress(currentOffset, boff)
      },
      PBinary.store(boff, b))
  }


  def addAddress(v: Code[Long]): Code[Unit] = Region.storeAddress(currentOffset, v)

  def addString(str: Code[String]): Code[Unit] = addBinary(str.invoke[Array[Byte]]("getBytes"))

  def addArray(t: PArray, f: (StagedRegionValueBuilder => Code[Unit])): Code[Unit] = f(new StagedRegionValueBuilder(mb, t, this))

  def addBaseStruct(t: PBaseStruct, f: (StagedRegionValueBuilder => Code[Unit])): Code[Unit] = f(new StagedRegionValueBuilder(mb, t, this))

  def addIRIntermediate(t: PType): (Code[_]) => Code[Unit] = t.fundamentalType match {
    case _: PBoolean => v => addBoolean(v.asInstanceOf[Code[Boolean]])
    case _: PInt32 => v => addInt(v.asInstanceOf[Code[Int]])
    case _: PInt64 => v => addLong(v.asInstanceOf[Code[Long]])
    case _: PFloat32 => v => addFloat(v.asInstanceOf[Code[Float]])
    case _: PFloat64 => v => addDouble(v.asInstanceOf[Code[Double]])
    case _: PBaseStruct => v => Region.copyFrom(v.asInstanceOf[Code[Long]], currentOffset, t.byteSize)
    case _: PArray => v => addAddress(v.asInstanceOf[Code[Long]])
    case _: PBinary => v => addAddress(v.asInstanceOf[Code[Long]])
    case ft => throw new UnsupportedOperationException("Unknown fundamental type: " + ft)
  }

  def addWithDeepCopy(t: PType, v: Code[_]): Code[Unit] =
    StagedRegionValueBuilder.deepCopy(
      EmitRegion(mb.asInstanceOf[EmitMethodBuilder], region),
      t, v, currentOffset)

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
