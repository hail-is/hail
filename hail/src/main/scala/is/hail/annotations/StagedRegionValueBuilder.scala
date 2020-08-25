package is.hail.annotations

import is.hail.asm4s.Code._
import is.hail.asm4s.{Code, FunctionBuilder, _}
import is.hail.expr.ir
import is.hail.expr.ir.{EmitClassBuilder, EmitFunctionBuilder, EmitMethodBuilder, EmitRegion, ParamType}
import is.hail.types.physical._
import is.hail.types.virtual.{TBoolean, TFloat32, TFloat64, TInt32, TInt64, Type}
import is.hail.utils._

object StagedRegionValueBuilder {
  def deepCopy(cb: EmitClassBuilder[_], region: Code[Region], typ: PType, value: Code[_], dest: Code[Long]): Code[Unit] = {
    val t = typ.fundamentalType
    val valueTI = typeToTypeInfo(t)
    val mb = cb.getOrGenEmitMethod("deepCopy", ("deepCopy", typ),
      FastIndexedSeq[ParamType](classInfo[Region], valueTI, LongInfo), UnitInfo) { mb =>
      val r = mb.getCodeParam[Region](1)
      val value = mb.getCodeParam(2)(valueTI)
      val dest = mb.getCodeParam[Long](3)
      mb.emit(t.constructAtAddressFromValue(mb, dest, r, t, value, true))
    }
    mb.invokeCode[Unit](region, value, dest)
  }

  def deepCopyFromOffset(cb: EmitClassBuilder[_], region: Code[Region], typ: PType, value: Code[Long]): Code[Long] = {
    val t = typ.fundamentalType
    val mb = cb.getOrGenEmitMethod("deepCopyFromOffset", ("deepCopyFromOffset", typ),
      FastIndexedSeq[ParamType](classInfo[Region], LongInfo), LongInfo) { mb =>
      val r = mb.getCodeParam[Region](1)
      val value = mb.getCodeParam[Long](2)
      mb.emit(t.copyFromType(mb, r, t, value, true))
    }
    mb.invokeCode[Long](region, value)
  }

  def deepCopyFromOffset(er: EmitRegion, typ: PType, value: Code[Long]): Code[Long] =
    deepCopyFromOffset(er.mb.ecb, er.region, typ, value)

  def deepCopy(er: EmitRegion, typ: PType, value: Code[_], dest: Code[Long]): Code[Unit] =
    deepCopy(er.mb.ecb, er.region, typ, value, dest)
}

class StagedRegionValueBuilder private (val mb: EmitMethodBuilder[_], val typ: PType, var region: Value[Region], val pOffset: Value[Long]) {
  def this(mb: EmitMethodBuilder[_], typ: PType, parent: StagedRegionValueBuilder) = {
    this(mb, typ, parent.region, parent.currentOffset)
  }

  def this(mb: EmitMethodBuilder[_], rowType: PType, r: Value[Region]) = {
    this(mb, rowType, r, null)
  }

  def this(er: ir.EmitRegion, rowType: PType) = {
    this(er.mb, rowType, er.region, null)
  }

  private val ftype = typ.fundamentalType

  private var staticIdx: Int = 0
  private var idx: Settable[Int] = _
  private var elementsOffset: Settable[Long] = _
  private val startOffset: Settable[Long] = mb.genFieldThisRef[Long]("srvb_start")

  ftype match {
    case t: PBaseStruct => elementsOffset = mb.genFieldThisRef[Long]("srvb_struct_addr")
    case t: PArray =>
      elementsOffset = mb.genFieldThisRef[Long]("srvb_array_addr")
      idx = mb.genFieldThisRef[Int]("srvb_array_idx")
    case _ =>
  }

  def offset: Value[Long] = startOffset

  def arrayIdx: Value[Int] = idx

  def currentOffset: Value[Long] = {
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

  def start(length: Code[Int], init: Boolean = true): Code[Unit] =
    Code.memoize(length, "srvb_start_length") { length =>
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
    val t = ftype.asInstanceOf[PCanonicalBaseStruct]
    var c = if (pOffset == null)
      startOffset.store(region.allocate(t.alignment, t.byteSize))
    else
      startOffset.store(pOffset)
    staticIdx = 0
    if (t.size > 0)
      c = Code(c, elementsOffset := startOffset + t.byteOffsets(0))
    if (init)
      c = Code(c, t.stagedInitialize(startOffset))
    c
  }

  def setMissing(): Code[Unit] = {
    ftype match {
      case t: PArray => t.setElementMissing(startOffset, idx)
      case t: PCanonicalBaseStruct => t.setFieldMissing(startOffset, staticIdx)
    }
  }

  def currentPType(): PType = {
    ftype match {
      case t: PArray => t.elementType
      case t: PCanonicalBaseStruct =>
        t.types(staticIdx)
      case t => t
    }
  }

  def checkType(knownType: Type): Unit = {
    val current = currentPType().virtualType
    if (current != knownType)
      throw new RuntimeException(s"bad SRVB addition: expected $current, tried to add $knownType")
  }

  def addBoolean(v: Code[Boolean]): Code[Unit] = {
    checkType(TBoolean)
    Region.storeByte(currentOffset, v.toI.toB)
  }

  def addInt(v: Code[Int]): Code[Unit] = {
    checkType(TInt32)
    Region.storeInt(currentOffset, v)
  }

  def addLong(v: Code[Long]): Code[Unit] = {
    checkType(TInt64)
    Region.storeLong(currentOffset, v)
  }

  def addFloat(v: Code[Float]): Code[Unit] = {
    checkType(TFloat32)
    Region.storeFloat(currentOffset, v)
  }

  def addDouble(v: Code[Double]): Code[Unit] = {
    checkType(TFloat64)
    Region.storeDouble(currentOffset, v)
  }

  def addBinary(bytes: Code[Array[Byte]]): Code[Unit] = {
    val b = mb.genFieldThisRef[Array[Byte]]("srvb_add_binary_bytes")
    val boff = mb.genFieldThisRef[Long]("srvb_add_binary_addr")
    val pbT = currentPType().asInstanceOf[PBinary]

    Code(
      b := bytes,
      boff := pbT.allocate(region, b.length()),
      ftype match {
        case _: PBinary => startOffset := boff
        case _ =>
          Region.storeAddress(currentOffset, boff)
      },
      pbT.store(boff, b))
  }


  def addAddress(v: Code[Long]): Code[Unit] = Region.storeAddress(currentOffset, v)

  def addString(str: Code[String]): Code[Unit] = addBinary(str.invoke[Array[Byte]]("getBytes"))

  def addArray(t: PArray, f: (StagedRegionValueBuilder => Code[Unit])): Code[Unit] = {
    if (!(t.fundamentalType isOfType currentPType()))
      throw new RuntimeException(s"Fundamental type doesn't match. current=${currentPType()}, t=${t.fundamentalType}, ftype=$ftype")
    f(new StagedRegionValueBuilder(mb, currentPType(), this))
  }

  def addBaseStruct(t: PBaseStruct, f: (StagedRegionValueBuilder => Code[Unit])): Code[Unit] = {
    if (!(t.fundamentalType isOfType currentPType()))
      throw new RuntimeException(s"Fundamental type doesn't match. current=${currentPType()}, t=${t.fundamentalType}, ftype=$ftype")
    f(new StagedRegionValueBuilder(mb, currentPType(), this))
  }

  def addIRIntermediate(t: PType, deepCopy: Boolean): (Code[_]) => Code[Unit] = t.fundamentalType match {
    case _: PBoolean => v => addBoolean(v.asInstanceOf[Code[Boolean]])
    case _: PInt32 => v => addInt(v.asInstanceOf[Code[Int]])
    case _: PInt64 => v => addLong(v.asInstanceOf[Code[Long]])
    case _: PFloat32 => v => addFloat(v.asInstanceOf[Code[Float]])
    case _: PFloat64 => v => addDouble(v.asInstanceOf[Code[Double]])
    case t =>
      val current = currentPType()
      val valueTI = typeToTypeInfo(t)
      val m = mb.getOrGenEmitMethod("addIRIntermediate", ("addIRIntermediate", current, t, deepCopy),
        FastIndexedSeq[ParamType](classInfo[Region], valueTI, LongInfo), UnitInfo) { mb =>
        val r = mb.getCodeParam[Region](1)
        val value = mb.getCodeParam(2)(valueTI)
        val dest = mb.getCodeParam[Long](3)
        mb.emit(current.constructAtAddressFromValue(mb, dest, r, t, value, deepCopy))
      }
       (v: Code[_]) => {
         assert(v.v != null)
         m.invokeCode[Unit](region, v, currentOffset) }
  }

  def addIRIntermediate(t: PType): (Code[_]) => Code[Unit] =
    addIRIntermediate(t, deepCopy = false)

  def addIRIntermediate(v: PCode, deepCopy: Boolean): Code[Unit] =
    addIRIntermediate(v.pt, deepCopy)(v.code)

  def addIRIntermediate(v: PCode): Code[Unit] =
    addIRIntermediate(v.pt, deepCopy = false)(v.code)

  def addWithDeepCopy(t: PType, v: Code[_]): Code[Unit] = {
    if (!(t.fundamentalType isOfType currentPType()))
      throw new RuntimeException(s"Fundamental type doesn't match. current=${currentPType()}, t=${t.fundamentalType}, ftype=$ftype")
    StagedRegionValueBuilder.deepCopy(
      EmitRegion(mb.asInstanceOf[EmitMethodBuilder[_]], region),
      t, v, currentOffset)
  }

  def advance(): Code[Unit] = {
    ftype match {
      case t: PArray => Code(
        elementsOffset := elementsOffset + t.elementByteSize,
        idx := idx + 1
      )
      case t: PCanonicalBaseStruct =>
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
