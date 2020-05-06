package is.hail.expr.types.physical

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s._
import is.hail.expr.ir.{Ascending, Descending, EmitCode, EmitCodeBuilder, EmitMethodBuilder, EmitValue, SortOrder, Stream}
import is.hail.expr.types.virtual.{TThunk, Type}
import is.hail.{asm4s, lir}
import is.hail.utils.FastIndexedSeq

trait PUnrealizable extends PType {
  def mux(mb: EmitMethodBuilder[_], cond: Code[Boolean], ifT: PUnrealizableCode, ifF: PUnrealizableCode): PUnrealizableCode

  private def unsupported: Nothing =
    throw new UnsupportedOperationException(s"$this is not realizable")

  override def byteSize: Long = unsupported

  override def alignment: Long = unsupported

  override def codeOrdering(mb: EmitMethodBuilder[_], other: PType, so: SortOrder): CodeOrdering =
    unsupported

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering =
    unsupported

  def copyFromType(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Long] =
    unsupported

  def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_], deepCopy: Boolean): Code[_] =
    unsupported

  override def copyFromPValue(mb: EmitMethodBuilder[_], region: Value[Region], pv: PCode): PCode = {
    assert(pv.pt == this)
    pv
  }

  protected def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long =
    unsupported

  override def copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long =
    unsupported

  def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit] =
    unsupported

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit =
    unsupported
}

trait PUnrealizableCode extends PCode { self =>
  def memoize(cb: EmitCodeBuilder, name: String): PValue = new PValue {
    val pt = self.pt
    var used: Boolean = false
    def get: PCode = {
      assert(!used)
      used = true
      self
    }
  }

  private def unsupported: Nothing =
    throw new UnsupportedOperationException(s"$pt is not realizable")

  def code: Code[_] = unsupported

  def codeTuple(): IndexedSeq[Code[_]] = unsupported

  override def typeInfo: TypeInfo[_] = unsupported

  override def tcode[T](implicit ti: TypeInfo[T]): Code[T] = unsupported

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] =
    unsupported

  override def allocateAndStore(mb: EmitMethodBuilder[_], r: Value[Region]): (Code[Unit], Code[Long]) =
    unsupported

  def memoizeField(cb: EmitCodeBuilder, name: String): PValue = unsupported
}

final case class PThunk(ct: CType) extends PType with PUnrealizable {
  def required = true
  def virtualType: Type = TThunk(ct)

  def apply(emb: EmitMethodBuilder[_])(f: EmitCodeBuilder => CCode): PThunkCode = {
    val start = new lir.Block
    val cb = new EmitCodeBuilder(emb, start)
    val end = f(cb)
    assert(end.typ == ct)
    PThunkCode(start, end)
  }

  def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb ++= "PCStream["
    ct.pretty(sb, indent, compact)
    sb += ']'
  }
  def setRequired(required: Boolean): PType = {
    assert(required)
    this
  }
  def mux(mb: EmitMethodBuilder[_], cond: Code[Boolean], ifT: PUnrealizableCode, ifF: PUnrealizableCode): PUnrealizableCode = {
    val branch = cond.toCCode
    branch.Ltrue.append(lir.goto(ifT.asInstanceOf[PThunkCode].start))
    branch.Lfalse.append(lir.goto(ifF.asInstanceOf[PThunkCode].start))
    PThunkCode(branch.start, ct.merge(mb, ifT.asInstanceOf[PThunkCode].end, ifF.asInstanceOf[PThunkCode].end))
  }
  def _asIdent = s"thunk_of_${ct.asIdent}"
}

object PThunkCode {
  def apply(start: lir.Block, end: CCode): PThunkCode =
    new PThunkCode(start, end)

  def apply(emb: EmitMethodBuilder[_])(f: EmitCodeBuilder => CCode): PThunkCode = {
    val start = new lir.Block
    val cb = new EmitCodeBuilder(emb, start)
    val end = f(cb)
    PThunkCode(start, end)
  }
}

class PThunkCode(val start: lir.Block, val end: CCode) extends PCode with PUnrealizableCode { self =>
  val pt: PThunk = PThunk(end.typ)

  def force(cb: EmitCodeBuilder): CCode = {
    cb.end.append(lir.goto(start))
    cb.define(CUnitCode(start))
    end
  }

  def addSetup(setup: Code[Unit]): PThunkCode = {
    val newStart = setup.start
    setup.end.append(lir.goto(start))
    PThunkCode(newStart, end)
  }

  def map(f: CCode => CCode): PThunkCode =
    PThunkCode(start, f(end))
}

object POptionCode {
  def present(pt: PType, v: Code[_]): POptionCode = {
    val start = new lir.Block
    POptionCode(start, COption.present(CRetCode(CUnitCode(start), PCode(pt, v))))
  }

  def missing(pt: PType): POptionCode = {
    val start = new lir.Block
    POptionCode(start, COption.missing(pt, CUnitCode(start)))
  }

  def define(mb: EmitMethodBuilder[_])(f: (EmitCodeBuilder) => COptionCode): POptionCode = {
    val start = new lir.Block
    val cb = new EmitCodeBuilder(mb, start)
    POptionCode(start, f(cb))
  }

  def define(cb: EmitCodeBuilder)(f: => COptionCode): POptionCode = {
    val start = new lir.Block
    cb.define(CUnitCode(start))
    POptionCode(start, f)
  }

  def mapN(ecs: IndexedSeq[POptionCode], cb: EmitCodeBuilder)(f: IndexedSeq[PCode] => PCode): COptionCode =
    COptionCode.mapN(ecs.map(ec => () => ec.force(cb)), cb)(f)
//
//  def codeTupleTypes(pt: PType): IndexedSeq[TypeInfo[_]] = {
//    val ts = pt.codeTupleTypes()
//    if (pt.required)
//      ts
//    else
//      ts :+ BooleanInfo
//  }
//
//  def fromCodeTuple(pt: PType, ct: IndexedSeq[Code[_]]): POptionCode = {
//    if (pt.required)
//      POptionCode(Code._empty, const(false), pt.fromCodeTuple(ct))
//    else
//      POptionCode(Code._empty, coerce[Boolean](ct.last), pt.fromCodeTuple(ct.init))
//  }
}

case class POptionCode(start: lir.Block, end: COptionCode) extends PCode with PUnrealizableCode {
  val pt: PThunk = PThunk(end.typ)

  def valueType: PType = end.typ.valueType

  def force(cb: EmitCodeBuilder): COptionCode = {
    cb.define(CUnitCode(start))
    end
  }

  def v: Code[_] = end.present.value.code

  def value[T: TypeInfo]: Code[T] = end.present.value.tcode[T]

  def map(f: PCode => PCode): POptionCode =
    copy(end = end.copy(present = end.present.copy(value = f(end.present.value))))

  def toEmitCode: EmitCode = EmitCode(
    Code._empty,
    new asm4s.CCode(start, end.missing.end, end.present.end.end),
    end.present.value)
}


abstract class CType {
  def dummy: CCode
  def mergeWithFlag(mb: EmitMethodBuilder[_], l: CCode, r: CCode, inL: => Either[Settable[Boolean], Value[Boolean]]): CCode
  def merge(mb: EmitMethodBuilder[_], l: CCode, r: CCode): CCode = {
    lazy val inL: Settable[Boolean] = Code.newLocal[Boolean]("ct_merge_inL")
    mergeWithFlag(mb, l, r, Left(inL))
  }
  def pretty(sb: StringBuilder, indent: Int, compact: Boolean = false): Unit
  def asIdent: String
}

object CCode {
  def merge[R <: CCode](mb: EmitMethodBuilder[_], l: R, r: R): R = {
    assert(l.typ == r.typ)
    l.typ.merge(mb, l, r).asInstanceOf[R]
  }
}

abstract class CCode {
  val typ: CType
  def appendAll(c: => Code[Unit]): Unit
}

case object CUnit extends CType {
  def dummy: CUnitCode = CUnitCode(new lir.Block)
  def mergeWithFlag(mb: EmitMethodBuilder[_], l: CCode, r: CCode, inL: => Either[Settable[Boolean], Value[Boolean]]): CUnitCode = {
    val newEnd = new lir.Block
    l.asInstanceOf[CUnitCode].end.append(lir.goto(newEnd))
    r.asInstanceOf[CUnitCode].end.append(lir.goto(newEnd))
    CUnitCode(newEnd)
  }
  def pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb ++= "Unit"
  }
  def asIdent: String = "cunit"
}

object CUnitCode {
  def apply(cb: EmitCodeBuilder): CUnitCode = CUnitCode(cb.getEnd)
}

case class CUnitCode(var end: lir.Block) extends CCode {
  val typ: CType = CUnit

  def append(c: Code[Unit]): Unit = {
    end.append(lir.goto(c.start))
    end = c.end
    c.clear()
  }

  def appendAll(c: => Code[Unit]): Unit = append(c)

  def consume[R <: CCode](cb: EmitCodeBuilder, body: => R): R = {
    cb.setEnd(end)
    end = null
    val ret = body
    assert(!cb.isOpenEnded)
    ret
  }

  def consumeR(cb: EmitCodeBuilder, body: => PCode): PCode = {
    cb.setEnd(end)
    end = null
    val ret = body
    assert(cb.isOpenEnded)
    ret
  }

  def consumeU(cb: EmitCodeBuilder, body: => Unit): Unit = {
    cb.setEnd(end)
    end = null
    body
    assert(cb.isOpenEnded)
  }
}

case object CNothing extends CType {
  def apply(): CNothingCode = dummy
  def dummy: CNothingCode = CCanonicalNothingCode
  def mergeWithFlag(mb: EmitMethodBuilder[_], l: CCode, r: CCode, inL: => Either[Settable[Boolean], Value[Boolean]]): CNothingCode = {
    assert(l.typ == CNothing)
    assert(r.typ == CNothing)
    CCanonicalNothingCode
  }
  def pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb ++= "CNothing"
  }
  def asIdent: String = "cnothing"
}

abstract class CNothingCode extends CCode {
  val typ: CNothing.type = CNothing
  def appendAll(c: => Code[Unit]): Unit = {}
}

case object CCanonicalNothingCode extends CNothingCode

case class CProd(endType: CType, valueType: PType) extends CType {
  def dummy: CProdCode = CProdCode(endType.dummy, valueType.defaultValue)
  def mergeWithFlag(mb: EmitMethodBuilder[_], l: CCode, r: CCode, inL: => Either[Settable[Boolean], Value[Boolean]]): CProdCode = {
    assert(l.typ == r.typ, s"${l.typ} != ${r.typ}")
    assert(l.typ == this)
    (l, r) match {
      case (CProdCode(lEnd, lValue: PUnrealizableCode), CProdCode(rEnd, rValue: PUnrealizableCode)) =>
        val inLv: Value[Boolean] = inL match {
          case Left(inL) =>
            lEnd.appendAll(inL := true)
            rEnd.appendAll(inL := false)
            inL
          case Right(inL) =>
            inL
        }
        CProdCode(
          endType.mergeWithFlag(mb, lEnd, rEnd, Right(inLv)),
          valueType.asInstanceOf[PUnrealizable].mux(mb, inLv, lValue, rValue))
      case (CProdCode(lEnd, lValue), CProdCode(rEnd, rValue)) =>
        val x = mb.newPLocal(valueType)
        lEnd.appendAll(x := lValue)
        rEnd.appendAll(x := rValue)
        CProdCode(endType.mergeWithFlag(mb, lEnd, rEnd, inL), x)
    }
  }
  def pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb ++= "CProd["
    endType.pretty(sb, indent, compact)
    if (compact)
      sb += ','
    else
      sb ++= ", "
    valueType.pretty(sb, indent, compact)
    sb += ']'
  }
  def asIdent: String = s"cret_of_${valueType.asIdent}"
}

case class CProdCode(end: CCode, value: PCode) extends CCode {
  val valueType: PType = value.pt
  val endType: CType = end.typ

  val typ: CProd = CProd(endType, valueType)

  def appendAll(c: => Code[Unit]): Unit = end.appendAll(c)
}

case class CRet(valueType: PType) extends CType {
  def dummy: CRetCode = new CRetCode(CUnit.dummy, valueType.defaultValue)
  def mergeWithFlag(mb: EmitMethodBuilder[_], l: CCode, r: CCode, inL: => Either[Settable[Boolean], Value[Boolean]]): CRetCode = {
    assert(l.typ == r.typ, s"${l.typ} != ${r.typ}")
    assert(l.typ == this)
    (l, r) match {
      case (CRetCode(lEnd, lValue: PUnrealizableCode), CRetCode(rEnd, rValue: PUnrealizableCode)) =>
        val inLv: Value[Boolean] = inL match {
          case Left(inL) =>
            lEnd.append(inL := true)
            rEnd.append(inL := false)
            inL
          case Right(inL) =>
            inL
        }
        CRetCode(
          CUnit.mergeWithFlag(mb, lEnd, rEnd, Right(inLv)),
          valueType.asInstanceOf[PUnrealizable].mux(mb, inLv, lValue, rValue))
      case (CRetCode(lEnd, lValue), CRetCode(rEnd, rValue)) =>
        val x = mb.newPLocal(valueType)
        lEnd.append(x := lValue)
        rEnd.append(x := rValue)
        CRetCode(CUnit.mergeWithFlag(mb, lEnd, rEnd, inL), x)
    }
  }
  def pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb ++= "CRet["
    valueType.pretty(sb, indent, compact)
    sb += ']'
  }
  def asIdent: String = s"cret_of_${valueType.asIdent}"
}

object CRetCode {
  def apply(cb: EmitCodeBuilder, value: PCode): CRetCode =
    CRetCode(CUnitCode(cb.getEnd), value)
}

case class CRetCode(end: CUnitCode, value: PCode) extends CCode {
  val valueType: PType = value.pt

  val typ: CRet = CRet(valueType)

  def appendAll(c: => Code[Unit]): Unit = end.appendAll(c)

  def consume[R <: CCode](cb: EmitCodeBuilder, f: PCode => R): R =
    end.consume(cb, f(value))

  def consumeR(cb: EmitCodeBuilder, f: PCode => PCode): PCode =
    end.consumeR(cb, f(value))

  def consumeU(cb: EmitCodeBuilder, f: PCode => Unit): Unit =
    end.consumeU(cb, f(value))
}

case class CRet2(t1: PType, t2: PType) extends CType {
  def dummy: CRet2Code = new CRet2Code(CUnit.dummy, t1.defaultValue, t2.defaultValue)
  def mergeWithFlag(mb: EmitMethodBuilder[_], l: CCode, r: CCode, inL: => Either[Settable[Boolean], Value[Boolean]]): CRet2Code = {
    assert(l.typ == r.typ, s"${l.typ} != ${r.typ}")
    assert(l.typ == this)
    (l, r) match {
      case (CRet2Code(lEnd, lv1, lv2), CRet2Code(rEnd, rv1, rv2)) =>
        val (lv: PCode, inL2) = if (t1.isRealizable) {
          val x = mb.newPLocal(t1)
          lEnd.append(x := lv1)
          rEnd.append(x := rv1)
          (x, () => inL)
        } else {
          val inLv: Value[Boolean] = inL match {
            case Left(inL) =>
              lEnd.append(inL := true)
              rEnd.append(inL := false)
              inL
            case Right(inL) =>
              inL
          }
          (t1.asInstanceOf[PUnrealizable].mux(mb, inLv, lv1.asInstanceOf[PUnrealizableCode], rv1.asInstanceOf[PUnrealizableCode]), () => Right(inLv))
        }
        val rv: PCode = if (t2.isRealizable) {
          val x = mb.newPLocal(t2)
          lEnd.append(x := lv2)
          rEnd.append(x := rv2)
          x
        } else {
          val inLv: Value[Boolean] = inL2() match {
            case Left(inL) =>
              lEnd.append(inL := true)
              rEnd.append(inL := false)
              inL
            case Right(inL) =>
              inL
          }
          t1.asInstanceOf[PUnrealizable].mux(mb, inLv, lv1.asInstanceOf[PUnrealizableCode], rv1.asInstanceOf[PUnrealizableCode])
        }

        CRet2Code(CUnit.mergeWithFlag(mb, lEnd, rEnd, inL), lv, rv)
    }
  }
  def pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb ++= "CRet2["
    t1.pretty(sb, indent, compact)
    if (compact)
      sb += ','
    else
      sb ++= ", "
    t2.pretty(sb, indent, compact)
    sb += ']'
  }
  def asIdent: String = s"cret2_of_${t1.asIdent}_and_${t2.asIdent}"
}

object CRet2Code {
  def apply(cb: EmitCodeBuilder, value1: PCode, value2: PCode): CRet2Code =
    CRet2Code(CUnitCode(cb.getEnd), value1, value2)
}

case class CRet2Code(end: CUnitCode, value1: PCode, value2: PCode) extends CCode {
  val typ: CRet2 = CRet2(value1.pt, value2.pt)

  def appendAll(c: => Code[Unit]): Unit = end.appendAll(c)

  def consume[R <: CCode](cb: EmitCodeBuilder, f: (PCode, PCode) => R): R =
    end.consume(cb, f(value1, value2))

  def consumeR(cb: EmitCodeBuilder, f: (PCode, PCode) => PCode): PCode =
    end.consumeR(cb, f(value1, value2))

  def consumeU(cb: EmitCodeBuilder, f: (PCode, PCode) => Unit): Unit =
    end.consumeU(cb, f(value1, value2))
}

object COption {
  def missing(valueType: PType, end: CUnitCode): COptionCode =
    COptionCode(end, CRet(valueType).dummy)
  def missing(cb: EmitCodeBuilder, valueType: PType): COptionCode =
    missing(valueType, CUnitCode(cb.getEnd))
  def present(ret: CRetCode): COptionCode =
    COptionCode(CUnit.dummy, ret)
  def present(cb: EmitCodeBuilder, value: PCode): COptionCode =
    present(CRetCode(cb, value))
}

case class COption(valueType: PType) extends CType {
  def dummy: COptionCode = COptionCode(CUnit.dummy, CRet(valueType).dummy)
  def mergeWithFlag(mb: EmitMethodBuilder[_], l: CCode, r: CCode, inL: => Either[Settable[Boolean], Value[Boolean]]): COptionCode = {
    assert(l.typ == r.typ)
    (l, r) match {
      case (COptionCode(lMissing, lPresent), COptionCode(rMissing, rPresent)) =>
        COptionCode(
          CUnit.mergeWithFlag(mb, lMissing, rMissing, inL),
          CRet(valueType).mergeWithFlag(mb, lPresent, rPresent, inL))
    }
  }
  def pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb ++= "COption["
    valueType.pretty(sb, indent, compact)
    sb += ']'
  }
  def asIdent: String = s"coption_of_${valueType.asIdent}"
}

object COptionCode {
  def apply(cb: EmitCodeBuilder, m: Code[Boolean], pc: => PCode): COptionCode = {
    val Lmissing = CodeLabel()
    val Lpresent = CodeLabel()
    cb.append(m.mux(Lmissing.goto, Lpresent.goto))
    cb.define(Lpresent)
    val present = COption.present(CRetCode(cb, pc))
    cb.clear()
    cb.define(Lmissing)
    val missing = COption.missing(cb, present.typ.valueType)
    cb.clear()

    CCode.merge(cb.emb, present, missing)
  }

  def mapN(ecs: IndexedSeq[() => COptionCode], cb: EmitCodeBuilder)(f: IndexedSeq[PCode] => PCode): COptionCode = {
    if (ecs.isEmpty)
      COption.present(CRetCode(cb, f(FastIndexedSeq())))
    else
      ecs.head().flatMap(cb)(pc => mapN(ecs.tail, cb)(pcs => f(pc +: pcs)))
  }
//
//  def mapN(ecs: IndexedSeq[POptionCode], cb: EmitCodeBuilder)(f: IndexedSeq[PCode] => PCode): COptionCode = {
//    if (ecs.isEmpty)
//      COption.present(CRetCode(cb, f(FastIndexedSeq())))
//    else
//      ecs.head.force(cb).flatMap(cb)(pc => mapN(ecs.tail, cb)(pcs => f(pc +: pcs)))
//  }

  def map2(cb: EmitCodeBuilder, opt1: => COptionCode, opt2: => COptionCode)(f: (PCode, PCode) => PCode): COptionCode =
    opt1.flatMap(cb)(pc1 => opt2.map(cb)(pc2 => f(pc1, pc2)))

  def firstPresent(cb: EmitCodeBuilder, valueType: PType, opts: IndexedSeq[COptionCode]): COptionCode = {
    if (opts.isEmpty)
      COption.missing(cb, valueType)
    else
      opts.head.consume(cb, {
        firstPresent(cb, valueType, opts.tail)
      }, { pc =>
        COption.present(CRetCode(cb, pc))
      })
  }

  def firstPresent(cb: EmitCodeBuilder, opts: IndexedSeq[COptionCode]): COptionCode = {
    assert(opts.nonEmpty)
    if (opts.length == 1)
      opts.head
    else
      opts.head.consume(cb, {
        firstPresent(cb, opts.tail)
      }, { pc =>
        COption.present(CRetCode(cb, pc))
      })
  }
}

case class COptionCode(missing: CUnitCode, present: CRetCode) extends CCode {
  val typ: COption = COption(present.valueType)

  def appendAll(c: => Code[Unit]): Unit = {
    missing.append(c)
    present.appendAll(c)
  }

  def consume[R <: CCode](cb: EmitCodeBuilder, ifMissing: => R, ifPresent: PCode => R): R = {
    val m = missing.consume(cb, ifMissing)
    val p = present.consume(cb, ifPresent)
    CCode.merge(cb.emb, m, p)
  }

  def consumeU(cb: EmitCodeBuilder, ifMissing: => Unit, ifPresent: PCode => Unit): Unit =
    cb.define(consume(cb, { ifMissing; cb.ret() }, pc => { ifPresent(pc); cb.ret() }))

  def consumeR(cb: EmitCodeBuilder, ifMissing: => PCode, ifPresent: PCode => PCode): PCode =
    cb.extract(consume(cb, cb.ret(ifMissing), v => cb.ret(ifPresent(v))))

  def flatMap(cb: EmitCodeBuilder)(f: PCode => COptionCode): COptionCode = {
    val p = present.consume(cb, f)
    val m = missing.consume(cb, COption.missing(cb, p.typ.valueType))
    CCode.merge(cb.emb, m, p)
  }

  def map(cb: EmitCodeBuilder)(f: PCode => PCode): COptionCode =
    flatMap(cb)(value => COption.present(CRetCode(cb, f(value))))

  def handle(cb: EmitCodeBuilder, ifMissing: => Unit): PCode = {
    val ret = consume(cb, {
      ifMissing
      assert(!cb.isOpenEnded)
      CRet(typ.valueType).dummy
    }, { value =>
      CRetCode(cb, value)
    })
    cb.end = ret.end.end
    ret.value
  }

  def ifPresent(cb: EmitCodeBuilder, f: PCode => Unit): Unit =
    consumeU(cb, {}, f)

  def getOrElse(cb: EmitCodeBuilder, orElse: PCode): PCode =
    consumeR(cb, orElse, v => v)
}
