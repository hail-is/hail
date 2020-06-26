package is.hail.expr.ir

import is.hail.annotations.CodeOrdering
import is.hail.asm4s._
import is.hail.utils._
import is.hail.types.physical._
import scala.reflect.ClassTag

abstract class EmitCodeOrdering(
  val emodb: EmitModuleBuilder,
  val ptLhs: PType,
  val ptRhs: PType
) {
  require(ptLhs isOfType ptRhs)
  final val ordClass = emodb.genEmitClass[Unit](s"ecord_for_${ptLhs.asIdent}_AND_${ptRhs.asIdent}")

  def selectOrdering(
    op: CodeOrdering.Op
  ): (EmitCodeBuilder, EmitCode, EmitCode) => Code[op.ReturnType] = { (cb, lhs, rhs) =>
    val code = op match {
      case CodeOrdering.Compare(_) => compare(cb, lhs, rhs)
      case CodeOrdering.Equiv(_) => equiv(cb, lhs, rhs)
      case CodeOrdering.Neq(_) => !equiv(cb, lhs, rhs)
      case CodeOrdering.Gt(_) => gt(cb, lhs, rhs)
      case CodeOrdering.Gteq(_) => gteq(cb, lhs, rhs)
      case CodeOrdering.Lt(_) => lt(cb, lhs, rhs)
      case CodeOrdering.Lteq(_) => lteq(cb, lhs, rhs)
    }
    coerce[op.ReturnType](code)
  }

  // override these
  def emitCompare(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Int]
  def emitEq(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean]
  def emitGt(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean]
  def emitGtEq(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean]
  def emitLt(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean]
  def emitLtEq(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean]

  // Call these for comparisons
  final def compare(cb: EmitCodeBuilder, lhs: EmitCode, rhs: EmitCode): Code[Int] =
    cb.invokeCode[Int](_compare, lhs, rhs)
  final def equiv(cb: EmitCodeBuilder, lhs: EmitCode, rhs: EmitCode): Code[Boolean] =
    cb.invokeCode[Boolean](_eq, lhs, rhs)
  final def gt(cb: EmitCodeBuilder, lhs: EmitCode, rhs: EmitCode): Code[Boolean] =
    cb.invokeCode[Boolean](_gt, lhs, rhs)
  final def gteq(cb: EmitCodeBuilder, lhs: EmitCode, rhs: EmitCode): Code[Boolean] =
    cb.invokeCode[Boolean](_gteq, lhs, rhs)
  final def lt(cb: EmitCodeBuilder, lhs: EmitCode, rhs: EmitCode): Code[Boolean] =
    cb.invokeCode[Boolean](_lt, lhs, rhs)
  final def lteq(cb: EmitCodeBuilder, lhs: EmitCode, rhs: EmitCode): Code[Boolean] =
    cb.invokeCode[Boolean](_lteq, lhs, rhs)

  final lazy val _compare: EmitMethodBuilder[_] = {
    val mb = ordClass.newStaticEmitMethod("compare", FastIndexedSeq(ptLhs, ptRhs), IntInfo)
    mb.emitWithBuilder[Int] { cb =>
      val lhs = mb.getEmitParam(0)
      val rhs = mb.getEmitParam(1)
      cb.ifx(lhs.m || rhs.m, {
        cb.ifx(lhs.m && rhs.m, {
          cb._return(0)
        }, {
          cb._return(lhs.m.mux(1, -1))
        })
      })
      emitCompare(cb, lhs.pv, rhs.pv)
    }
    mb
  }

  final lazy val _eq: EmitMethodBuilder[_] = {
    val mb = ordClass.newStaticEmitMethod("equiv", FastIndexedSeq(ptLhs, ptRhs), BooleanInfo)
    mb.emitWithBuilder { cb =>
      val lhs = mb.getEmitParam(0)
      val rhs = mb.getEmitParam(1)
      cb.ifx(lhs.m || rhs.m, { cb._return(lhs.m && rhs.m) })
      emitEq(cb, lhs.pv, rhs.pv)
    }
    mb
  }

  final lazy val _gt: EmitMethodBuilder[_] = {
    val mb = ordClass.newStaticEmitMethod("gt", FastIndexedSeq(ptLhs, ptRhs), BooleanInfo)
    mb.emitWithBuilder { cb =>
      val lhs = mb.getEmitParam(0)
      val rhs = mb.getEmitParam(1)
      cb.ifx(lhs.m || rhs.m, { cb._return(!rhs.m) })
      emitGt(cb, lhs.pv, rhs.pv)
    }
    mb
  }

  final lazy val _gteq: EmitMethodBuilder[_] = {
    val mb = ordClass.newStaticEmitMethod("gteq", FastIndexedSeq(ptLhs, ptRhs), BooleanInfo)
    mb.emitWithBuilder { cb =>
      val lhs = mb.getEmitParam(0)
      val rhs = mb.getEmitParam(1)
      cb.ifx(lhs.m || rhs.m, { cb._return(!rhs.m || lhs.m) })
      emitGtEq(cb, lhs.pv, rhs.pv)
    }
    mb
  }

  final lazy val _lt: EmitMethodBuilder[_] = {
    val mb = ordClass.newStaticEmitMethod("lt", FastIndexedSeq(ptLhs, ptRhs), BooleanInfo)
    mb.emitWithBuilder { cb =>
      val lhs = mb.getEmitParam(0)
      val rhs = mb.getEmitParam(1)
      cb.ifx(lhs.m || rhs.m, { cb._return(!lhs.m) })
      emitLt(cb, lhs.pv, rhs.pv)
    }
    mb
  }

  final lazy val _lteq: EmitMethodBuilder[_] = {
    val mb = ordClass.newStaticEmitMethod("lteq", FastIndexedSeq(ptLhs, ptRhs), BooleanInfo)
    mb.emitWithBuilder { cb =>
      val lhs = mb.getEmitParam(0)
      val rhs = mb.getEmitParam(1)
      cb.ifx(lhs.m || rhs.m, { cb._return(!lhs.m || rhs.m) })
      emitLtEq(cb, lhs.pv, rhs.pv)
    }
    mb
  }
}

abstract class ConsistentEmitCodeOrdering (
  emodb: EmitModuleBuilder,
  ptLhs: PType,
  ptRhs: PType
) extends EmitCodeOrdering(emodb, ptLhs, ptRhs) {
  def emitEq(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean] = emitCompare(cb, lhs, rhs).ceq(0)
  def emitGt(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean] = emitCompare(cb, lhs, rhs) > 0
  def emitGtEq(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean] = emitCompare(cb, lhs, rhs) >= 0
  def emitLt(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean] = emitCompare(cb, lhs, rhs) < 0
  def emitLtEq(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean] = emitCompare(cb, lhs, rhs) <= 0
}

class ContainerEmitCodeOrdering(
  emodb: EmitModuleBuilder,
  override val ptLhs: PContainer,
  override val ptRhs: PContainer
) extends EmitCodeOrdering(emodb, ptLhs, ptRhs) {
  val elementOrdering = emodb.getCodeOrdering2(ptLhs.elementType, ptRhs.elementType)

  private[this] def setup(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): (PIndexableValue, PIndexableValue) = {
    val lhsv = lhs.asIndexable.memoize(cb, "container_ord_lhs")
    val rhsv = rhs.asIndexable.memoize(cb, "container_ord_rhs")
    lhsv -> rhsv
  }

  def loop(cb: EmitCodeBuilder, lhs: PIndexableValue, rhs: PIndexableValue)(
    f: (EmitCode, EmitCode) => Unit
  ): Unit = {
    val i = cb.newLocal[Int]("i", 0)
    val lim = cb.newLocal("lim", lhs.loadLength().min(rhs.loadLength()))
    cb.whileLoop(i < lim, {
      val left = EmitCode.fromI(cb.emb)(lhs.loadElement(_, i))
      val right = EmitCode.fromI(cb.emb)(rhs.loadElement(_, i))
      f(left, right)
      cb.assign(i, i + 1)
    })
  }

  def emitCompare(cb: EmitCodeBuilder, lhsc: PCode, rhsc: PCode): Code[Int] = {
    val (lhs, rhs) = setup(cb, lhsc, rhsc)

    val cmp = cb.newLocal[Int]("cmp", 0)

    loop(cb, lhs, rhs) { (lhs, rhs) =>
      cb.assign(cmp, elementOrdering.compare(cb, lhs, rhs))
      cb.ifx(cmp.cne(0), cb._return(cmp))
    }

    cmp.ceq(0).mux(
      Code.invokeStatic2[java.lang.Integer, Int, Int, Int]("compare", lhs.loadLength(), rhs.loadLength()),
      cmp)
  }

  def emitEq(cb: EmitCodeBuilder, lhsc: PCode, rhsc: PCode): Code[Boolean] = {
    val (lhs, rhs) = setup(cb, lhsc, rhsc)

    cb.ifx(lhs.loadLength().cne(lhs.loadLength()), {
      cb._return(false)
    })

    val equiv = cb.newLocal[Boolean]("eq", true)
    loop(cb, lhs, rhs) { (lhs, rhs) =>
      cb.assign(equiv, elementOrdering.equiv(cb, lhs, rhs))
      cb.ifx(!equiv, cb._return(equiv))
    }

    equiv
  }

  def emitGt(cb: EmitCodeBuilder, lhsc: PCode, rhsc: PCode): Code[Boolean] = {
    val (lhs, rhs) = setup(cb, lhsc, rhsc)
    val gt = cb.newLocal("gt", false)
    val eq = cb.newLocal("eq", true)

    loop(cb, lhs, rhs) { (lhsEC, rhsEC) =>
      val lhs = cb.memoize(lhsEC, "lhs_item")
      val rhs = cb.memoize(rhsEC, "rhs_item")
      cb.assign(gt, elementOrdering.gt(cb, lhs, rhs))
      cb.assign(eq, !gt && elementOrdering.equiv(cb, lhs, rhs))

      cb.ifx(!eq, cb._return(gt))
    }

    gt || (eq && lhs.loadLength() > rhs.loadLength())
  }

  def emitGtEq(cb: EmitCodeBuilder, lhsc: PCode, rhsc: PCode): Code[Boolean] = {
    val (lhs, rhs) = setup(cb, lhsc, rhsc)
    val gteq = cb.newLocal("gteq", true)
    val eq = cb.newLocal("eq", true)

    loop(cb, lhs, rhs) { (lhsEC, rhsEC) =>
      val lhs = cb.memoize(lhsEC, "lhs_item")
      val rhs = cb.memoize(rhsEC, "rhs_item")
      cb.assign(gteq, elementOrdering.gteq(cb, lhs, rhs))
      cb.assign(eq, elementOrdering.equiv(cb, lhs, rhs))

      cb.ifx(!eq, cb._return(gteq))
    }

    gteq && (!eq || lhs.loadLength() >= rhs.loadLength)
  }

  def emitLt(cb: EmitCodeBuilder, lhsc: PCode, rhsc: PCode): Code[Boolean] = {
    val (lhs, rhs) = setup(cb, lhsc, rhsc)
    val lt = cb.newLocal("lt", false)
    val eq = cb.newLocal("eq", true)

    loop(cb, lhs, rhs) { (lhsEC, rhsEC) =>
      val lhs = cb.memoize(lhsEC, "lhs_item")
      val rhs = cb.memoize(rhsEC, "rhs_item")
      cb.assign(lt, elementOrdering.lt(cb, lhs, rhs))
      cb.assign(eq, !lt && elementOrdering.equiv(cb, lhs, rhs))

      cb.ifx(!eq, cb._return(lt))
    }

    lt || (eq && lhs.loadLength() < rhs.loadLength)
  }

  def emitLtEq(cb: EmitCodeBuilder, lhsc: PCode, rhsc: PCode): Code[Boolean] = {
    val (lhs, rhs) = setup(cb, lhsc, rhsc)
    val lteq = cb.newLocal("lteq", true)
    val eq = cb.newLocal("eq", true)

    loop(cb, lhs, rhs) { (lhsEC, rhsEC) =>
      val lhs = cb.memoize(lhsEC, "lhs_item")
      val rhs = cb.memoize(rhsEC, "rhs_item")
      cb.assign(lteq, elementOrdering.lteq(cb, lhs, rhs))
      cb.assign(eq, elementOrdering.equiv(cb, lhs, rhs))

      cb.ifx(!eq, cb._return(lteq))
    }

    lteq && (!eq || lhs.loadLength() <= rhs.loadLength)
  }
}

class StructEmitCodeOrdering(
  emodb: EmitModuleBuilder,
  override val ptLhs: PBaseStruct,
  override val ptRhs: PBaseStruct
) extends EmitCodeOrdering(emodb, ptLhs, ptRhs) {
  val fldOrds: Array[EmitCodeOrdering] = ptLhs.types.zip(ptRhs.types).map { case (ptfl, ptfr) =>
    emodb.getCodeOrdering2(ptfl, ptfr)
  }

  val setup: (EmitCodeBuilder, PCode, PCode) => (PBaseStructValue, PBaseStructValue) = {
    case (cb: EmitCodeBuilder, lhs: PBaseStructCode, rhs: PBaseStructCode) =>
      lhs.memoize(cb, "structord_lhs") -> rhs.memoize(cb, "structord_rhs")
  }

  def emitCompare(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Int] = {
    val (lhsv, rhsv) = setup(cb, lhs, rhs)
    val cmp = cb.newLocal(s"struct_compare_result", 0)

    fldOrds.zipWithIndex.foreach { case (ord, i) =>
      val lhs = EmitCode.fromI(cb.emb) { cb => lhsv.loadField(cb, i) }
      val rhs = EmitCode.fromI(cb.emb) { cb => rhsv.loadField(cb, i) }
      cb.assign(cmp, ord.compare(cb, lhs, rhs))
      cb.ifx(cmp.cne(0), {
        cb._return(cmp)
      })
    }

    const(0)
  }

  def emitEq(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean] = {
    val (lhsv, rhsv) = setup(cb, lhs, rhs)
    val eq = cb.newLocal[Boolean]("eq", true)

    fldOrds.zipWithIndex.foreach { case (ord, i) =>
      val lhs = EmitCode.fromI(cb.emb) { cb => lhsv.loadField(cb, i) }
      val rhs = EmitCode.fromI(cb.emb) { cb => rhsv.loadField(cb, i) }
      cb.assign(eq, ord.equiv(cb, lhs, rhs))
      cb.ifx(!eq, {
        cb._return(eq)
      })
    }

    eq
  }

  def emitGt(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean] = {
    val (lhsv, rhsv) = setup(cb, lhs, rhs)
    val gt = cb.newLocal("gt", false)
    val eq = cb.newLocal("eq", true)

    fldOrds.zipWithIndex.foreach { case (ord, i) =>
      val lhs = cb.memoize(EmitCode.fromI(cb.emb) { cb => lhsv.loadField(cb, i) },
        s"fld_lhs_${i}_${ord.ptLhs.asIdent}")
      val rhs = cb.memoize(EmitCode.fromI(cb.emb) { cb => rhsv.loadField(cb, i) },
        s"fld_rhs_${i}_${ord.ptRhs.asIdent}")

      cb.assign(gt, ord.gt(cb, lhs, rhs))
      cb.assign(eq, !gt && ord.equiv(cb, lhs, rhs))

      cb.ifx(!eq, cb._return(gt))
    }

    gt
  }

  def emitGtEq(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean] = {
    val (lhsv, rhsv) = setup(cb, lhs, rhs)
    val gteq = cb.newLocal("gteq", true)
    val eq = cb.newLocal("eq", true)

    fldOrds.zipWithIndex.foreach { case (ord, i) =>
      val lhs = cb.memoize(EmitCode.fromI(cb.emb) { cb => lhsv.loadField(cb, i) },
        s"fld_lhs_${i}_${ord.ptLhs.asIdent}")
      val rhs = cb.memoize(EmitCode.fromI(cb.emb) { cb => rhsv.loadField(cb, i) },
        s"fld_rhs_${i}_${ord.ptRhs.asIdent}")

      cb.assign(gteq, ord.gteq(cb, lhs, rhs))
      cb.assign(eq, ord.equiv(cb, lhs, rhs))

      cb.ifx(!eq, cb._return(gteq))
    }

    gteq
  }

  def emitLt(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean] = {
    val (lhsv, rhsv) = setup(cb, lhs, rhs)
    val lt = cb.newLocal("lt", true)
    val eq = cb.newLocal("eq", true)

    fldOrds.zipWithIndex.foreach { case (ord, i) =>
      val lhs = cb.memoize(EmitCode.fromI(cb.emb) { cb => lhsv.loadField(cb, i) },
        s"fld_lhs_${i}_${ord.ptLhs.asIdent}")
      val rhs = cb.memoize(EmitCode.fromI(cb.emb) { cb => rhsv.loadField(cb, i) },
        s"fld_rhs_${i}_${ord.ptRhs.asIdent}")

      cb.assign(lt, ord.lt(cb, lhs, rhs))
      cb.assign(eq, !lt && ord.equiv(cb, lhs, rhs))

      cb.ifx(!eq, cb._return(lt))
    }

    lt
  }

  def emitLtEq(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean] = {
    val (lhsv, rhsv) = setup(cb, lhs, rhs)
    val lteq = cb.newLocal("lteq", true)
    val eq = cb.newLocal("eq", true)

    fldOrds.zipWithIndex.foreach { case (ord, i) =>
      val lhs = cb.memoize(EmitCode.fromI(cb.emb) { cb => lhsv.loadField(cb, i) },
        s"fld_lhs_${i}_${ord.ptLhs.asIdent}")
      val rhs = cb.memoize(EmitCode.fromI(cb.emb) { cb => rhsv.loadField(cb, i) },
        s"fld_rhs_${i}_${ord.ptRhs.asIdent}")

      cb.assign(lteq, ord.lteq(cb, lhs, rhs))
      cb.assign(eq, ord.equiv(cb, lhs, rhs))

      cb.ifx(!eq, cb._return(lteq))
    }

    lteq
  }
}

class IntervalEmitCodeOrdering(
  emodb: EmitModuleBuilder,
  override val ptLhs: PInterval,
  override val ptRhs: PInterval
) extends EmitCodeOrdering(emodb, ptLhs, ptRhs) {
  val pointOrd = emodb.getCodeOrdering2(ptLhs.pointType, ptRhs.pointType)

  val setup: (EmitCodeBuilder, PCode, PCode) => (PIntervalValue, PIntervalValue) = {
    case (cb, lhs: PIntervalCode, rhs: PIntervalCode) =>
      lhs.memoize(cb, "intervalord_lhs") -> rhs.memoize(cb, "intervalord_rhs")
  }

  def emitCompare(cb: EmitCodeBuilder, lhsc: PCode, rhsc: PCode): Code[Int] = {
    val (lhs, rhs) = setup(cb, lhsc, rhsc)
    val cmp = cb.newLocal("intervalord_cmp", 0)

    val lstart = EmitCode.fromI(cb.emb)(lhs.loadStart(_))
    val rstart = EmitCode.fromI(cb.emb)(rhs.loadStart(_))
    cb.assign(cmp, pointOrd.compare(cb, lstart, rstart))
    cb.ifx(cmp.cne(0), cb._return(cmp))

    cb.ifx(lhs.includesStart().cne(rhs.includesStart()), {
      cb._return(lhs.includesStart().mux(-1, 1))
    })

    val lend = EmitCode.fromI(cb.emb)(lhs.loadEnd(_))
    val rend = EmitCode.fromI(cb.emb)(rhs.loadEnd(_))
    cb.assign(cmp, pointOrd.compare(cb, lend, rend))
    cb.ifx(cmp.cne(0), cb._return(cmp))

    cb.ifx(lhs.includesEnd().cne(rhs.includesEnd()), {
      cb._return(lhs.includesEnd().mux(-1, 1))
    })

    cmp // cmp is 0
  }

  def emitEq(cb: EmitCodeBuilder, lhsc: PCode, rhsc: PCode): Code[Boolean] = {
    val (lhs, rhs) = setup(cb, lhsc, rhsc)

    cb.ifx(lhs.includesStart().cne(rhs.includesStart()) ||
      lhs.includesEnd().cne(rhs.includesEnd()), {
        cb._return(false)
      })

    val lstart = EmitCode.fromI(cb.emb)(lhs.loadStart(_))
    val rstart = EmitCode.fromI(cb.emb)(rhs.loadStart(_))

    cb.ifx(!pointOrd.equiv(cb, lstart, rstart), cb._return(false))

    val lend = EmitCode.fromI(cb.emb)(lhs.loadEnd(_))
    val rend = EmitCode.fromI(cb.emb)(rhs.loadEnd(_))

    cb.ifx(!pointOrd.equiv(cb, lend, rend), cb._return(false))

    const(true)
  }

  def emitGt(cb: EmitCodeBuilder, lhsc: PCode, rhsc: PCode): Code[Boolean] = {
    val (lhs, rhs) = setup(cb, lhsc, rhsc)
    val lstart = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadStart(_)), "linterval_start")
    val rstart = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadStart(_)), "rinterval_start")

    cb.ifx(pointOrd.gt(cb, lstart, rstart), cb._return(true))
    cb.ifx(!pointOrd.equiv(cb, lstart, rstart), cb._return(false))
    cb.ifx(!lhs.includesStart() && rhs.includesStart(), cb._return(true))
    cb.ifx(lhs.includesStart().cne(rhs.includesStart()), cb._return(false))

    val lend = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadEnd(_)), "linterval_end")
    val rend = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadEnd(_)), "rinterval_end")

    cb.ifx(pointOrd.gt(cb, lend, rend), cb._return(true))
    pointOrd.equiv(cb, lend, rend) && lhs.includesEnd() && !rhs.includesEnd()
  }

  def emitGtEq(cb: EmitCodeBuilder, lhsc: PCode, rhsc: PCode): Code[Boolean] = {
    val (lhs, rhs) = setup(cb, lhsc, rhsc)
    val lstart = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadStart(_)), "linterval_start")
    val rstart = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadStart(_)), "rinterval_start")

    cb.ifx(!pointOrd.gteq(cb, lstart, rstart), cb._return(false))
    cb.ifx(!pointOrd.equiv(cb, lstart, rstart), cb._return(true))
    cb.ifx(!lhs.includesStart() && rhs.includesStart(), cb._return(true))
    cb.ifx(lhs.includesStart().cne(rhs.includesStart()), cb._return(false))

    val lend = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadEnd(_)), "linterval_end")
    val rend = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadEnd(_)), "rinterval_end")
    cb.ifx(!pointOrd.gteq(cb, lend, rend), cb._return(false))
    !pointOrd.equiv(cb, lend, rend) || lhs.includesEnd() || !rhs.includesEnd()
  }

  def emitLt(cb: EmitCodeBuilder, lhsc: PCode, rhsc: PCode): Code[Boolean] = {
    val (lhs, rhs) = setup(cb, lhsc, rhsc)
    val lstart = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadStart(_)), "linterval_start")
    val rstart = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadStart(_)), "rinterval_start")

    cb.ifx(pointOrd.lt(cb, lstart, rstart), cb._return(true))
    cb.ifx(!pointOrd.equiv(cb, lstart, rstart), cb._return(false))
    cb.ifx(lhs.includesStart() && !rhs.includesStart(), cb._return(true))
    cb.ifx(lhs.includesStart().cne(rhs.includesStart()), cb._return(false))

    val lend = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadEnd(_)), "linterval_end")
    val rend = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadEnd(_)), "rinterval_end")

    cb.ifx(pointOrd.lt(cb, lend, rend), cb._return(true))
    pointOrd.equiv(cb, lend, rend) && !lhs.includesEnd() && rhs.includesEnd()
  }

  def emitLtEq(cb: EmitCodeBuilder, lhsc: PCode, rhsc: PCode): Code[Boolean] = {
    val (lhs, rhs) = setup(cb, lhsc, rhsc)
    val lstart = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadStart(_)), "linterval_start")
    val rstart = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadStart(_)), "rinterval_start")

    cb.ifx(!pointOrd.lteq(cb, lstart, rstart), cb._return(false))
    cb.ifx(!pointOrd.equiv(cb, lstart, rstart), cb._return(true))
    cb.ifx(lhs.includesStart() && !rhs.includesStart(), cb._return(true))
    cb.ifx(lhs.includesStart().cne(rhs.includesStart()), cb._return(false))

    val lend = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadEnd(_)), "linterval_end")
    val rend = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadEnd(_)), "rinterval_end")
    cb.ifx(!pointOrd.lteq(cb, lend, rend), cb._return(false))
    !pointOrd.equiv(cb, lend, rend) || !lhs.includesEnd() || rhs.includesEnd()
  }
}
