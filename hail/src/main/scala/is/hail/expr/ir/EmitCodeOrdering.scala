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
  assert(ptLhs isOfType ptRhs)
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
