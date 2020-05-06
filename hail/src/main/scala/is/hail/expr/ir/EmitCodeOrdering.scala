package is.hail.expr.ir

import is.hail.asm4s._
import is.hail.utils._
import is.hail.types.physical._
import scala.reflect.ClassTag

abstract class EmitCodeOrdering {
  val emodb: EmitModuleBuilder
  val ptLhs: PType
  val ptRhs: PType
  final val ordClass = emodb.genEmitClass[Unit](s"ecord_for_${ptLhs.asIdent}_AND_${ptRhs.asIdent}")

  // override these
  def emitCompare(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Int]
  def emitEq(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean]
  def emitGt(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean]
  def emitGtEq(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean]
  def emitLt(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean]
  def emitLtEq(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean]

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
}

abstract class ConsistentEmitCodeOrdering (
  val ptLhs: PType,
  val ptRhs: PType,
  val emodb: EmitModuleBuilder
) extends EmitCodeOrdering {
  assert(ptLhs isOfType ptRhs)

  def emitEq(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean] = emitCompare(cb, lhs, rhs).ceq(0)
  def emitGt(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean] = emitCompare(cb, lhs, rhs) > 0
  def emitGtEq(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean] = emitCompare(cb, lhs, rhs) >= 0
  def emitLt(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean] = emitCompare(cb, lhs, rhs) < 0
  def emitLtEq(cb: EmitCodeBuilder, lhs: PCode, rhs: PCode): Code[Boolean] = emitCompare(cb, lhs, rhs) <= 0
}
