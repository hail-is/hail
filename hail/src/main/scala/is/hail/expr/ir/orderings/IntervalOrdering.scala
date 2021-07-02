package is.hail.expr.ir.orderings

import is.hail.asm4s.{Code, CodeLabel}
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder}
import is.hail.types.physical.stypes.interfaces.SInterval
import is.hail.types.physical.{PCode, PIntervalCode, PIntervalValue}

object IntervalOrdering {

  def make(t1: SInterval, t2: SInterval, ecb: EmitClassBuilder[_]): CodeOrdering = new CodeOrdering {

    val type1: SInterval = t1
    val type2: SInterval = t2

    private val setup: (EmitCodeBuilder, PCode, PCode) => (PIntervalValue, PIntervalValue) = {
      case (cb, lhs: PIntervalCode, rhs: PIntervalCode) =>
        lhs.memoize(cb, "intervalord_lhs") -> rhs.memoize(cb, "intervalord_rhs")
    }

    override def _compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] = {
      val pointCompare = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Compare())
      val cmp = cb.newLocal[Int]("intervalord_cmp", 0)

      val (lhs, rhs) = setup(cb, x, y)
      val lstart = EmitCode.fromI(cb.emb)(lhs.loadStart(_).typecast)
      val rstart = EmitCode.fromI(cb.emb)(rhs.loadStart(_).typecast)
      cb.assign(cmp, pointCompare(cb, lstart, rstart))
      cb.ifx(cmp.ceq(0), {
        cb.ifx(lhs.includesStart().cne(rhs.includesStart()), {
          cb.assign(cmp, lhs.includesStart().mux(-1, 1))
        }, {
          val lend = EmitCode.fromI(cb.emb)(lhs.loadEnd(_).typecast)
          val rend = EmitCode.fromI(cb.emb)(rhs.loadEnd(_).typecast)
          cb.assign(cmp, pointCompare(cb, lend, rend))
          cb.ifx(cmp.ceq(0), {
            cb.ifx(lhs.includesEnd().cne(rhs.includesEnd()), {
              cb.assign(cmp, lhs.includesEnd().mux(1, -1))
            })
          })
        })
      })

      cmp
    }

    override def _equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val pointEq = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Equiv())

      val Lout = CodeLabel()
      val ret = cb.newLocal[Boolean]("interval_eq", true)
      val exitWith = (value: Code[Boolean]) => {
        cb.assign(ret, value)
        cb.goto(Lout)
      }

      val (lhs, rhs) = setup(cb, x, y)

      cb.ifx(lhs.includesStart().cne(rhs.includesStart()) ||
        lhs.includesEnd().cne(rhs.includesEnd()), {
        exitWith(false)
      })

      val lstart = EmitCode.fromI(cb.emb)(lhs.loadStart(_).typecast)
      val rstart = EmitCode.fromI(cb.emb)(rhs.loadStart(_).typecast)
      cb.ifx(!pointEq(cb, lstart, rstart), exitWith(false))

      val lend = EmitCode.fromI(cb.emb)(lhs.loadEnd(_).typecast)
      val rend = EmitCode.fromI(cb.emb)(rhs.loadEnd(_).typecast)
      cb.ifx(!pointEq(cb, lend, rend), exitWith(false))

      cb.define(Lout)
      ret
    }

    override def _ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val pointLt = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Lt())
      val pointEq = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Equiv())

      val Lout = CodeLabel()
      val ret = cb.newLocal[Boolean]("interval_lt")
      val exitWith = (value: Code[Boolean]) => {
        cb.assign(ret, value)
        cb.goto(Lout)
      }

      val (lhs, rhs) = setup(cb, x, y)
      val lstart = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadStart(_).typecast), "linterval_start")
      val rstart = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadStart(_).typecast), "rinterval_start")

      cb.ifx(pointLt(cb, lstart, rstart), exitWith(true))
      cb.ifx(!pointEq(cb, lstart, rstart), exitWith(false))
      cb.ifx(lhs.includesStart() && !rhs.includesStart(), exitWith(true))
      cb.ifx(lhs.includesStart().cne(rhs.includesStart()), exitWith(false))

      val lend = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadEnd(_).typecast), "linterval_end")
      val rend = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadEnd(_).typecast), "rinterval_end")

      cb.ifx(pointLt(cb, lend, rend), exitWith(true))
      cb.assign(ret, pointEq(cb, lend, rend) && !lhs.includesEnd() && rhs.includesEnd())

      cb.define(Lout)
      ret
    }

    override def _lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val pointLtEq = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Lteq())
      val pointEq = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Equiv())

      val Lout = CodeLabel()
      val ret = cb.newLocal[Boolean]("interval_lteq")
      val exitWith = (value: Code[Boolean]) => {
        cb.assign(ret, value)
        cb.goto(Lout)
      }

      val (lhs, rhs) = setup(cb, x, y)
      val lstart = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadStart(_).typecast), "linterval_start")
      val rstart = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadStart(_).typecast), "rinterval_start")

      cb.ifx(!pointLtEq(cb, lstart, rstart), exitWith(false))
      cb.ifx(!pointEq(cb, lstart, rstart), exitWith(true))
      cb.ifx(lhs.includesStart() && !rhs.includesStart(), exitWith(true))
      cb.ifx(lhs.includesStart().cne(rhs.includesStart()), exitWith(false))

      val lend = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadEnd(_).typecast), "linterval_end")
      val rend = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadEnd(_).typecast), "rinterval_end")
      cb.ifx(!pointLtEq(cb, lend, rend), exitWith(false))
      cb.assign(ret, !pointEq(cb, lend, rend) || !lhs.includesEnd() || rhs.includesEnd())

      cb.define(Lout)
      ret
    }

    override def _gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val pointGt = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Gt())
      val pointEq = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Equiv())

      val Lout = CodeLabel()
      val ret = cb.newLocal[Boolean]("interval_gt")
      val exitWith = (value: Code[Boolean]) => {
        cb.assign(ret, value)
        cb.goto(Lout)
      }

      val (lhs, rhs) = setup(cb, x, y)
      val lstart = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadStart(_).typecast), "linterval_start")
      val rstart = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadStart(_).typecast), "rinterval_start")

      cb.ifx(pointGt(cb, lstart, rstart), exitWith(true))
      cb.ifx(!pointEq(cb, lstart, rstart), exitWith(false))
      cb.ifx(!lhs.includesStart() && rhs.includesStart(), exitWith(true))
      cb.ifx(lhs.includesStart().cne(rhs.includesStart()), exitWith(false))

      val lend = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadEnd(_).typecast), "linterval_end")
      val rend = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadEnd(_).typecast), "rinterval_end")

      cb.ifx(pointGt(cb, lend, rend), exitWith(true))
      cb.assign(ret, pointEq(cb, lend, rend) && lhs.includesEnd() && !rhs.includesEnd())

      cb.define(Lout)
      ret
    }

    override def _gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = {
      val pointGtEq = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Gteq())
      val pointEq = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Equiv())

      val Lout = CodeLabel()
      val ret = cb.newLocal[Boolean]("interval_gteq")
      val exitWith = (value: Code[Boolean]) => {
        cb.assign(ret, value)
        cb.goto(Lout)
      }

      val (lhs, rhs) = setup(cb, x, y)
      val lstart = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadStart(_).typecast), "linterval_start")
      val rstart = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadStart(_).typecast), "rinterval_start")

      cb.ifx(!pointGtEq(cb, lstart, rstart), exitWith(false))
      cb.ifx(!pointEq(cb, lstart, rstart), exitWith(true))
      cb.ifx(!lhs.includesStart() && rhs.includesStart(), exitWith(true))
      cb.ifx(lhs.includesStart().cne(rhs.includesStart()), exitWith(false))

      val lend = cb.memoize(EmitCode.fromI(cb.emb)(lhs.loadEnd(_).typecast), "linterval_end")
      val rend = cb.memoize(EmitCode.fromI(cb.emb)(rhs.loadEnd(_).typecast), "rinterval_end")
      cb.ifx(!pointGtEq(cb, lend, rend), exitWith(false))
      cb.assign(ret, !pointEq(cb, lend, rend) || lhs.includesEnd() || !rhs.includesEnd())

      cb.define(Lout)
      ret
    }
  }

}
