package is.hail.expr.ir.orderings

import is.hail.asm4s.{Code, CodeLabel, Value}
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder}
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.interfaces.SInterval

object IntervalOrdering {

  def make(t1: SInterval, t2: SInterval, ecb: EmitClassBuilder[_]): CodeOrdering = new CodeOrdering {

    override val type1: SInterval = t1
    override val type2: SInterval = t2

    override def _compareNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Int] = {
      val pointCompare = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Compare())
      val cmp = cb.newLocal[Int]("intervalord_cmp", 0)

      val lhs = x.asInterval
      val rhs = y.asInterval
      val lstart = cb.memoize(lhs.loadStart(cb))
      val rstart = cb.memoize(rhs.loadStart(cb))
      cb.assign(cmp, pointCompare(cb, lstart, rstart))
      cb.if_(cmp.ceq(0), {
        cb.if_(lhs.includesStart.cne(rhs.includesStart), {
          cb.assign(cmp, lhs.includesStart.mux(-1, 1))
        }, {
          val lend = cb.memoize(lhs.loadEnd(cb))
          val rend = cb.memoize(rhs.loadEnd(cb))
          cb.assign(cmp, pointCompare(cb, lend, rend))
          cb.if_(cmp.ceq(0), {
            cb.if_(lhs.includesEnd.cne(rhs.includesEnd), {
              cb.assign(cmp, lhs.includesEnd.mux(1, -1))
            })
          })
        })
      })

      cmp
    }

    override def _equivNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
      val pointEq = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Equiv())

      val Lout = CodeLabel()
      val ret = cb.newLocal[Boolean]("interval_eq", true)
      val exitWith = (value: Code[Boolean]) => {
        cb.assign(ret, value)
        cb.goto(Lout)
      }

      val lhs = x.asInterval
      val rhs = y.asInterval

      cb.if_(lhs.includesStart.cne(rhs.includesStart) ||
        lhs.includesEnd.cne(rhs.includesEnd), {
        exitWith(false)
      })

      val lstart = cb.memoize(lhs.loadStart(cb))
      val rstart = cb.memoize(rhs.loadStart(cb))
      cb.if_(!pointEq(cb, lstart, rstart), exitWith(false))

      val lend = cb.memoize(lhs.loadEnd(cb))
      val rend = cb.memoize(rhs.loadEnd(cb))
      cb.if_(!pointEq(cb, lend, rend), exitWith(false))

      cb.define(Lout)
      ret
    }

    override def _ltNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
      val pointLt = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Lt())
      val pointEq = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Equiv())

      val Lout = CodeLabel()
      val ret = cb.newLocal[Boolean]("interval_lt")
      val exitWith = (value: Code[Boolean]) => {
        cb.assign(ret, value)
        cb.goto(Lout)
      }

      val lhs = x.asInterval
      val rhs = y.asInterval
      val lstart = cb.memoize(lhs.loadStart(cb))
      val rstart = cb.memoize(rhs.loadStart(cb))

      cb.if_(pointLt(cb, lstart, rstart), exitWith(true))
      cb.if_(!pointEq(cb, lstart, rstart), exitWith(false))
      cb.if_(lhs.includesStart && !rhs.includesStart, exitWith(true))
      cb.if_(lhs.includesStart.cne(rhs.includesStart), exitWith(false))

      val lend = cb.memoize(lhs.loadEnd(cb))
      val rend = cb.memoize(rhs.loadEnd(cb))

      cb.if_(pointLt(cb, lend, rend), exitWith(true))
      cb.assign(ret, pointEq(cb, lend, rend) && !lhs.includesEnd && rhs.includesEnd)

      cb.define(Lout)
      ret
    }

    override def _lteqNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
      val pointLtEq = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Lteq())
      val pointEq = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Equiv())

      val Lout = CodeLabel()
      val ret = cb.newLocal[Boolean]("interval_lteq")
      val exitWith = (value: Code[Boolean]) => {
        cb.assign(ret, value)
        cb.goto(Lout)
      }

      val lhs = x.asInterval
      val rhs = y.asInterval
      val lstart = cb.memoize(lhs.loadStart(cb))
      val rstart = cb.memoize(rhs.loadStart(cb))

      cb.if_(!pointLtEq(cb, lstart, rstart), exitWith(false))
      cb.if_(!pointEq(cb, lstart, rstart), exitWith(true))
      cb.if_(lhs.includesStart && !rhs.includesStart, exitWith(true))
      cb.if_(lhs.includesStart.cne(rhs.includesStart), exitWith(false))

      val lend = cb.memoize(lhs.loadEnd(cb))
      val rend = cb.memoize(rhs.loadEnd(cb))
      cb.if_(!pointLtEq(cb, lend, rend), exitWith(false))
      cb.assign(ret, !pointEq(cb, lend, rend) || !lhs.includesEnd || rhs.includesEnd)

      cb.define(Lout)
      ret
    }

    override def _gtNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
      val pointGt = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Gt())
      val pointEq = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Equiv())

      val Lout = CodeLabel()
      val ret = cb.newLocal[Boolean]("interval_gt")
      val exitWith = (value: Code[Boolean]) => {
        cb.assign(ret, value)
        cb.goto(Lout)
      }

      val lhs = x.asInterval
      val rhs = y.asInterval
      val lstart = cb.memoize(lhs.loadStart(cb))
      val rstart = cb.memoize(rhs.loadStart(cb))

      cb.if_(pointGt(cb, lstart, rstart), exitWith(true))
      cb.if_(!pointEq(cb, lstart, rstart), exitWith(false))
      cb.if_(!lhs.includesStart && rhs.includesStart, exitWith(true))
      cb.if_(lhs.includesStart.cne(rhs.includesStart), exitWith(false))

      val lend = cb.memoize(lhs.loadEnd(cb))
      val rend = cb.memoize(rhs.loadEnd(cb))

      cb.if_(pointGt(cb, lend, rend), exitWith(true))
      cb.assign(ret, pointEq(cb, lend, rend) && lhs.includesEnd && !rhs.includesEnd)

      cb.define(Lout)
      ret
    }

    override def _gteqNonnull(cb: EmitCodeBuilder, x: SValue, y: SValue): Value[Boolean] = {
      val pointGtEq = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Gteq())
      val pointEq = ecb.getOrderingFunction(t1.pointType, t2.pointType, CodeOrdering.Equiv())

      val Lout = CodeLabel()
      val ret = cb.newLocal[Boolean]("interval_gteq")
      val exitWith = (value: Code[Boolean]) => {
        cb.assign(ret, value)
        cb.goto(Lout)
      }

      val lhs = x.asInterval
      val rhs = y.asInterval
      val lstart = cb.memoize(lhs.loadStart(cb))
      val rstart = cb.memoize(rhs.loadStart(cb))

      cb.if_(!pointGtEq(cb, lstart, rstart), exitWith(false))
      cb.if_(!pointEq(cb, lstart, rstart), exitWith(true))
      cb.if_(!lhs.includesStart && rhs.includesStart, exitWith(true))
      cb.if_(lhs.includesStart.cne(rhs.includesStart), exitWith(false))

      val lend = cb.memoize(lhs.loadEnd(cb))
      val rend = cb.memoize(rhs.loadEnd(cb))
      cb.if_(!pointGtEq(cb, lend, rend), exitWith(false))
      cb.assign(ret, !pointEq(cb, lend, rend) || lhs.includesEnd || !rhs.includesEnd)

      cb.define(Lout)
      ret
    }
  }

}
