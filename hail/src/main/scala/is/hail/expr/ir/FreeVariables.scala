package is.hail.expr.ir

import is.hail.expr.types.virtual.Type
import is.hail.utils.ArrayBuilder

object FreeVariables {
  def apply(ir: IR, supportsAgg: Boolean, supportsScan: Boolean): BindingEnv[Unit] = {

    def compute(ir1: IR, baseEnv: BindingEnv[Unit]): BindingEnv[Unit] = {
      ir1 match {
        case x@Ref(name, _) =>
          baseEnv.bindEval(name, ())
        case TableAggregate(_, _) =>
        case MatrixAggregate(_, _) =>
        case ArrayAggScan(a, name, query) =>
          val aE = compute(a, baseEnv)
          val qE = compute(query, baseEnv.copy(scan = Some(Env.empty)))
          aE.merge(qE.copy(eval = qE.eval.bindIterable(qE.scan.get.m - name)))
        case ArrayAgg(a, name, query) =>
          val aE = compute(a, baseEnv)
          val qE = compute(query, baseEnv.copy(agg = Some(Env.empty)))
          aE.merge(qE.copy(eval = qE.eval.bindIterable(qE.agg.get.m - name)))
        case _ =>
          ir1.children
            .iterator
            .zipWithIndex
            .map {
              case (child: IR, i) =>
                val base = ChildEnvWithoutBindings(ir1, i, base)
                val sub = compute(child, ChildEnvWithoutBindings(ir1, i, base))
                  .subtract(NewBindings(ir1, i, base))
                if (UsesAggEnv(ir1, i))
                  sub.copy(agg = Some(sub.eval))
                else if (UsesScanEnv(ir1, i))
                  sub.copy(scan= Some(sub.eval))
                else
                  sub
              case _ =>
                baseEnv
            }
          .fold(baseEnv)(_.merge(_))
      }
    }

    compute(ir, BindingEnv(Env.empty,
      if (supportsAgg) Some(Env.empty) else None,
      if (supportsScan) Some(Env.empty) else None))
  }
}
