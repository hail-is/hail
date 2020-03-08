package is.hail.expr.ir

object FreeVariables {
  def apply(ir: IR, supportsAgg: Boolean, supportsScan: Boolean): BindingEnv[Unit] = {

    def compute(ir1: IR, baseEnv: BindingEnv[Unit]): BindingEnv[Unit] = {
      ir1 match {
        case Ref(name, _) =>
          baseEnv.bindEval(name, ())
        case TableAggregate(_, _) => baseEnv
        case MatrixAggregate(_, _) => baseEnv
        case StreamAggScan(a, name, query) =>
          val aE = compute(a, baseEnv)
          val qE = compute(query, baseEnv.copy(scan = Some(Env.empty)))
          aE.merge(qE.copy(eval = qE.eval.bindIterable(qE.scan.get.m - name), scan = None))
        case StreamAgg(a, name, query) =>
          val aE = compute(a, baseEnv)
          val qE = compute(query, baseEnv.copy(agg = Some(Env.empty)))
          aE.merge(qE.copy(eval = qE.eval.bindIterable(qE.agg.get.m - name), agg = None))
        case _ =>
          ir1.children
            .iterator
            .zipWithIndex
            .map {
              case (child: IR, i) =>
                val childEnv = ChildEnvWithoutBindings(ir1, i, baseEnv)
                val sub = compute(child, childEnv)
                  .subtract(NewBindings(ir1, i, childEnv))
                if (UsesAggEnv(ir1, i))
                  sub.copy(eval = Env.empty[Unit], agg = Some(sub.eval), scan = baseEnv.scan)
                else if (UsesScanEnv(ir1, i))
                  sub.copy(eval = Env.empty[Unit], agg = baseEnv.agg, scan = Some(sub.eval))
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
