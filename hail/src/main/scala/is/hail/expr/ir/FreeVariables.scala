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
          aE.merge(qE.copy(eval = qE.eval.bindIterable(qE.scan.get.m - name), scan = baseEnv.scan))
        case StreamAgg(a, name, query) =>
          val aE = compute(a, baseEnv)
          val qE = compute(query, baseEnv.copy(agg = Some(Env.empty)))
          aE.merge(qE.copy(eval = qE.eval.bindIterable(qE.agg.get.m - name), agg = baseEnv.agg))
        case ApplyAggOp(init, seq, _) =>
          val initEnv = baseEnv.copy(agg = None)
          val initFreeVars = init.iterator.map(x => compute(x, initEnv)).fold(initEnv)(_.merge(_))
            .copy(agg = Some(Env.empty[Unit]))
          val seqEnv = baseEnv.promoteAgg
          seq.iterator.map { x =>
            val e = compute(x, seqEnv)
            e.copy(eval = Env.empty[Unit], agg = Some(e.eval))
          }.fold(initFreeVars)(_.merge(_))
        case ApplyScanOp(init, seq, _) =>
          val initEnv = baseEnv.copy(scan = None)
          val initFreeVars = init.iterator.map(x => compute(x, initEnv)).fold(initEnv)(_.merge(_))
            .copy(scan = Some(Env.empty[Unit]))
          val seqEnv = baseEnv.promoteScan
          seq.iterator.map { x =>
            val e = compute(x, seqEnv)
            e.copy(eval = Env.empty[Unit], scan = Some(e.eval))
          }.fold(initFreeVars)(_.merge(_))
        case AggFold(zero, seqOp, combOp, accumName, otherAccumName, isScan) =>
          val zeroEnv = if (isScan) baseEnv.copy(scan = None) else baseEnv.copy(agg = None)
          val zeroFreeVarsCompute = compute(zero, zeroEnv)
          val zeroFreeVars = if (isScan) zeroFreeVarsCompute.copy(scan = Some(Env.empty[Unit]))
          else zeroFreeVarsCompute.copy(agg = Some(Env.empty[Unit]))
          val seqOpEnv = if (isScan) baseEnv.promoteScan else baseEnv.promoteAgg
          val seqOpFreeVarsCompute = compute(seqOp, seqOpEnv)
          val seqOpFreeVars = if (isScan) {
            seqOpFreeVarsCompute.copy(
              eval = Env.empty[Unit],
              scan = Some(seqOpFreeVarsCompute.eval),
            )
          } else {
            seqOpFreeVarsCompute.copy(eval = Env.empty[Unit], agg = Some(seqOpFreeVarsCompute.eval))
          }
          val combEval = Env.fromSeq(IndexedSeq((accumName, {}), (otherAccumName, {})))
          val combOpFreeVarsCompute = compute(combOp, baseEnv.copy(eval = combEval))
          val combOpFreeVars = combOpFreeVarsCompute.copy(
            eval = Env.empty[Unit],
            scan = Some(combOpFreeVarsCompute.eval),
          )
          zeroFreeVars.merge(seqOpFreeVars).merge(combOpFreeVars)
        case _ =>
          ir1.children
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

    compute(
      ir,
      BindingEnv(
        Env.empty,
        if (supportsAgg) Some(Env.empty[Unit]) else None,
        if (supportsScan) Some(Env.empty[Unit]) else None,
      ),
    )
  }
}
