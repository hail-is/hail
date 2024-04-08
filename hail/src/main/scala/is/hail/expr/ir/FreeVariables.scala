package is.hail.expr.ir

import is.hail.types.virtual.Type

import scala.collection.mutable

class FreeVariableEnv(boundVars: Env[Unit], freeVars: mutable.Set[String]) {
  def this(boundVars: Env[Unit]) =
    this(boundVars, mutable.Set.empty)

  private def copy(boundVars: Env[Unit]): FreeVariableEnv =
    new FreeVariableEnv(boundVars, freeVars)

  def visitRef(name: String): Unit =
    if (!boundVars.contains(name))
      freeVars += name

  def bindIterable(bindings: Seq[(String, Type)]): FreeVariableEnv =
    copy(boundVars.bindIterable(bindings.view.map(b => (b._1, ()))))

  def getFreeVars: Env[Unit] = new Env[Unit].bindIterable(freeVars.view.map(n => (n, ())))
}

case class FreeVariableBindingEnv(
  evalVars: Option[FreeVariableEnv],
  aggVars: Option[FreeVariableEnv],
  scanVars: Option[FreeVariableEnv],
) extends GenericBindingEnv[FreeVariableBindingEnv, Type] {
  def visitRef(name: String): Unit =
    evalVars.foreach(_.visitRef(name))

  def getFreeVars: BindingEnv[Unit] = BindingEnv(
    evalVars.map(_.getFreeVars).getOrElse(Env.empty),
    aggVars.map(_.getFreeVars),
    scanVars.map(_.getFreeVars),
  )

  override def promoteAgg: FreeVariableBindingEnv =
    copy(evalVars = aggVars, aggVars = None)

  override def promoteScan: FreeVariableBindingEnv =
    copy(evalVars = scanVars, scanVars = None)

  override def bindEval(bindings: (String, Type)*): FreeVariableBindingEnv =
    copy(evalVars = evalVars.map(_.bindIterable(bindings)))

//  override def bindEvalAndAggScan(bindings: (String, Type)*): FreeVariableBindingEnv =
//    copy(
//      evalVars = evalVars.map(_.bindIterable(bindings)),
//      aggVars = aggVars.map(_.bindIterable(bindings)),
//      scanVars = scanVars.map(_.bindIterable(bindings)),
//    )

  override def dropEval: FreeVariableBindingEnv = copy(evalVars = None)

  override def bindAgg(bindings: (String, Type)*): FreeVariableBindingEnv =
    copy(aggVars = aggVars.map(_.bindIterable(bindings)))

  override def bindScan(bindings: (String, Type)*): FreeVariableBindingEnv =
    copy(scanVars = scanVars.map(_.bindIterable(bindings)))

  override def createAgg: FreeVariableBindingEnv = copy(aggVars = evalVars)

  override def createScan: FreeVariableBindingEnv = copy(scanVars = evalVars)

  override def noAgg: FreeVariableBindingEnv = copy(aggVars = None)

  override def noScan: FreeVariableBindingEnv = copy(scanVars = None)

//  override def emptyAgg: FreeVariableBindingEnv = copy(aggVars = None)
//
//  override def emptyScan: FreeVariableBindingEnv = copy(scanVars = None)

  override def onlyRelational(keepAggCapabilities: Boolean): FreeVariableBindingEnv =
    FreeVariableBindingEnv(None, None, None)

  override def bindRelational(bindings: (String, Type)*): FreeVariableBindingEnv =
    this
}

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
        case AggFold(zero, seqOp, _, accumName, _, isScan) =>
          val zeroEnv = if (isScan) baseEnv.copy(scan = None) else baseEnv.copy(agg = None)
          val zeroFreeVarsCompute = compute(zero, zeroEnv)
          val zeroFreeVars = if (isScan) zeroFreeVarsCompute.copy(scan = Some(Env.empty[Unit]))
          else zeroFreeVarsCompute.copy(agg = Some(Env.empty[Unit]))
          val seqOpEnv = if (isScan) baseEnv.promoteScan else baseEnv.promoteAgg
          val seqOpFreeVarsCompute = compute(seqOp, seqOpEnv)
          val seqOpFreeVars = if (isScan) {
            seqOpFreeVarsCompute.copy(
              eval = Env.empty[Unit],
              scan = Some(seqOpFreeVarsCompute.eval.delete(accumName)),
            )
          } else {
            seqOpFreeVarsCompute.copy(eval = Env.empty[Unit], agg = Some(seqOpFreeVarsCompute.eval))
          }
          // the comb op can't refer to anything bound outside, so it can't have free variables
          zeroFreeVars.merge(seqOpFreeVars)
        case _ =>
          ir1.children
            .zipWithIndex
            .map {
              case (child: IR, i) =>
                val bindings = Bindings.segregated(ir1, i, baseEnv)
                val childEnv = bindings.childEnvWithoutBindings
                val sub = compute(child, childEnv).subtract(bindings.newBindings)
                val env = if (UsesAggEnv(ir1, i))
                  sub.copy(eval = Env.empty[Unit], agg = Some(sub.eval), scan = baseEnv.scan)
                else if (UsesScanEnv(ir1, i))
                  sub.copy(eval = Env.empty[Unit], agg = baseEnv.agg, scan = Some(sub.eval))
                else
                  sub
                assert(
                  (env.agg.isDefined == baseEnv.agg.isDefined || env.scan.isDefined == baseEnv.scan.isDefined)
                )
                env
              case _ =>
                baseEnv
            }
            .fold(baseEnv)(_.merge(_))
      }
    }

    val old = compute(
      ir,
      BindingEnv(
        Env.empty,
        if (supportsAgg) Some(Env.empty[Unit]) else None,
        if (supportsScan) Some(Env.empty[Unit]) else None,
      ),
    )

    val new_ = {
      val env = FreeVariableBindingEnv(
        Some(new FreeVariableEnv(Env.empty)),
        if (supportsAgg) Some(new FreeVariableEnv(Env.empty)) else None,
        if (supportsScan) Some(new FreeVariableEnv(Env.empty)) else None,
      )
      VisitIR.withEnv(ir, env) { (ir, env) =>
        ir match {
          case Ref(name, _) => env.visitRef(name)
          case _ =>
        }
      }
      env.getFreeVars
    }

    assert(old == new_, s"old: $old\nnew: ${new_}\nir:\n${Pretty.sexprStyle(ir, allowUnboundRefs = true)}")
    old
  }
}
