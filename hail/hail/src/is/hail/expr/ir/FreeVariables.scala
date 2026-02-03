package is.hail.expr.ir

import is.hail.expr.ir.defs.Ref
import is.hail.types.virtual.Type

import scala.collection.mutable

case class FreeVariableEnv(boundVars: Env[Unit], freeVars: mutable.Set[Name]) {
  def this(boundVars: Env[Unit]) =
    this(boundVars, mutable.Set.empty)

  private def copy(boundVars: Env[Unit]): FreeVariableEnv =
    new FreeVariableEnv(boundVars, freeVars)

  def visitRef(name: Name): Unit =
    if (!boundVars.contains(name))
      freeVars += name

  def bindIterable(bindings: Seq[(Name, Type)]): FreeVariableEnv =
    copy(boundVars.bindIterable(bindings.view.map(b => (b._1, ()))))

  def getFreeVars: Env[Unit] = new Env[Unit].bindIterable(freeVars.view.map(n => (n, ())))
}

case class FreeVariableBindingEnv(
  evalVars: Option[FreeVariableEnv],
  aggVars: Option[FreeVariableEnv],
  scanVars: Option[FreeVariableEnv],
) extends GenericBindingEnv[FreeVariableBindingEnv, Type] {
  override def extend(bindings: Bindings[Type]): FreeVariableBindingEnv = {
    val Bindings(all, eval, agg, scan, relational, dropEval) = bindings
    var newEnv = this
    if (dropEval) newEnv = newEnv.noEval
    agg match {
      case AggEnv.Drop => newEnv = newEnv.noAgg
      case AggEnv.Promote => newEnv = newEnv.promoteAgg
      case AggEnv.Create(bindings) =>
        newEnv = newEnv.createAgg.bindAgg(bindings.map(all): _*)
      case AggEnv.Bind(bindings) =>
        newEnv = newEnv.bindAgg(bindings.map(all): _*)
      case _ =>
    }
    scan match {
      case AggEnv.Drop => newEnv = newEnv.noScan
      case AggEnv.Promote => newEnv = newEnv.promoteScan
      case AggEnv.Create(bindings) =>
        newEnv = newEnv.createScan.bindScan(bindings.map(all): _*)
      case AggEnv.Bind(bindings) =>
        newEnv = newEnv.bindScan(bindings.map(all): _*)
      case _ =>
    }
    if (eval.nonEmpty) newEnv = newEnv.bindEval(eval.map(all): _*)
    if (relational.nonEmpty)
      newEnv = newEnv.bindRelational(relational.map(all): _*)
    newEnv
  }

  def visitRef(name: Name): Unit =
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

  override def bindEval(bindings: (Name, Type)*): FreeVariableBindingEnv =
    copy(evalVars = evalVars.map(_.bindIterable(bindings)))

  override def noEval: FreeVariableBindingEnv = copy(evalVars = None)

  override def bindAgg(bindings: (Name, Type)*): FreeVariableBindingEnv =
    copy(aggVars = aggVars.map(_.bindIterable(bindings)))

  override def bindScan(bindings: (Name, Type)*): FreeVariableBindingEnv =
    copy(scanVars = scanVars.map(_.bindIterable(bindings)))

  override def createAgg: FreeVariableBindingEnv = copy(aggVars = evalVars)

  override def createScan: FreeVariableBindingEnv = copy(scanVars = evalVars)

  override def noAgg: FreeVariableBindingEnv = copy(aggVars = None)

  override def noScan: FreeVariableBindingEnv = copy(scanVars = None)

  override def onlyRelational(keepAggCapabilities: Boolean): FreeVariableBindingEnv =
    FreeVariableBindingEnv(None, None, None)

  override def bindRelational(bindings: (Name, Type)*): FreeVariableBindingEnv =
    this
}

object FreeVariables {
  def apply(ir: IR, supportsAgg: Boolean, supportsScan: Boolean): BindingEnv[Unit] = {
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
}
