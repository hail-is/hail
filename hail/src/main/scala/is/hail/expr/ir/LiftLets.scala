package is.hail.expr.ir

import scala.collection.mutable

case class LetBinding(name: String, value: IR, valueBindings: List[LetBinding], bodyBindings: List[LetBinding])

object LiftLets {

  def renderBindings(lbs: List[LetBinding], indent: Int = 2) {
    lbs.foreach { lb =>
      println(" " * indent + s"* ${ lb.name } => ${ lb.value }")
      if (lb.valueBindings.nonEmpty) {
        println(" " * (indent + 2) + "V:")
        renderBindings(lb.valueBindings, indent + 4)
      }
      if (lb.bodyBindings.nonEmpty) {
        println(" " * (indent + 2) + "B:")
        renderBindings(lb.bodyBindings, indent + 4)
      }
    }
  }

  private def breaksScope(x: BaseIR): Boolean = {
    (x: @unchecked) match {
      case _: TableAggregate => true
      case _: MatrixAggregate => true
      case _: TableCollect => true
      case _: ApplyAggOp => true
      case _: ApplyScanOp => true
      case _: AggFilter => true
      case _: AggGroupBy => true
      case _: AggExplode => true
      case _: IR => false
      case _ => true
    }
  }

  def prependBindings(x: IR, bindings: List[LetBinding]): IR = {
    val m = mutable.Map.empty[IR, String]
    bindings.foldRight(x) { case (binding, ir) =>
      // reduce equivalent lets
      m.get(binding.value) match {
        case Some(prevName) if binding.valueBindings.isEmpty =>
          if (prevName != binding.name)
            Subst(ir, Env(binding.name -> Ref(prevName, binding.value.typ)))
          else prependBindings(ir, binding.bodyBindings)
        case None =>
          m += binding.value -> binding.name
          Let(binding.name, prependBindings(binding.value, binding.valueBindings), prependBindings(ir, binding.bodyBindings))
      }
    }
  }

  def apply(ir0: BaseIR): BaseIR = {
    val (lifted, lb) = lift(ir0)
    if (lb.nonEmpty)
      prependBindings(lifted.asInstanceOf[IR], lb)
    else
      lifted
  }

  def letBindingMentions(lb: LetBinding, name: String): Boolean = {
    assert(lb.name != name)
    Mentions(lb.value, name) ||
      lb.valueBindings.exists(letBindingMentions(_, name)) ||
      lb.bodyBindings.exists(letBindingMentions(_, name))
  }

  def lift(ir0: BaseIR): (BaseIR, List[LetBinding]) = {
    (ir0: @unchecked) match {
      case Let(name, value, body) =>
        val (liftedBody, bodyBindings) = lift(body)
        val (liftedValue: IR, valueBindings) = lift(value)
        val subInclusion = bodyBindings.map(lb => letBindingMentions(lb, name))
        val lb = (LetBinding(
          name,
          liftedValue,
          valueBindings,
          bodyBindings.zip(subInclusion).filter(_._2).map(_._1))
          :: bodyBindings.zip(subInclusion).filter { case (_, sub) => !sub }.map(_._1))
        liftedBody -> lb
      case ir1 if breaksScope(ir1) =>
        ir1.copy(ir1.children.map(apply)) -> Nil
      case ir0: IR =>
        val bindings = Bindings(ir0)
        val (newChildren, letBindings) = Children(ir0)
          .zipWithIndex
          .map { case (c, i) =>
            val (liftedChild, lbs) = lift(c)
            if (lbs.nonEmpty) {
              val subInclusion = lbs.map(lb => bindings.exists(b => letBindingMentions(lb, b)))
              prependBindings(liftedChild.asInstanceOf[IR], lbs.zip(subInclusion).filter(_._2).map(_._1)) -> lbs.zip(subInclusion).filter(t => !t._2).map(_._1)
            } else liftedChild -> Nil
          }.unzip
        ir0.copy(newChildren) -> letBindings.fold(Nil)(_ ::: _)
    }
  }
}