package is.hail.expr.ir.functions

import is.hail.expr.ir._

object AnonymizeBindings {
  // FIXME: only rewrite non-anonymous symbols when those arrive
  private val emptyEnv = Env.empty[Ref]

  private def breaksScope(x: BaseIR): Boolean = {
    (x: @unchecked) match {
      case _: TableAggregate => true
      case _: MatrixAggregate => true
      case _: TableCollect => true
      case _: IR => false
      case _ => true
    }
  }

  private def anonymize(ir: BaseIR, env: Env[Ref]): BaseIR = {
    (ir: @unchecked) match {
      case r: Ref => env.lookupOption(r.name).getOrElse(r)
      case x if breaksScope(x) => ir.copy(ir.children.map(anonymize(_, emptyEnv)))
      case ir: IR =>
        val bindings = Bindings.getRefs(ir)
        if (bindings.isEmpty)
          ir.copy(ir.children.map(anonymize(_, env)))
        else {
          val newBindings = bindings.map(_ => genUID())
          Bindings.copyBindings(ir, newBindings, ir.children.zipWithIndex.map { case (c, i) =>
            val substEnv = bindings.zip(newBindings).foldLeft(env) { case (e, (binding, newBinding)) =>
              if (Binds(ir, binding.name, i))
                e.bind(binding.name, Ref(newBinding, binding.typ))
              else e
            }
            anonymize(c, substEnv)
          })
        }
    }
  }

  def apply(ir: BaseIR): BaseIR = {
    anonymize(ir, Env.empty)
  }
}
