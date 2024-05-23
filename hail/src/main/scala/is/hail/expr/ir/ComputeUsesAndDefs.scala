package is.hail.expr.ir

import scala.collection.mutable

case class UsesAndDefs(
  uses: Memo[mutable.Set[RefEquality[BaseRef]]],
  defs: Memo[BaseIR],
  free: mutable.Set[RefEquality[BaseRef]],
)

object ComputeUsesAndDefs {
  def apply(ir0: BaseIR, errorIfFreeVariables: Boolean = true): UsesAndDefs = {
    val uses = Memo.empty[mutable.Set[RefEquality[BaseRef]]]
    val defs = Memo.empty[BaseIR]
    val free = mutable.Set.empty[RefEquality[BaseRef]]

    def compute(ir: BaseIR, env: BindingEnv[BaseIR]): Unit = {

      ir match {
        case r: BaseRef =>
          val e = r match {
            case _: Ref => env.eval
            case _: Recur => env.eval
            case _: RelationalRef => env.relational
          }
          e.lookupOption(r.name) match {
            case Some(decl) =>
              if (!defs.contains(r)) {
                val re = RefEquality(r)
                uses.lookup(decl) += re
                defs.bind(re, decl)
              }
            case None =>
              if (errorIfFreeVariables)
                throw new RuntimeException(s"found variable with no definition: ${r.name}")
              else
                free += RefEquality(r)
          }
        case _ =>
      }

      ir.children
        .zipWithIndex
        .foreach { case (child, i) =>
          val e = ChildEnvWithoutBindings(ir, i, env)
          val b = NewBindings(ir, i, env).mapValues[BaseIR](_ => ir)
          if (!b.allEmpty && !uses.contains(ir))
            uses.bind(ir, mutable.Set.empty[RefEquality[BaseRef]])
          val childEnv = e.merge(b)
          compute(child, childEnv)
        }
    }

    compute(ir0, BindingEnv[BaseIR](scan = Some(Env.empty), agg = Some(Env.empty)))

    UsesAndDefs(uses, defs, free)
  }
}
