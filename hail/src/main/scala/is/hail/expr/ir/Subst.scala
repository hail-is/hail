package is.hail.expr.ir

import is.hail.utils._

object Subst {
  def apply(e: IR): IR = apply(e, BindingEnv.empty[IR])

  def apply(e: IR, env: BindingEnv[IR]): IR = subst(e, env)

  private def subst(e: IR, env: BindingEnv[IR]): IR = {
    if (env.allEmpty)
      return e

    e match {
      case x@Ref(name, _) =>
        env.eval.lookupOption(name).getOrElse(x)
      case _ =>
        e.copy(
          e.children
            .iterator
            .zipWithIndex
            .map { case (child: IR, i) =>

              val childEnv = ChildEnvWithoutBindings(child, i, env)
              val b = Bindings(e, i).map(_._1)
              val ab = AggBindings(e, i).map(_._1)
              val sb = ScanBindings(e, i).map(_._1)

              if (UsesAggEnv(e, i)) {
                subst(child, BindingEnv(childEnv.aggOrEmpty.delete(ab)))
              } else if (UsesScanEnv(e, i)) {
                subst(child, BindingEnv(childEnv.scanOrEmpty.delete(sb)))
              } else {
                if (b.isEmpty && ab.isEmpty && sb.isEmpty) // optimize the common case
                  subst(child, childEnv)
                else {
                  subst(child,
                    childEnv.copy(eval = childEnv.eval.delete(b),
                      agg = childEnv.agg.map(_.delete(ab)),
                      scan = childEnv.scan.map(_.delete(sb)))
                  )
                }
              }
            case (child, _) => child
            }.toFastIndexedSeq)

    }
  }
}
