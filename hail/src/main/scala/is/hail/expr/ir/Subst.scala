package is.hail.expr.ir

import is.hail.utils._

object Subst {
  def apply(e: IR): IR = apply(e, BindingEnv.empty[IR])

  def apply(e: IR, env: BindingEnv[IR]): IR = {
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
              val b = Bindings(e, i).map(_._1)
              val ab = AggBindings(e, i).map(_._1)
              val sb = ScanBindings(e, i).map(_._1)

              if (UsesAggEnv(e, i)) {
                assert(b.isEmpty)
                assert(sb.isEmpty)
                if (env.agg.isEmpty)
                  throw new RuntimeException(s"$i: $e")
                apply(child, BindingEnv(env.agg.get.delete(ab)))
              } else if (UsesScanEnv(e, i)) {
                assert(b.isEmpty)
                assert(ab.isEmpty)
                if (env.scan.isEmpty)
                  throw new RuntimeException(s"$i: $e")
                apply(child, BindingEnv(env.agg.get.delete(sb)))
              } else {
                if (b.isEmpty && ab.isEmpty && sb.isEmpty) // optimize the common case
                  apply(child, env)
                else {
                  apply(child,
                    env.copy(eval = env.eval.delete(b),
                      agg = env.agg.map(_.delete(ab)),
                      scan = env.scan.map(_.delete(sb)))
                  )
                }
              }
            case (child, _) => child
            }.toFastIndexedSeq)

    }
  }
}
