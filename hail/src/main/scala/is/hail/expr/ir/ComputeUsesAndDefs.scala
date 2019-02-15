package is.hail.expr.ir

import scala.collection.mutable

case class UsesAndDefs(uses: Memo[mutable.Set[RefEquality[Ref]]], defs: Memo[RefEquality[IR]])

object ComputeUsesAndDefs {
  def apply(ir0: IR): UsesAndDefs = {
    val uses = Memo.empty[mutable.Set[RefEquality[Ref]]]
    val defs = Memo.empty[RefEquality[IR]]

    val inAgg = InAgg(ir0)

    def memoize(ir1: IR, env: Env[RefEquality[IR]], aggEnv: Env[RefEquality[IR]]) {
      ir1 match {
        case r@Ref(name, _) =>
          (if (inAgg.lookup(r)) aggEnv.lookupOption(name) else env.lookupOption(name)).foreach { decl =>
            val re = RefEquality(r)
            uses.lookup(decl) += re
            defs.bind(re, decl)
          }
        case ir =>
          val bindings = Bindings(ir)
          val aggBindings = AggBindings(ir)

          if (bindings.nonEmpty || aggBindings.nonEmpty)
              uses.bind(ir, mutable.Set.empty[RefEquality[Ref]])
          Children(ir).iterator.zipWithIndex.foreach {
            case (child: IR, idx) =>
              var env_ = env
              var aggEnv_ = aggEnv

              bindings.foreach { binding =>
                if (Binds(ir, binding, idx)) {
                  env_ = env_.bind(binding, RefEquality(ir))
                }
              }
              aggBindings.foreach { binding =>
                if (AggBinds(ir, binding, idx)) {
                  aggEnv_ = aggEnv_.bind(binding, RefEquality(ir))
                }
              }
              memoize(child, env_, aggEnv_)
            case _ =>
          }
      }
    }

    memoize(ir0, Env.empty, Env.empty)
    UsesAndDefs(uses, defs)
  }
}
