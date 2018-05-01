package is.hail.expr.ir

import is.hail.expr.types.TAggregable

object Subst {
  def apply(e: IR): IR = apply(e, Env.empty, Env.empty, None)
  def apply(e: IR, env: Env[IR]): IR = apply(e, env, Env.empty, None)
  def apply(e: IR, env: Env[IR], aggEnv: Env[IR], aggTyp: Option[TAggregable]): IR = {
    def subst(e: IR, env: Env[IR] = env, aggEnv: Env[IR] = aggEnv, aggTyp: Option[TAggregable] = aggTyp): IR = apply(e, env, aggEnv, aggTyp)

    e match {
      case x@Ref(name, typ) =>
        env.lookupOption(name).getOrElse(x)
      case x@AggIn(typ) =>
        aggTyp.map(AggIn).getOrElse(x)
      case Let(name, v, body) =>
        Let(name, subst(v), subst(body, env.delete(name)))
      case ArrayMap(a, name, body) =>
        ArrayMap(subst(a), name, subst(body, env.delete(name)))
      case ArrayFilter(a, name, cond) =>
        ArrayFilter(subst(a), name, subst(cond, env.delete(name)))
      case ArrayFlatMap(a, name, body) =>
        ArrayFlatMap(subst(a), name, subst(body, env.delete(name)))
      case ArrayFold(a, zero, accumName, valueName, body) =>
        ArrayFold(subst(a), subst(zero), accumName, valueName, subst(body, env.delete(accumName).delete(valueName)))
      case ArrayFor(a, valueName, body) =>
        ArrayFor(subst(a), valueName, subst(body, env.delete(valueName)))
      case ApplyAggOp(a, op, args) =>
        val substitutedArgs = args.map(arg => Recur(subst(_))(arg))
        ApplyAggOp(subst(a, aggEnv, Env.empty), op, substitutedArgs)
      case _ =>
        Recur(subst(_))(e)
    }
  }
}
