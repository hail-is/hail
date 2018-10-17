package is.hail.expr.ir

object Subst {
  def apply(e: IR): IR = apply(e, Env.empty, Env.empty)
  def apply(e: IR, env: Env[IR]): IR = apply(e, env, Env.empty)
  def apply(e: IR, env: Env[IR], aggEnv: Env[IR]): IR = {
    def subst(e: IR, env: Env[IR] = env, aggEnv: Env[IR] = aggEnv): IR = apply(e, env, aggEnv)

    e match {
      case x@Ref(name, typ) =>
        env.lookupOption(name).getOrElse(x)
      case Let(name, v, body) =>
        val newv = subst(v)
        Let(name, newv, subst(body, env.delete(name).bind(name, Ref(name, newv.typ))))
      case ArrayMap(a, name, body) =>
        ArrayMap(subst(a), name, subst(body, env.delete(name)))
      case ArrayFilter(a, name, cond) =>
        ArrayFilter(subst(a), name, subst(cond, env.delete(name)))
      case ArrayFlatMap(a, name, body) =>
        ArrayFlatMap(subst(a), name, subst(body, env.delete(name)))
      case ArrayFold(a, zero, accumName, valueName, body) =>
        ArrayFold(subst(a), subst(zero), accumName, valueName, subst(body, env.delete(accumName).delete(valueName)))
      case ArrayScan(a, zero, accumName, valueName, body) =>
        ArrayScan(subst(a), subst(zero), accumName, valueName, subst(body, env.delete(accumName).delete(valueName)))
      case ArrayFor(a, valueName, body) =>
        ArrayFor(subst(a), valueName, subst(body, env.delete(valueName)))
      case AggFilter(cond, aggIR) =>
        AggFilter(subst(cond, aggEnv), subst(aggIR, aggEnv))
      case AggExplode(array, name, aggBody) =>
        AggExplode(subst(array, aggEnv), name, subst(aggBody, aggEnv.delete(name), aggEnv.delete(name)))
      case AggGroupBy(key, aggIR) =>
        AggGroupBy(subst(key, aggEnv), subst(aggIR, aggEnv))
      case ApplyAggOp(seqOpArgs, constructorArgs, initOpArgs, aggSig) =>
        val substSeqOpArgs = seqOpArgs.map(arg => subst(arg, aggEnv, Env.empty))
        val substConstructorArgs = constructorArgs.map(arg => MapIR(subst(_))(arg))
        val substInitOpArgs = initOpArgs.map(initOpArgs =>
          initOpArgs.map(arg => MapIR(subst(_))(arg)))
        ApplyAggOp(substSeqOpArgs, substConstructorArgs, substInitOpArgs, aggSig)
      case _ =>
        MapIR(subst(_))(e)
    }
  }
}
