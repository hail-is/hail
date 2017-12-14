package is.hail.expr.ir

object Subst {
  def apply(e: IR): IR = apply(e, new Env[IR]())
  def apply(e: IR, env: Env[IR]): IR = {
    def infer(e: IR, env: Env[IR] = env): IR = apply(e, env)
    e match {
      case x@Ref(name, _) => env.lookupOption(name).getOrElse(x)
      case Let(name, v, body, _) =>
        Let(name, infer(v), infer(body, env.delete(name)))
      case MapNA(name, v, body, _) =>
        MapNA(name, infer(v), infer(body, env.delete(name)))
      case ArrayMap(a, name, body, _) =>
        ArrayMap(infer(a), name, infer(body, env.delete(name)))
      case ArrayFold(a, zero, accumName, valueName, body, _) =>
        ArrayFold(infer(a), infer(zero), accumName, valueName, infer(body, env.delete(accumName).delete(valueName)))
      case _ =>
        Recur(infer(_))(e)
    }
  }
}
