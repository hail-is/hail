package is.hail.expr.ir

object ForwardLets {
  def extract(x: IR): (String, IR, IR) = (x: @unchecked) match {
    case Let(name, value, body) => (name, value, body)
    case AggLet(name, value, body) => (name, value, body)
  }

  private def rewriteIR(ir0: IR): IR = {
    val UsesAndDefs(uses, _) = ComputeUsesAndDefs(ir0)
    val nestingDepth = NestingDepth(ir0)

    def rewrite(ir: IR, env: Env[IR]): IR = {
      ir match {
        case _: Let | _: AggLet =>
          val (name, value, body) = extract(ir)
          val refs = uses.lookup(ir)

          if (value.isInstanceOf[Ref] ||
            refs.isEmpty ||
            (refs.size == 1 && nestingDepth.lookup(refs.head) == nestingDepth.lookup(ir)))
            rewrite(body, env.bind(name -> value))
          else
            MapIR(rewrite(_, env))(ir)
        case x@Ref(name, _) =>
          env.lookupOption(name) match {
            case Some(fwd) => fwd
            case None => x
          }
        case _ =>
          MapIR(rewrite(_, env))(ir)
      }
    }

    rewrite(ir0, Env.empty)
  }

  def apply(ir0: BaseIR, needsCopy: Boolean = true): BaseIR =
    MapIRSubtrees(rewriteIR)(
      new NormalizeNames().apply(
        (if (needsCopy) ir0.deepCopy() else ir0).asInstanceOf[IR],
        Env.empty[String],
        Some(Env.empty[String]),
        freeVariables = true))
}
