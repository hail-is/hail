package is.hail.expr.ir

class NormalizeNames {
  var count: Int = 0

  def gen(): String = {
    count += 1
    count.toString
  }

  def apply(ir: IR, env: Env[String]): IR = apply(ir, env, None)

  def apply(ir: IR, env: Env[String], aggEnv: Option[Env[String]]): IR = {
    def normalize(ir: IR, env: Env[String] = env, aggEnv: Option[Env[String]] = aggEnv): IR = apply(ir, env, aggEnv)

    ir match {
      case Let(name, value, body) =>
        val newName = gen()
        Let(newName, normalize(value), normalize(body, env.bind(name, newName)))
      case Ref(name, typ) =>
        Ref(env.lookup(name), typ)
      case AggLet(name, value, body) =>
        val newName = gen()
        AggLet(newName, normalize(value), normalize(body, env, Some(aggEnv.get.bind(name, newName))))
      case ArraySort(a, left, right, compare) =>
        val newLeft = gen()
        val newRight = gen()
        ArraySort(normalize(a), newLeft, newRight, normalize(compare, env.bind(left -> newLeft, right -> newRight)))
      case ArrayMap(a, name, body) =>
        val newName = gen()
        ArrayMap(normalize(a), newName, normalize(body, env.bind(name, newName)))
      case ArrayFilter(a, name, body) =>
        val newName = gen()
        ArrayFilter(normalize(a), newName, normalize(body, env.bind(name, newName)))
      case ArrayFlatMap(a, name, body) =>
        val newName = gen()
        ArrayFlatMap(normalize(a), newName, normalize(body, env.bind(name, newName)))
      case ArrayFold(a, zero, accumName, valueName, body) =>
        val newAccumName = gen()
        val newValueName = gen()
        ArrayFold(normalize(a), normalize(zero), newAccumName, newValueName, normalize(body, env.bind(accumName -> newAccumName, valueName -> newValueName)))
      case ArrayScan(a, zero, accumName, valueName, body) =>
        val newAccumName = gen()
        val newValueName = gen()
        ArrayScan(normalize(a), normalize(zero), newAccumName, newValueName, normalize(body, env.bind(accumName -> newAccumName, valueName -> newValueName)))
      case ArrayFor(a, valueName, body) =>
        val newValueName = gen()
        ArrayFor(normalize(a), newValueName, normalize(body, env.bind(valueName, newValueName)))
      case ArrayAgg(a, name, body) =>
        assert(aggEnv.isEmpty)
        val newName = gen()
        ArrayAgg(normalize(a), newName, normalize(body, env, Some(env.bind(name, newName))))
      case ArrayLeftJoinDistinct(left, right, l, r, keyF, joinF) =>
        val newL = gen()
        val newR = gen()
        val newEnv = env.bind(l -> newL, r -> newR)
        ArrayLeftJoinDistinct(normalize(left), normalize(right), newL, newR, normalize(keyF, newEnv), normalize(joinF, newEnv))
      case AggExplode(a, name, aggBody) =>
        val newName = gen()
        AggExplode(normalize(a, aggEnv.get, None), newName, normalize(aggBody, env, Some(aggEnv.get.bind(name, newName))))
      case AggFilter(cond, aggIR) =>
        AggFilter(normalize(cond, aggEnv.get, None), normalize(aggIR))
      case AggGroupBy(key, aggIR) =>
        AggGroupBy(normalize(key, aggEnv.get, None), normalize(aggIR))
      case AggArrayPerElement(a, name, aggBody) =>
        val newName = gen()
        AggArrayPerElement(normalize(a, aggEnv.get, None), newName, normalize(aggBody, env, Some(aggEnv.get.bind(name, newName))))
      case ApplyAggOp(ctorArgs, initOpArgs, seqOpArgs, aggSig) =>
        ApplyAggOp(ctorArgs.map(a => normalize(a)),
          initOpArgs.map(_.map(a => normalize(a))),
          seqOpArgs.map(a => normalize(a, aggEnv.get, None)),
          aggSig)
      case Uniroot(argname, function, min, max) =>
        val newArgname = gen()
        Uniroot(newArgname, normalize(function, env.bind(argname, newArgname)), normalize(min), normalize(max))
      case _ =>
        // FIXME when Binding lands, assert nothing is bound in any child
        Copy(ir, ir.children.map {
          case c: IR => normalize(c)
        })
    }
  }
}
