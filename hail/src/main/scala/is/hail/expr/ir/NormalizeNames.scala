package is.hail.expr.ir

class NormalizeNames {
  var count: Int = 0

  def gen(): String = {
    count += 1
    count.toString
  }

  def apply(ir: IR, env: Env[String]): IR = apply(ir, BindingEnv(env))

  def apply(ir: IR, env: BindingEnv[String]): IR = {
    def normalize(ir: IR, env: BindingEnv[String] = env): IR = apply(ir, env)

    ir match {
      case Let(name, value, body) =>
        val newName = gen()
        Let(newName, normalize(value), normalize(body, env.copy(eval = env.eval.bind(name, newName))))
      case Ref(name, typ) =>
        Ref(env.eval.lookup(name), typ)
      case AggLet(name, value, body, isScan) =>
        val newName = gen()
        val (valueEnv, bodyEnv) = if (isScan)
          env.promoteScan -> env.bindScan(name, newName)
        else
          env.promoteAgg -> env.bindAgg(name, newName)
        AggLet(newName, normalize(value, valueEnv), normalize(body, bodyEnv), isScan)
      case ArraySort(a, left, right, compare) =>
        val newLeft = gen()
        val newRight = gen()
        ArraySort(normalize(a), newLeft, newRight, normalize(compare, env.bindEval(left -> newLeft, right -> newRight)))
      case ArrayMap(a, name, body) =>
        val newName = gen()
        ArrayMap(normalize(a), newName, normalize(body, env.bindEval(name, newName)))
      case ArrayFilter(a, name, body) =>
        val newName = gen()
        ArrayFilter(normalize(a), newName, normalize(body, env.bindEval(name, newName)))
      case ArrayFlatMap(a, name, body) =>
        val newName = gen()
        ArrayFlatMap(normalize(a), newName, normalize(body, env.bindEval(name, newName)))
      case ArrayFold(a, zero, accumName, valueName, body) =>
        val newAccumName = gen()
        val newValueName = gen()
        ArrayFold(normalize(a), normalize(zero), newAccumName, newValueName, normalize(body, env.bindEval(accumName -> newAccumName, valueName -> newValueName)))
      case ArrayScan(a, zero, accumName, valueName, body) =>
        val newAccumName = gen()
        val newValueName = gen()
        ArrayScan(normalize(a), normalize(zero), newAccumName, newValueName, normalize(body, env.bindEval(accumName -> newAccumName, valueName -> newValueName)))
      case ArrayFor(a, valueName, body) =>
        val newValueName = gen()
        ArrayFor(normalize(a), newValueName, normalize(body, env.bindEval(valueName, newValueName)))
      case ArrayAgg(a, name, body) =>
        assert(env.agg.isEmpty)
        val newName = gen()
        ArrayAgg(normalize(a), newName, normalize(body, env.copy(agg = Some(env.eval.bind(name, newName)))))
      case ArrayLeftJoinDistinct(left, right, l, r, keyF, joinF) =>
        val newL = gen()
        val newR = gen()
        val newEnv = env.bindEval(l -> newL, r -> newR)
        ArrayLeftJoinDistinct(normalize(left), normalize(right), newL, newR, normalize(keyF, newEnv), normalize(joinF, newEnv))
      case AggExplode(a, name, aggBody, isScan) =>
        val newName = gen()
        val (aEnv, bodyEnv) = if (isScan)
          env.promoteScan -> env.bindScan(name, newName)
        else
          env.promoteAgg -> env.bindAgg(name, newName)
        AggExplode(normalize(a, aEnv), newName, normalize(aggBody, bodyEnv), isScan)
      case AggFilter(cond, aggIR, isScan) =>
        val condEnv = if (isScan)
          env.promoteScan
        else
          env.promoteAgg
        AggFilter(normalize(cond, condEnv), normalize(aggIR), isScan)
      case AggGroupBy(key, aggIR, isScan) =>
        val keyEnv = if (isScan)
          env.promoteScan
        else
          env.promoteAgg
        AggGroupBy(normalize(key, keyEnv), normalize(aggIR), isScan)
      case AggArrayPerElement(a, elementName, indexName, aggBody, isScan) =>
        val newElementName = gen()
        val newIndexName = gen()
        val (aEnv, bodyEnv) = if (isScan)
          env.promoteScan -> env.bindScan(elementName, newElementName)
        else
          env.promoteAgg -> env.bindAgg(elementName, newElementName)
        AggArrayPerElement(normalize(a, aEnv), newElementName, newIndexName, normalize(aggBody, bodyEnv.bindEval(indexName, newIndexName)), isScan)
      case ApplyAggOp(ctorArgs, initOpArgs, seqOpArgs, aggSig) =>
        ApplyAggOp(ctorArgs.map(a => normalize(a)),
          initOpArgs.map(_.map(a => normalize(a))),
          seqOpArgs.map(a => normalize(a, env.promoteAgg)),
          aggSig)
      case ApplyScanOp(ctorArgs, initOpArgs, seqOpArgs, aggSig) =>
        ApplyScanOp(ctorArgs.map(a => normalize(a)),
          initOpArgs.map(_.map(a => normalize(a))),
          seqOpArgs.map(a => normalize(a, env.promoteScan)),
          aggSig)
      case Uniroot(argname, function, min, max) =>
        val newArgname = gen()
        Uniroot(newArgname, normalize(function, env.bindEval(argname, newArgname)), normalize(min), normalize(max))
      case _ =>
        // FIXME when Binding lands, assert nothing is bound in any child
        Copy(ir, ir.children.map {
          case c: IR => normalize(c)
        })
    }
  }
}
