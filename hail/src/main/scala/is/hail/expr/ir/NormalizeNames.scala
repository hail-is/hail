package is.hail.expr.ir

import is.hail.utils._

class NormalizeNames(stopAtRelational: Boolean = true, allowFreeVariables: Boolean = false) {
  var count: Int = 0

  def gen(): String = {
    count += 1
    count.toString
  }

  def apply(ir: IR, env: Env[String]): IR = apply(ir, BindingEnv(env))

  def apply(ir: IR, env: BindingEnv[String]): IR = normalizeIR(ir, env)

  def apply(ir: BaseIR): BaseIR = {
    ir match {
      case ir: IR => normalizeIR(ir, BindingEnv.empty)
      case baseIR => normalizeBaseIR(baseIR)
    }
  }

  private def normalizeBaseIR(ir0: BaseIR): BaseIR = {
    if (stopAtRelational)
      return ir0

    assert(!ir0.isInstanceOf[IR])

    ir0.copy(ir0.children
      .iterator
      .zipWithIndex
      .map {
        case (ir: IR, i) =>
          val b = Bindings(ir0, i).map { case (binding, _) => binding -> binding }
          val ab = AggBindings(ir0, i).map { case (binding, _) => binding -> binding }
          val sb = ScanBindings(ir0, i).map { case (binding, _) => binding -> binding }

          normalizeIR(ir, BindingEnv(
            Env.fromSeq(b),
            agg = if (ab.nonEmpty) Some(Env.fromSeq(ab)) else None,
            scan = if (sb.nonEmpty) Some(Env.fromSeq(sb)) else None
          ))
        case (child, _) => normalizeBaseIR(child)
      }.toFastIndexedSeq)
  }

  private def normalizeIR(ir: IR, env: BindingEnv[String]): IR = {

    def normalize(ir: IR, env: BindingEnv[String] = env): IR = normalizeIR(ir, env)

    ir match {
      case Let(name, value, body) =>
        val newName = gen()
        Let(newName, normalize(value), normalize(body, env.copy(eval = env.eval.bind(name, newName))))
      case Ref(name, typ) =>
        val newName = env.eval.lookupOption(name) match {
          case Some(n) => n
          case None =>
            if (!allowFreeVariables)
              throw new RuntimeException(s"found free variable in normalize: $name")
            else
              name
        }
        Ref(newName, typ)
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
          case other => normalizeBaseIR(other)
        })
    }
  }
}
