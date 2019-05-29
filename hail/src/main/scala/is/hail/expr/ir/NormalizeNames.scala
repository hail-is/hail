package is.hail.expr.ir

import is.hail.utils._

class NormalizeNames(normFunction: Int => String, allowFreeVariables: Boolean = false) {
  var count: Int = 0

  def gen(): String = {
    count += 1
    normFunction(count)
  }

  def apply(ir: IR, env: Env[String]): IR = apply(ir, BindingEnv(env))

  def apply(ir: IR, env: BindingEnv[String]): IR = normalizeIR(ir, env)

  def apply(ir: BaseIR): BaseIR = {
    ir match {
      case ir: IR => normalizeIR(ir, BindingEnv(Env.empty, Some(Env.empty), Some(Env.empty)))
      case tir: TableIR => normalizeTable(tir)
      case mir: MatrixIR => normalizeMatrix(mir)
      case bmir: BlockMatrixIR => normalizeBlockMatrix(bmir)
    }
  }

  private def normalizeTable(tir: TableIR): TableIR = {
    tir.copy(tir
      .children
      .iterator
      .zipWithIndex
      .map {
        case (child: IR, i) => normalizeIR(child, NewBindings(tir, i).mapValuesWithKey({ case (k, _) => k }))
        case (child: TableIR, _) => normalizeTable(child)
        case (child: MatrixIR, _) => normalizeMatrix(child)
        case (child: BlockMatrixIR, _) => normalizeBlockMatrix(child)
      }.toFastIndexedSeq)
  }

  private def normalizeMatrix(mir: MatrixIR): MatrixIR = {
    mir.copy(mir
      .children
      .iterator
      .zipWithIndex
      .map {
        case (child: IR, i) => normalizeIR(child, NewBindings(mir, i).mapValuesWithKey({ case (k, _) => k }))
        case (child: TableIR, _) => normalizeTable(child)
        case (child: MatrixIR, _) => normalizeMatrix(child)
        case (child: BlockMatrixIR, _) => normalizeBlockMatrix(child)
      }.toFastIndexedSeq)
  }

  private def normalizeBlockMatrix(bmir: BlockMatrixIR): BlockMatrixIR = {
    bmir.copy(bmir
      .children
      .iterator
      .zipWithIndex
      .map {
        case (child: IR, i) => normalizeIR(child, NewBindings(bmir, i).mapValuesWithKey({ case (k, _) => k }))
        case (child: TableIR, _) => normalizeTable(child)
        case (child: MatrixIR, _) => normalizeMatrix(child)
        case (child: BlockMatrixIR, _) => normalizeBlockMatrix(child)
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
        // FIXME: Uncomment when bindings are threaded through test suites
        // assert(env.agg.isEmpty)
        val newName = gen()
        ArrayAgg(normalize(a), newName, normalize(body, env.copy(agg = Some(env.eval.bind(name, newName)))))
      case ArrayAggScan(a, name, body) =>
        // FIXME: Uncomment when bindings are threaded through test suites
        // assert(env.scan.isEmpty)
        val newName = gen()
        val newEnv = env.eval.bind(name, newName)
        ArrayAggScan(normalize(a), newName, normalize(body, env.copy(eval = newEnv, scan = Some(newEnv))))
      case ArrayLeftJoinDistinct(left, right, l, r, keyF, joinF) =>
        val newL = gen()
        val newR = gen()
        val newEnv = env.bindEval(l -> newL, r -> newR)
        ArrayLeftJoinDistinct(normalize(left), normalize(right), newL, newR, normalize(keyF, newEnv), normalize(joinF, newEnv))
      case NDArrayMap(nd, name, body) =>
        val newName = gen()
        NDArrayMap(normalize(nd), newName, normalize(body, env.bindEval(name -> newName)))
      case NDArrayMap2(l, r, lName, rName, body) =>
        val newLName = gen()
        val newRName = gen()
        NDArrayMap2(normalize(l), normalize(r), newLName, newRName, normalize(body, env.bindEval(lName -> newLName, rName -> newRName)))
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
      case AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, isScan) =>
        val newElementName = gen()
        val newIndexName = gen()
        val (aEnv, bodyEnv) = if (isScan)
          env.promoteScan -> env.bindScan(elementName, newElementName)
        else
          env.promoteAgg -> env.bindAgg(elementName, newElementName)
        AggArrayPerElement(normalize(a, aEnv), newElementName, newIndexName, normalize(aggBody, bodyEnv.bindEval(indexName, newIndexName)), knownLength.map(normalize(_, env)), isScan)
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
      case TableAggregate(child, query) =>
        TableAggregate(normalizeTable(child),
          normalizeIR(query, BindingEnv(child.typ.globalEnv, agg = Some(child.typ.rowEnv))
            .mapValuesWithKey({ case (k, _) => k })))
      case MatrixAggregate(child, query) =>
        MatrixAggregate(normalizeMatrix(child),
          normalizeIR(query, BindingEnv(child.typ.globalEnv, agg = Some(child.typ.entryEnv))
            .mapValuesWithKey({ case (k, _) => k })))
      case CollectDistributedArray(ctxs, globals, cname, gname, body) =>
        val newC = gen()
        val newG = gen()
        CollectDistributedArray(normalize(ctxs), normalize(globals), newC, newG, normalize(body, BindingEnv.eval(cname -> newC, gname -> newG)))
      case RelationalLet(name, value, body) =>
        RelationalLet(name, normalize(value, BindingEnv.empty), normalize(body))
      case _ =>
        Copy(ir, ir.children.map {
          case child: IR => normalize(child)
          case child: TableIR => normalizeTable(child)
          case child: MatrixIR => normalizeMatrix(child)
          case child: BlockMatrixIR => normalizeBlockMatrix(child)
        })
    }
  }
}
