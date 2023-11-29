package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.utils.StackSafe._

class NormalizeNames(normFunction: Int => String, allowFreeVariables: Boolean = false) {
  var count: Int = 0

  def gen(): String = {
    count += 1
    normFunction(count)
  }

  def apply(ctx: ExecuteContext, ir: IR, env: Env[String]): IR =
    normalizeIR(ir.noSharing(ctx), BindingEnv(env)).run().asInstanceOf[IR]

  def apply(ctx: ExecuteContext, ir: IR, env: BindingEnv[String]): IR =
    normalizeIR(ir.noSharing(ctx), env).run().asInstanceOf[IR]

  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    normalizeIR(ir.noSharing(ctx), BindingEnv(agg=Some(Env.empty), scan=Some(Env.empty))).run()

  private def normalizeIR(ir: BaseIR, env: BindingEnv[String], context: Array[String] = Array()): StackFrame[BaseIR] = {

    def normalizeBaseIR(next: BaseIR, env: BindingEnv[String] = env): StackFrame[BaseIR] =
      call(normalizeIR(next, env, context :+ ir.getClass().getName()))

    def normalize(next: IR, env: BindingEnv[String] = env): StackFrame[IR] =
      call(normalizeIR(next, env, context :+ ir.getClass().getName()).asInstanceOf[StackFrame[IR]])

    ir match {
      case Let(bindings, body) =>
        val newBindings: Array[(String, IR)] =
          Array.ofDim(bindings.length)

        for {
          (env, _) <- bindings.foldLeft(done((env, 0))) {
            case (get, (name, value)) =>
              for {
                (env, idx) <- get
                newValue <- normalize(value, env)
                newName = gen()
                _ = newBindings(idx) = newName -> newValue
              } yield (env.bindEval(name, newName), idx + 1)
          }
          newBody <- normalize(body, env)
        } yield Let(newBindings, newBody)

      case Ref(name, typ) =>
        val newName = env.eval.lookupOption(name) match {
          case Some(n) => n
          case None =>
            if (!allowFreeVariables)
              throw new RuntimeException(s"found free variable in normalize: $name, ${context.reverse.mkString(", ")}; ${env.pretty(x => x)}")
            else
              name
        }
        done(Ref(newName, typ))
      case Recur(name, args, typ) =>
        val newName = env.eval.lookupOption(name) match {
          case Some(n) => n
          case None =>
            if (!allowFreeVariables)
              throw new RuntimeException(s"found free loop variable in normalize: $name, ${context.reverse.mkString(", ")}; ${env.pretty(x => x)}")
            else
              name
        }
        for {
          newArgs <- args.mapRecur(v => normalize(v))
        } yield Recur(newName, newArgs, typ)
      case AggLet(name, value, body, isScan) =>
        val newName = gen()
        val (valueEnv, bodyEnv) = if (isScan)
          env.promoteScan -> env.bindScan(name, newName)
        else
          env.promoteAgg -> env.bindAgg(name, newName)
        for {
          newValue <- normalize(value, valueEnv)
          newBody <- normalize(body, bodyEnv)
        } yield AggLet(newName, newValue, newBody, isScan)
      case TailLoop(name, args, resultType, body) =>
        val newFName = gen()
        val newNames = Array.tabulate(args.length)(i => gen())
        val (names, values) = args.unzip
        for {
          newValues <- values.mapRecur(v => normalize(v))
          newBody <- normalize(body, env.copy(eval = env.eval.bind(names.zip(newNames) :+ name -> newFName: _*)))
        } yield TailLoop(newFName, newNames.zip(newValues), resultType, newBody)
      case ArraySort(a, left, right, lessThan) =>
        val newLeft = gen()
        val newRight = gen()
        for {
          newA <- normalize(a)
          newLessThan <- normalize(lessThan, env.bindEval(left -> newLeft, right -> newRight))
        } yield ArraySort(newA, newLeft, newRight, newLessThan)
      case StreamMap(a, name, body) =>
        val newName = gen()
        for {
          newA <- normalize(a)
          newBody <- normalize(body, env.bindEval(name, newName))
        } yield StreamMap(newA, newName, newBody)
      case StreamZip(as, names, body, behavior, errorID) =>
        val newNames = names.map(_ => gen())
        for {
          newAs <- as.mapRecur(normalize(_))
          newBody <- normalize(body, env.bindEval(names.zip(newNames): _*))
        } yield StreamZip(newAs, newNames, newBody, behavior, errorID)
      case StreamZipJoin(as, key, curKey, curVals, joinF) =>
        val newCurKey = gen()
        val newCurVals = gen()
        for {
          newAs <- as.mapRecur(normalize(_))
          newJoinF <- normalize(joinF, env.bindEval(curKey -> newCurKey, curVals -> newCurVals))
        } yield StreamZipJoin(newAs, key, newCurKey, newCurVals, newJoinF)
      case StreamZipJoinProducers(contexts, ctxName, makeProducer, key, curKey, curVals, joinF) =>
        val newCtxName = gen()
        val newCurKey = gen()
        val newCurVals = gen()
        for {
          newCtxs <- normalize(contexts)
          newMakeProducer <- normalize(makeProducer, env.bindEval(ctxName -> newCtxName))
          newJoinF <- normalize(joinF, env.bindEval(curKey -> newCurKey, curVals -> newCurVals))
        } yield StreamZipJoinProducers(newCtxs, newCtxName, newMakeProducer, key, newCurKey, newCurVals, newJoinF)
      case StreamLeftIntervalJoin(left, right, lKeyNames, rIntrvlName, lEltName, rEltName, body) =>
        for {
          newL <- normalize(left)
          newR <- normalize(right)
          newLName = gen()
          newRName = gen()
          newB <- normalize(body, env.bindEval(lEltName -> newLName, rEltName -> newRName))
        } yield StreamLeftIntervalJoin(newL, newR, lKeyNames, rIntrvlName, newLName, newRName, newB)
      case StreamFilter(a, name, body) =>
        val newName = gen()
        for {
          newA <- normalize(a)
          newBody <- normalize(body, env.bindEval(name, newName))
        } yield StreamFilter(newA, newName, newBody)
      case StreamTakeWhile(a, name, body) =>
        val newName = gen()
        for {
          newA <- normalize(a)
            newBody <- normalize(body, env.bindEval(name, newName))
        } yield StreamTakeWhile(newA, newName, newBody)
      case StreamDropWhile(a, name, body) =>
        val newName = gen()
        for {
          newA <- normalize(a)
            newBody <- normalize(body, env.bindEval(name, newName))
        } yield StreamDropWhile(newA, newName, newBody)
      case StreamFlatMap(a, name, body) =>
        val newName = gen()
        for {
          newA <- normalize(a)
          newBody <- normalize(body, env.bindEval(name, newName))
        } yield StreamFlatMap(newA, newName, newBody)
      case StreamFold(a, zero, accumName, valueName, body) =>
        val newAccumName = gen()
        val newValueName = gen()
        for {
          newA <- normalize(a)
          newZero <- normalize(zero)
          newBody <- normalize(body, env.bindEval(accumName -> newAccumName, valueName -> newValueName))
        } yield StreamFold(newA, newZero, newAccumName, newValueName, newBody)
      case StreamFold2(a, accum, valueName, seq, res) =>
        val newValueName = gen()
        for {
          newA <- normalize(a)
          newAccum <- accum.mapRecur { case (old, ir) =>
            val newName = gen()
            for {
              newIr <- normalize(ir)
            } yield ((old, newName), (newName, newIr))
          }
          (accNames, newAcc) = newAccum.unzip
          resEnv = env.bindEval(accNames: _*)
          seqEnv = resEnv.bindEval(valueName, newValueName)
          newSeq <- seq.mapRecur(normalize(_, seqEnv))
          newRes <- normalize(res, resEnv)
        } yield StreamFold2(newA, newAcc, newValueName, newSeq, newRes)
      case StreamScan(a, zero, accumName, valueName, body) =>
        val newAccumName = gen()
        val newValueName = gen()
        for {
          newA <- normalize(a)
          newZero <- normalize(zero)
          newBody <- normalize(body, env.bindEval(accumName -> newAccumName, valueName -> newValueName))
        } yield StreamScan(newA, newZero, newAccumName, newValueName, newBody)
      case StreamFor(a, valueName, body) =>
        val newValueName = gen()
        for {
          newA <- normalize(a)
          newBody <- normalize(body, env.bindEval(valueName, newValueName))
        } yield StreamFor(newA, newValueName, newBody)
      case StreamAgg(a, name, body) =>
        // FIXME: Uncomment when bindings are threaded through test suites
        // assert(env.agg.isEmpty)
        val newName = gen()
        for {
          newA <- normalize(a)
          newBody <- normalize(body, env.copy(agg = Some(env.eval.bind(name, newName))))
        } yield
        StreamAgg(newA, newName, newBody)
      case RunAggScan(a, name, init, seq, result, sig) =>
        val newName = gen()
        val e2 = env.bindEval(name, newName)
        for {
          newA <- normalize(a)
          newInit <- normalize(init, env)
          newSeq <- normalize(seq, e2)
          newResult <- normalize(result, e2)
        } yield RunAggScan(newA, newName, newInit, newSeq, newResult, sig)
      case StreamAggScan(a, name, body) =>
        // FIXME: Uncomment when bindings are threaded through test suites
        // assert(env.scan.isEmpty)
        val newName = gen()
        val newEnv = env.eval.bind(name, newName)
        for {
          newA <- normalize(a)
          newBody <- normalize(body, env.copy(eval = newEnv, scan = Some(newEnv)))
        } yield StreamAggScan(newA, newName, newBody)
      case StreamJoinRightDistinct(left, right, lKey, rKey, l, r, joinF, joinType) =>
        val newL = gen()
        val newR = gen()
        val newEnv = env.bindEval(l -> newL, r -> newR)
        for {
          newLeft <- normalize(left)
          newRight <- normalize(right)
          newJoinF <- normalize(joinF, newEnv)
        } yield StreamJoinRightDistinct(newLeft, newRight, lKey, rKey, newL, newR, newJoinF, joinType)
      case NDArrayMap(nd, name, body) =>
        val newName = gen()
        for {
          newNd <- normalize(nd)
          newBody <- normalize(body, env.bindEval(name -> newName))
        } yield NDArrayMap(newNd, newName, newBody)
      case NDArrayMap2(l, r, lName, rName, body, errorID) =>
        val newLName = gen()
        val newRName = gen()
        for {
          newL <- normalize(l)
          newR <- normalize(r)
          newBody <- normalize(body, env.bindEval(lName -> newLName, rName -> newRName))
        } yield NDArrayMap2(newL, newR, newLName, newRName, newBody, errorID)
      case AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, isScan) =>
        val newElementName = gen()
        val newIndexName = gen()
        val (aEnv, bodyEnv) = if (isScan)
          env.promoteScan -> env.bindScan(elementName, newElementName)
        else
          env.promoteAgg -> env.bindAgg(elementName, newElementName)
        for {
          newA <- normalize(a, aEnv)
          newAggBody <- normalize(aggBody, bodyEnv.bindEval(indexName, newIndexName))
          newKnownLength <- knownLength.mapRecur(normalize(_, env))
        } yield AggArrayPerElement(newA, newElementName, newIndexName, newAggBody, newKnownLength, isScan)
      case CollectDistributedArray(ctxs, globals, cname, gname, body, dynamicID, staticID, tsd) =>
        val newC = gen()
        val newG = gen()
        for {
          newCtxs <- normalize(ctxs)
          newGlobals <- normalize(globals)
          newBody <- normalize(body, BindingEnv.eval(cname -> newC, gname -> newG))
          newDynamicID <- normalize(dynamicID)
        } yield CollectDistributedArray(newCtxs, newGlobals, newC, newG, newBody, newDynamicID, staticID, tsd)
      case RelationalLet(name, value, body) =>
        val newName = gen()
        for {
          newValue <- normalize(value, env)
          newBody <- normalize(body, env.noAgg.noScan.bindRelational(name, newName))
        } yield RelationalLet(newName, newValue, newBody)
      case RelationalRef(name, typ) =>
        val newName = env.relational.lookupOption(name).getOrElse(
          if (!allowFreeVariables) throw new RuntimeException(s"found free variable in normalize: $name, ${context.reverse.mkString(", ")}; ${env.pretty(x => x)}")
          else name
        )
        done(RelationalRef(newName, typ))

      case x =>
        x.mapChildrenWithIndexStackSafe { (child, i) =>
          normalizeBaseIR(child, ChildBindings.transformed(x, i, env, { case (name, _) => name }))
        }
    }
  }
}
