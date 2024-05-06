package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.NormalizeNames.needsRenaming
import is.hail.types.virtual.Type
import is.hail.utils.StackSafe._

import scala.collection.mutable

object NormalizeNames {
  def apply[T <: BaseIR](
    ctx: ExecuteContext,
    ir: T,
    env: BindingEnv[String] = BindingEnv.empty.createAgg.createScan,
    normFunction: Int => String = iruid(_),
    allowFreeVariables: Boolean = false,
  ): T = {
    val noSharing = ir.noSharing(ctx)
    val _ = new NormalizeNames(normFunction, allowFreeVariables).normalizeIROld(
      noSharing,
      env,
    ).run().asInstanceOf[T]
    val old = new NormalizeNames(iruid(_), allowFreeVariables).normalizeIROld(
      noSharing,
      env,
    ).run().asInstanceOf[T]
    val new_ = new NormalizeNames(iruid(_), allowFreeVariables).normalizeIROld(
      noSharing,
      env,
    ).run().asInstanceOf[T]
//    if (env.allEmpty) TypeCheck(ctx, res.asInstanceOf[BaseIR])
    assert(new_ == old, s"new:\n${Pretty.sexprStyle(new_)}\nold:\n${Pretty.sexprStyle(old)}")
    old
  }

  protected def needsRenaming(ir: BaseIR): Boolean = ir match {
    case _: RelationalLetMatrixTable | _: TableGen | _: TableMapPartitions | _: RelationalLetTable =>
      true
    case _: MatrixIR | _: TableIR =>
      false
    case _: TableAggregate | _: MatrixAggregate =>
      false
    case _: IR | _: BlockMatrixIR =>
      true
  }
}

class NormalizeNames(normFunction: Int => String, allowFreeVariables: Boolean = false) {
  var count: Int = 0

  def gen(): String = {
    count += 1
    normFunction(count)
  }

  private def normalizeIR(ir: BaseIR, env: BindingEnv[String]): StackFrame[BaseIR] = {
    val bindingsMap = mutable.AnyRefMap.empty[String, String]
    val updateEnv: (BindingEnv[String], Bindings[Type]) => BindingEnv[String] =
      if (needsRenaming(ir)) { (env, bindings) =>
        val bindingsNames = bindings.map((name, _) => bindingsMap.getOrElseUpdate(name, gen()))
        env.extend(bindingsNames)
      } else { (env, bindings) => env.extend(bindings.dropBindings) }
    ir.mapChildrenWithEnvStackSafe(env, updateEnv)(normalizeIR).map {
      case Ref(name, typ) =>
        val newName = env.eval.lookupOption(name).getOrElse {
          if (!allowFreeVariables) throw new RuntimeException(
            s"found free variable in normalize: $name; ${env.pretty(x => x)}"
          )
          else name
        }
        Ref(newName, typ)
      case RelationalRef(name, typ) =>
        val newName = env.relational.lookupOption(name).getOrElse(
          if (!allowFreeVariables) throw new RuntimeException(
            s"found free variable in normalize: $name; ${env.pretty(x => x)}"
          )
          else name
        )
        RelationalRef(newName, typ)
      case Recur(name, args, typ) =>
        val newName = env.eval.lookupOption(name) match {
          case Some(n) => n
          case None =>
            if (!allowFreeVariables)
              throw new RuntimeException(s"found free loop variable in normalize: $name")
            else
              name
        }
        Recur(newName, args, typ)
      case x => modifyNames(x, bindingsMap)
    }
  }

  private def modifyNames(ir: BaseIR, map: collection.Map[String, String]): BaseIR = {
    def rename(name: String): String = map.getOrElse(name, name)

    ir match {
      case Block(bindings, body) =>
        val newBindings = bindings.map(b => b.copy(name = rename(b.name)))
        Block(newBindings, body)
      case TailLoop(name, args, resultType, body) =>
        TailLoop(rename(name), args.map { case (name, ir) => (rename(name), ir) }, resultType, body)
      case x: StreamMap => x.copy(name = rename(x.name))
      case x: StreamZip => x.copy(names = x.names.map(rename))
      case x: StreamZipJoin => x.copy(curKey = rename(x.curKey), curVals = rename(x.curVals))
      case x: StreamZipJoinProducers =>
        x.copy(ctxName = rename(x.ctxName), curKey = rename(x.curKey), curVals = rename(x.curVals))
      case x: StreamLeftIntervalJoin =>
        x.copy(lname = rename(x.lname), rname = rename(x.rname))
      case x: StreamFor => x.copy(valueName = rename(x.valueName))
      case x: StreamFlatMap => x.copy(name = rename(x.name))
      case x: StreamFilter => x.copy(name = rename(x.name))
      case x: StreamTakeWhile => x.copy(elementName = rename(x.elementName))
      case x: StreamDropWhile => x.copy(elementName = rename(x.elementName))
      case x: StreamFold => x.copy(accumName = rename(x.accumName), valueName = rename(x.valueName))
      case x: StreamFold2 => x.copy(
          accum = x.accum.map { case (name, ir) => (rename(name), ir) },
          valueName = rename(x.valueName),
        )
      case x: StreamBufferedAggregate => x.copy(name = rename(x.name))
      case x: RunAggScan => x.copy(name = rename(x.name))
      case x: StreamAgg => x.copy(name = rename(x.name))
      case x: StreamScan => x.copy(accumName = rename(x.accumName), valueName = rename(x.valueName))
      case x: StreamAggScan => x.copy(name = rename(x.name))
      case x: StreamJoinRightDistinct => x.copy(l = rename(x.l), r = rename(x.r))
      case x: ArraySort => x.copy(left = rename(x.left), right = rename(x.right))
      case x: ArrayMaximalIndependentSet =>
        x.copy(tieBreaker = x.tieBreaker.map { case (l, r, f) => (rename(l), rename(r), f) })
      case x: AggArrayPerElement =>
        x.copy(indexName = rename(x.indexName), elementName = rename(x.elementName))
      case x: AggFold =>
        x.copy(accumName = rename(x.accumName), otherAccumName = rename(x.otherAccumName))
      case x: NDArrayMap => x.copy(valueName = rename(x.valueName))
      case x: NDArrayMap2 => x.copy(lName = rename(x.lName), rName = rename(x.rName))
      case x: CollectDistributedArray => x.copy(cname = rename(x.cname), gname = rename(x.gname))
      case x: AggExplode => x.copy(name = rename(x.name))
      case x: RelationalLet => x.copy(name = rename(x.name))
      case x: RelationalLetTable => x.copy(name = rename(x.name))
      case x: RelationalLetMatrixTable => x.copy(name = rename(x.name))
      case x: TableMapPartitions => x.copy(
          globalName = rename(x.globalName),
          partitionStreamName = rename(x.partitionStreamName),
        )
      case x: TableGen => x.copy(cname = rename(x.cname), gname = rename(x.gname))
      case x: BlockMatrixMap => x.copy(eltName = rename(x.eltName))
      case x: BlockMatrixMap2 =>
        x.copy(leftName = rename(x.leftName), rightName = rename(x.rightName))
      case x: RelationalLetBlockMatrix => x.copy(name = rename(x.name))

      case x => x
    }
  }

  def normalizeIROld(ir: BaseIR, env: BindingEnv[String], context: Array[String] = Array())
    : StackFrame[BaseIR] = {

//    @nowarn("cat=unused-locals&msg=default argument")
    def normalizeBaseIR(next: BaseIR, env: BindingEnv[String] = env): StackFrame[BaseIR] =
      call(normalizeIROld(next, env, context :+ ir.getClass().getName()))

    def normalize(next: IR, env: BindingEnv[String] = env): StackFrame[IR] =
      call(
        normalizeIROld(next, env, context :+ ir.getClass().getName()).asInstanceOf[StackFrame[IR]]
      )

    ir match {
      case Ref(name, typ) =>
        val newName = env.eval.lookupOption(name) match {
          case Some(n) => n
          case None =>
            if (!allowFreeVariables)
              throw new RuntimeException(
                s"found free variable in normalize: $name, ${context.reverse.mkString(", ")}; ${env.pretty(x => x)}"
              )
            else
              name
        }
        done(Ref(newName, typ))
      case RelationalRef(name, typ) =>
        val newName = env.relational.lookupOption(name).getOrElse(
          if (!allowFreeVariables) throw new RuntimeException(
            s"found free variable in normalize: $name, ${context.reverse.mkString(", ")}; ${env.pretty(x => x)}"
          )
          else name
        )
        done(RelationalRef(newName, typ))
      case Recur(name, args, typ) =>
        val newName = env.eval.lookupOption(name) match {
          case Some(n) => n
          case None =>
            if (!allowFreeVariables)
              throw new RuntimeException(
                s"found free loop variable in normalize: $name, ${context.reverse.mkString(", ")}; ${env.pretty(x => x)}"
              )
            else
              name
        }
        for {
          newArgs <- args.mapRecur(v => normalize(v))
        } yield Recur(newName, newArgs, typ)
      case Block(bindings, body) =>
        val newBindings: Array[Binding] = Array.ofDim(bindings.length)

        for {
          (env, _) <- bindings.foldLeft(done((env, 0))) {
            case (get, Binding(name, value, scope)) =>
              for {
                (env, idx) <- get
                newValue <- normalize(value, env.promoteScope(scope))
              } yield {
                val newName = gen()
                newBindings(idx) = Binding(newName, newValue, scope)
                (env.bindInScope(name, newName, scope), idx + 1)
              }
          }
          newBody <- normalize(body, env)
        } yield Block(newBindings, newBody)

      case TailLoop(name, args, resultType, body) =>
        val newFName = gen()
        val newNames = Array.tabulate(args.length)(i => gen())
        val (names, values) = args.unzip
        for {
          newValues <- values.mapRecur(v => normalize(v))
          newBody <- normalize(
            body,
            env.copy(eval = env.eval.bind(names.zip(newNames) :+ name -> newFName: _*)),
          )
        } yield TailLoop(newFName, newNames.zip(newValues), resultType, newBody)
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
        } yield StreamZipJoinProducers(newCtxs, newCtxName, newMakeProducer, key, newCurKey,
          newCurVals, newJoinF)
      case StreamLeftIntervalJoin(left, right, lKeyNames, rIntrvlName, lEltName, rEltName, body) =>
        val newLName = gen()
        val newRName = gen()
        for {
          newL <- normalize(left)
          newR <- normalize(right)
          newB <- normalize(body, env.bindEval(lEltName -> newLName, rEltName -> newRName))
        } yield StreamLeftIntervalJoin(newL, newR, lKeyNames, rIntrvlName, newLName, newRName, newB)
      case StreamFor(a, valueName, body) =>
        val newValueName = gen()
        for {
          newA <- normalize(a)
          newBody <- normalize(body, env.bindEval(valueName, newValueName))
        } yield StreamFor(newA, newValueName, newBody)
      case StreamFlatMap(a, name, body) =>
        val newName = gen()
        for {
          newA <- normalize(a)
          newBody <- normalize(body, env.bindEval(name, newName))
        } yield StreamFlatMap(newA, newName, newBody)
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
      case StreamFold(a, zero, accumName, valueName, body) =>
        val newAccumName = gen()
        val newValueName = gen()
        for {
          newA <- normalize(a)
          newZero <- normalize(zero)
          newBody <-
            normalize(body, env.bindEval(accumName -> newAccumName, valueName -> newValueName))
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
      case StreamBufferedAggregate(child, initAggs, newKey, seqOps, name, aggSigs, bufferSize) =>
        val newName = gen()
        for {
          newChild <- normalize(child)
          newEnv = env.bindEval(name -> newName)
          newInitAggs <- normalize(initAggs, newEnv)
          newNewKey <- normalize(newKey, newEnv)
          newSeqOps <- normalize(seqOps, newEnv)
        } yield StreamBufferedAggregate(newChild, newInitAggs, newNewKey, newSeqOps, newName,
          aggSigs, bufferSize)
      case RunAggScan(a, name, init, seq, result, sig) =>
        val newName = gen()
        val e2 = env.bindEval(name, newName)
        for {
          newA <- normalize(a)
          newInit <- normalize(init, env)
          newSeq <- normalize(seq, e2)
          newResult <- normalize(result, e2)
        } yield RunAggScan(newA, newName, newInit, newSeq, newResult, sig)
      case StreamAgg(a, name, body) =>
        // FIXME: Uncomment when bindings are threaded through test suites
        // assert(env.agg.isEmpty)
        val newName = gen()
        for {
          newA <- normalize(a)
          newBody <- normalize(body, env.copy(agg = Some(env.eval.bind(name, newName))))
        } yield StreamAgg(newA, newName, newBody)
      case StreamScan(a, zero, accumName, valueName, body) =>
        val newAccumName = gen()
        val newValueName = gen()
        for {
          newA <- normalize(a)
          newZero <- normalize(zero)
          newBody <-
            normalize(body, env.bindEval(accumName -> newAccumName, valueName -> newValueName))
        } yield StreamScan(newA, newZero, newAccumName, newValueName, newBody)
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
        } yield StreamJoinRightDistinct(newLeft, newRight, lKey, rKey, newL, newR, newJoinF,
          joinType)
      case ArraySort(a, left, right, lessThan) =>
        val newLeft = gen()
        val newRight = gen()
        for {
          newA <- normalize(a)
          newLessThan <- normalize(lessThan, env.bindEval(left -> newLeft, right -> newRight))
        } yield ArraySort(newA, newLeft, newRight, newLessThan)
      case ArrayMaximalIndependentSet(edges, Some((l, r, tieBreaker))) =>
        val newLeft = gen()
        val newRight = gen()
        for {
          newEdges <- normalize(edges)
          newTieBreaker <- normalize(tieBreaker, env.bindEval(l -> newLeft, r -> newRight))
        } yield ArrayMaximalIndependentSet(newEdges, Some((newLeft, newRight, newTieBreaker)))
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
        } yield AggArrayPerElement(newA, newElementName, newIndexName, newAggBody, newKnownLength,
          isScan)
      case AggFold(zero, seqOp, combOp, accumName, otherAccumName, isScan) =>
        val newAccumName = gen()
        val newOtherAccumName = gen()
        for {
          newZero <- normalize(zero)
          newSeqOp <- normalize(
            seqOp,
            (if (isScan) env.promoteScan else env.promoteAgg).bindEval(accumName -> newAccumName),
          )
          newCombOp <- normalize(
            combOp,
            env.bindEval(accumName -> newAccumName, otherAccumName -> newOtherAccumName),
          )
        } yield AggFold(newZero, newSeqOp, newCombOp, newAccumName, newOtherAccumName, isScan)
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
      case CollectDistributedArray(ctxs, globals, cname, gname, body, dynamicID, staticID, tsd) =>
        val newC = gen()
        val newG = gen()
        for {
          newCtxs <- normalize(ctxs)
          newGlobals <- normalize(globals)
          newBody <- normalize(body, BindingEnv.eval(cname -> newC, gname -> newG))
          newDynamicID <- normalize(dynamicID)
        } yield CollectDistributedArray(newCtxs, newGlobals, newC, newG, newBody, newDynamicID,
          staticID, tsd)
      case AggExplode(array, name, aggBody, isScan) =>
        val newName = gen()
        for {
          newArray <- normalize(array, if (isScan) env.promoteScan else env.promoteAgg)
          newAggBody <- normalize(
            aggBody,
            if (isScan) env.bindScan(name -> newName) else env.bindAgg(name -> newName),
          )
        } yield AggExplode(newArray, newName, newAggBody, isScan)
      case RelationalLet(name, value, body) =>
        val newName = gen()
        for {
          newValue <- normalize(value, env)
          newBody <- normalize(body, env.noAgg.noScan.bindRelational(name, newName))
        } yield RelationalLet(newName, newValue, newBody)
      case RelationalLetTable(name, value, body) =>
        val newName = gen()
        for {
          newValue <- normalize(value, env)
          newBody <- normalizeBaseIR(body, env.noAgg.noScan.bindRelational(name, newName))
        } yield RelationalLetTable(newName, newValue, newBody.asInstanceOf[TableIR])
      case RelationalLetMatrixTable(name, value, body) =>
        val newName = gen()
        for {
          newValue <- normalize(value, env)
          newBody <- normalizeBaseIR(body, env.noAgg.noScan.bindRelational(name, newName))
        } yield RelationalLetMatrixTable(newName, newValue, newBody.asInstanceOf[MatrixIR])
      case TableMapPartitions(child, globalName, streamName, body, requestedKey, allowedOverlap) =>
        val newGlobalName = gen()
        val newStreamName = gen()
        for {
          newChild <- normalizeBaseIR(child)
          newBody <- normalize(
            body,
            env.onlyRelational().bindEval(globalName -> newGlobalName, streamName -> newStreamName),
          )
        } yield TableMapPartitions(
          newChild.asInstanceOf[TableIR],
          newGlobalName,
          newStreamName,
          newBody,
          requestedKey,
          allowedOverlap,
        )
      case TableGen(contexts, globals, cname, gname, body, partitioner, errorId) =>
        val newCname = gen()
        val newGname = gen()
        for {
          newContexts <- normalize(contexts)
          newGlobals <- normalize(globals)
          newBody <-
            normalize(body, env.onlyRelational().bindEval(cname -> newCname, gname -> newGname))
        } yield TableGen(newContexts, newGlobals, newCname, newGname, newBody, partitioner, errorId)
      case BlockMatrixMap(child, eltName, f, needsDense) =>
        val newEltName = gen()
        for {
          newChild <- normalizeBaseIR(child)
          newF <- normalize(f, env.bindEval(eltName -> newEltName))
        } yield BlockMatrixMap(newChild.asInstanceOf[BlockMatrixIR], newEltName, newF, needsDense)
      case BlockMatrixMap2(left, right, leftName, rightName, f, sparsityStrategy) =>
        val newLeftName = gen()
        val newRightName = gen()
        for {
          newLeft <- normalizeBaseIR(left)
          newRight <- normalizeBaseIR(right)
          newF <- normalize(f, env.bindEval(leftName -> newLeftName, rightName -> newRightName))
        } yield BlockMatrixMap2(
          newLeft.asInstanceOf[BlockMatrixIR],
          newRight.asInstanceOf[BlockMatrixIR],
          newLeftName,
          newRightName,
          newF,
          sparsityStrategy,
        )
      case RelationalLetBlockMatrix(name, value, body) =>
        val newName = gen()
        for {
          newValue <- normalize(value, env)
          newBody <- normalizeBaseIR(body, env.noAgg.noScan.bindRelational(name, newName))
        } yield RelationalLetBlockMatrix(newName, newValue, newBody.asInstanceOf[BlockMatrixIR])
      case x =>
        x.mapChildrenWithIndexStackSafe { (child, i) =>
          normalizeBaseIR(
            child,
            env.extend(Bindings.get(x, i).map((name, _) => name)),
          )
        }
    }
  }
}
