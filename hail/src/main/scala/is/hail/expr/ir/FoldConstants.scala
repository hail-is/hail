package is.hail.expr.ir

import is.hail.expr.ir.Bindings.empty
import is.hail.expr.ir.analyses.ParentPointers
import is.hail.types.virtual.{TArray, TNDArray, TStream, TStruct, TTuple}
import is.hail.utils.{FastIndexedSeq, HailException}

object FoldConstants {
  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ExecuteContext.scopedNewRegion(ctx) { ctx =>
      foldConstants(ctx, ir)
    }

  private def foldConstants(ctx: ExecuteContext, ir: BaseIR): BaseIR = {
    RewriteBottomUp(ir, {
      case _: Ref |
           _: In |
           _: RelationalRef |
           _: RelationalLet |
           _: ApplySeeded |
           _: UUID4 |
           _: ApplyAggOp |
           _: ApplyScanOp |
           _: AggLet |
           _: Begin |
           _: MakeNDArray |
           _: NDArrayShape |
           _: NDArrayReshape |
           _: NDArrayConcat |
           _: NDArraySlice |
           _: NDArrayFilter |
           _: NDArrayMap |
           _: NDArrayMap2 |
           _: NDArrayReindex |
           _: NDArrayAgg |
           _: NDArrayWrite |
           _: NDArrayMatMul |
           _: Die => None
      case ir: IR if ir.typ.isInstanceOf[TStream] => None
      case ir: IR if !IsConstant(ir) &&
        Interpretable(ir) &&
        ir.children.forall {
          case c: IR => IsConstant(c)
          case _ => false
        } =>
        try {
          Some(Literal.coerce(ir.typ, Interpret.alreadyLowered(ctx, ir)))
        } catch {
          case _: HailException => None
        }
      case _ => None
    })

  }
  def findConstantSubTrees(baseIR: BaseIR): Memo[Unit] = {
    val constantSubTrees = Memo.empty[Unit]
    val usesAndDefs = ComputeUsesAndDefs(baseIR)
    findConstantHelper(baseIR, constantSubTrees, usesAndDefs)
    constantSubTrees
  }
  def findConstantHelper(ir: BaseIR, memo: Memo[Unit], usesAndDefs: UsesAndDefs): Unit = {
    def recur(ir: BaseIR): Unit = findConstantHelper(ir, memo, usesAndDefs) //Does this or a lot of this
                                                                            //even need arguments if base ir

    def bindRefRecur(arg: IR, body: IR): Unit = {
      recur(arg)
      if (memo.contains(arg)) {
        bindAllRefs(arg)
      }
      recur(body)
    }
    def bindRefs(arg: IR, name: String): Unit = {
      //take place of below two methods
      //confused if the arg IR is always current IR or passed IR
      //Pretty sure its either both or always the current one we
      //are on
      val refs = if (name == null)  usesAndDefs.uses(ir).filter(ref => ref.t.name == name)
                 else usesAndDefs.uses(ir)
      refs.foreach(ref => memo.bind(ref, ()))
    }
    def bindAllRefs(arg: IR): Unit = usesAndDefs.uses(ir).foreach(ref => memo.bind(ref, ()))
    def checkNameBind(name: String): Unit = {
      //Should this take an ir as as an argument
      val refs = usesAndDefs.uses(ir).filter(ref => ref.t.name == name)
      refs.foreach(ref => memo.bind(ref,()))
    }
    def stringIRSeqRecurBind(seq: IndexedSeq[(String, IR)]): Unit =
      seq.foreach { case (name, streamIr) => if (memo.contains(streamIr)) checkNameBind(name)}

    def twoRefIRBindRecur(firstIR: IR, secondIR: IR, firstName: String, secondName: String, body: IR): Unit = {
      recur(firstIR)
      recur(secondIR)
      if (memo.contains(firstIR)) {
        if (firstName == null) bindAllRefs(firstIR) //Ugly ;-;
        checkNameBind(firstName)
      }
      if (memo.contains(secondIR)) {
        if (secondName == null) bindAllRefs(secondIR)
        checkNameBind(secondName)
      }
      recur(body)
    }

    if (IsConstant(ir)) {
      memo.bind(ir, ())
    }
    else if (ir.isInstanceOf[Ref]) {}
    else {
      ir match {
        case Let(name, value, body) => bindRefRecur(value, body)
        case StreamMap(a, name, body) => bindRefRecur(a, body)
        case StreamZip(as, names, body, _) => {
          as.foreach(seq => recur(seq))
          stringIRSeqRecurBind(names.zip(as))
          recur(body)
        }
        case StreamZipJoin(as, key, curKey, curVals, body) => {
          as.foreach(seq => recur(seq))
          val allConstant = as.forall(streamIR => memo.contains(streamIR))
          if (allConstant) as.foreach(streamIr => bindAllRefs(streamIr))
          recur(body)
          // or is it if(allConstant) key.zip(as).foreach{ case (name, streamIr) => checkNameBind(name)

        }
        case StreamFor(a, name, body) => bindRefRecur(a, body)
        case StreamFlatMap(a, name, body) => bindRefRecur(a, body)
        case StreamFilter(a, name, body) => bindRefRecur(a, body)
        case StreamFold(a, zero, accumName, valueName, body) =>
          twoRefIRBindRecur(a, zero, valueName, accumName, body)
        case StreamFold2(a, accum, valueName, seq, result) => {
          recur(a)
          accum.map(item => item._2).foreach(streamIr => recur(streamIr))
          if(memo.contains(a)) checkNameBind(valueName)
          stringIRSeqRecurBind(accum)
          seq.foreach(seqIR => recur(seqIR))
          recur(result)   //prob not right
        }

        case RunAggScan(a, name, _, _, _, _) => ???
        case StreamScan(a, zero, valueName, accumName, body) =>
          twoRefIRBindRecur(a, zero, accumName, valueName, body)
        case StreamAggScan(a, name, body) => bindRefRecur(a, body)
        case StreamJoinRightDistinct(ll, rr, lkeys, rkeys, l, r, body, _) =>
          twoRefIRBindRecur(ll, rr, firstName = null, secondName =null, body)
        case ArraySort(a, left, right, body) => bindRefRecur(a, body)
        case AggArrayPerElement(a, _, indexName, _, _, _) => ???
        case NDArrayMap(nd, name, body) => bindRefRecur(nd, body)
        case NDArrayMap2(l, r, lName, rName, body) =>
          twoRefIRBindRecur(l, r, lName, rName, body)
        case _ =>
          ir.children.foreach(child => {
            findConstantHelper(child, memo, usesAndDefs)
          })
      }
      val isConstantSubtree = ir.children.forall(child => {
        memo.contains(child)
      }) && !badIRs(ir) && ir.isInstanceOf[IR]
      if (isConstantSubtree) {
        memo.bind(ir, ())
      }
    }
  }

  def badIRs(baseIR: BaseIR): Boolean = {
    baseIR.isInstanceOf[ApplySeeded] || baseIR.isInstanceOf[UUID4] || baseIR.isInstanceOf[In]||
      baseIR.isInstanceOf[TailLoop]
  }

  def fixupStreams(ir: BaseIR, constantSubtrees: Memo[Unit]): Unit = {
    val parents = ParentPointers(ir)
    fixupStreamsHelper(ir, constantSubtrees, parents)
  }

  def fixupStreamsHelper(ir: BaseIR, constantSubtrees: Memo[Unit], parents: Memo[BaseIR]): Unit = {

  }
}
