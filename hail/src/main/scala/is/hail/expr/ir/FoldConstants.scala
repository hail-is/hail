package is.hail.expr.ir

import is.hail.expr.ir.Bindings.empty
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
    def recur(ir: BaseIR): Unit = findConstantHelper(ir, memo, usesAndDefs)

    def basicBindRecur(arg: IR, body: IR): Unit = {
      recur(arg)
      if (memo.contains(arg)) {
        usesAndDefs.uses(ir).foreach(ref => memo.bind(ref, ()))
      }
      recur(body)
    }
    def checkNameBind(name: String): Unit = {
      val refs = usesAndDefs.uses(ir).filter(ref => ref.t.name == name)
      refs.foreach(ref => memo.bind(ref,()))
    }
    def basicTwoRefIRBindRecur(firstIR: IR, secondIR: IR, firstName: String, secondName: String, body: IR): Unit = {
      recur(firstIR)
      recur(secondIR)
      if (memo.contains(firstIR)) {
        checkNameBind(firstName)
      }
      if (memo.contains(secondIR)) {
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
        case Let(name, value, body) => basicBindRecur(value, body)
        case TailLoop(name, args, body) => ???
        case StreamMap(a, name, body) => basicBindRecur(a, body)
        case StreamZip(as, names, body, _) => ???
        case StreamZipJoin(as, key, curKey, curVals, _) => ???
        case StreamFor(a, name, body) => basicBindRecur(a, body)
        case StreamFlatMap(a, name, body) => basicBindRecur(a, body)
        case StreamFilter(a, name, body) => basicBindRecur(a, body)
        case StreamFold(a, zero, accumName, valueName, body) =>
          basicTwoRefIRBindRecur(a, zero, valueName, accumName, body)
        case StreamFold2(a, accum, valueName, seq, result) => ???
        case RunAggScan(a, name, _, _, _, _) => ???
        case StreamScan(a, zero, valueName, accumName, body) => //Same as fold
          basicTwoRefIRBindRecur(a, zero, accumName, valueName, body)
        case StreamAggScan(a, name, _) => ???
        case StreamJoinRightDistinct(ll, rr, _, _, l, r, _, _) => ???
        case ArraySort(a, left, right, body) => ???
        case AggArrayPerElement(a, _, indexName, _, _, _) => ???
        case NDArrayMap(nd, name, body) => basicBindRecur(nd, body)
        case NDArrayMap2(l, r, lName, rName, body) =>
          basicTwoRefIRBindRecur(l, r, lName, rName, body)
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
    baseIR.isInstanceOf[ApplySeeded] || baseIR.isInstanceOf[UUID4] || baseIR.isInstanceOf[In]
  }
}
