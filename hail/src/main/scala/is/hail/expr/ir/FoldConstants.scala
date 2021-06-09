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
    if (IsConstant(ir)) {
      memo.bind(ir, ())
    }
    else {
      ir match {
        case Ref(name, typ) => {}
        case Let(name, value, body) =>
          recur(value)
          if (memo.contains(value)) {
            usesAndDefs.uses(ir).foreach(ref => memo.bind(ref, ()))
          }
          recur(body)
        case TailLoop(name, args, body) => ???
        case StreamMap(a, name, body) =>
          recur(a)
          if (memo.contains(a)) {
            usesAndDefs.uses(ir).foreach(ref => memo.bind(ref, ()))
          }
          recur(body)
        case StreamZip(as, names, _, _) => ???
        case StreamZipJoin(as, key, curKey, curVals, _) => ???
        case StreamFor(a, name, body) =>
          recur(a)
          if (memo.contains(a)) {
            usesAndDefs.uses(ir).foreach(ref => memo.bind(ref, ()))
          }
          recur(body)
        case StreamFlatMap(a, name, _) => ???
        case StreamFilter(a, name, body) =>
          recur(a)
          if (memo.contains(a)) {
            usesAndDefs.uses(ir).foreach(ref => memo.bind(ref, ()))
          }
          recur(body)
        case StreamFold(a, zero, accumName, valueName, _) => ???
        case StreamFold2(a, accum, valueName, seq, result) => ???
        case RunAggScan(a, name, _, _, _, _) => ???
        case StreamScan(a, zero, accumName, valueName, _) => ???
        case StreamAggScan(a, name, _) => ???
        case StreamJoinRightDistinct(ll, rr, _, _, l, r, _, _) => ???
        case ArraySort(a, left, right, _) => ???
        case AggArrayPerElement(a, _, indexName, _, _, _) => ???
        case NDArrayMap(nd, name, body) =>
          recur(nd)
          if (memo.contains(nd)) {
            usesAndDefs.uses(ir).foreach(ref => memo.bind(ref, ()))
          }
          recur(body)
        case NDArrayMap2(l, r, lName, rName, _) => ???
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
    // TODO: Fill this in with bad IRS
    false
  }
}
