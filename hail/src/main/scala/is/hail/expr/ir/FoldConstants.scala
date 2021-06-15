package is.hail.expr.ir

import is.hail.expr.ir.Bindings.empty
import is.hail.expr.ir.analyses.ParentPointers
import is.hail.types.virtual.{TArray, TNDArray, TStream, TStruct, TTuple, TVoid}
import is.hail.utils.{FastIndexedSeq, HailException}
import org.apache.spark.sql.Row

import scala.collection.mutable.ArrayBuffer

object FoldConstants {
  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ExecuteContext.scopedNewRegion(ctx) { ctx =>
      foldConstants(ctx, ir)
    }

  def mainMethod(ctx: ExecuteContext, ir : BaseIR): Row = {
    println(Pretty(ir))
    val constantSubTrees = Memo.empty[Unit]
    val usesAndDefs = ComputeUsesAndDefs(ir)
    val constantRefs = Set[String]()
    visitIR(ir, constantRefs, constantSubTrees)
    val parents = ParentPointers(ir)
    val constants = ArrayBuffer[IR]()
    val bindings = ArrayBuffer[(String,IR)]()
    getConstantIRsAndRefs(ir, constantSubTrees, constants, bindings)
    val constantsIS = constants.toIndexedSeq
    val bindingsIS = bindings.toIndexedSeq
    val constantTuple = MakeTuple.ordered(constantsIS)
    val letWrapped = bindingsIS.foldRight[IR](constantTuple){ case ((name, binding), accum) => Let(name, binding, accum)}
    val compiled = CompileAndEvaluate[Any](ctx, letWrapped)
    return compiled.asInstanceOf[Row]
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

  def visitIR(baseIR: BaseIR, constantRefs: Set[String], memo: Memo[Unit]): Option[Set[String]] = {

    baseIR match {
      case Ref(name, _) => {
        if (constantRefs.contains(name)) return Some(Set())
        else return Some(Set(name))
      }
      case let@Let(name, value, body) => {
        val valueDeps = visitIR(value, constantRefs, memo)
        if (!valueDeps.isEmpty) {
          val bodyConstantRefs = if (valueDeps.get.isEmpty) {
            constantRefs + name
          }
          else constantRefs
          val bodyDeps = visitIR(body, bodyConstantRefs, memo)
          val nodeDeps = bodyDeps.map(bD => valueDeps.get ++ (bD - name))
          nodeDeps.map(nD => if (nD.isEmpty) memo.bind(let, ()))
          return nodeDeps
        }
        return valueDeps
      }
      case _ => {
        val childrenDeps = baseIR.children.map { child => visitIR(child, constantRefs, memo) }
        val allConstantChildren = childrenDeps.forall(child => !child.isEmpty)
        if (!allConstantChildren) return None
        val constChildrenDeps = childrenDeps.map(child => child.get)
        val nodeDeps = baseIR.children.zip(constChildrenDeps).zipWithIndex.map { case ((child, childDep), index) =>
          childDep -- Bindings.apply(baseIR, index).map(ref => ref._1)
        }.foldLeft(Set[String]())((accum, elem) => elem ++ accum)
        baseIR match {
          case ir: IR =>
            if (nodeDeps.isEmpty && ir.typ.isRealizable && !badIRs(ir)) {
              memo.bind(ir, ())
              return Some(nodeDeps)
            }
            else if (!badIRs(ir))
              return Some(nodeDeps)
            else return None

          case _ => None
        }
      }
    }
  }



  def badIRs(baseIR: BaseIR): Boolean = {
    baseIR.isInstanceOf[ApplySeeded] || baseIR.isInstanceOf[UUID4] || baseIR.isInstanceOf[In]||
      baseIR.isInstanceOf[TailLoop] || baseIR.typ == TVoid
  }

  def getConstantIRsAndRefs(ir: BaseIR, constantSubTrees: Memo[Unit], constants : ArrayBuffer[IR],
                                  refs : ArrayBuffer[(String,IR)]) : Unit  = {
    ir match {
      case ir: IR if constantSubTrees.contains(ir) => constants += ir

      case let@Let(name, value, body) => {
        if (constantSubTrees.contains(value)) {
          refs += ((name, value))
          constants += let
        }
        else getConstantIRsAndRefs(value, constantSubTrees, constants, refs)
        getConstantIRsAndRefs(body, constantSubTrees, constants, refs)
      }
      case _ => ir.children.foreach(child => getConstantIRsAndRefs(child, constantSubTrees, constants, refs))
    }
  }
}
