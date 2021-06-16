package is.hail.expr.ir

import is.hail.expr.ir.Bindings.empty
import is.hail.expr.ir.analyses.ParentPointers
import is.hail.types.virtual.{TArray, TNDArray, TStream, TStruct, TTuple, TVoid}
import is.hail.utils.{FastIndexedSeq, HailException}
import org.apache.spark.sql.Row
import is.hail.utils._

import scala.collection.mutable.ArrayBuffer

object FoldConstants {
  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ExecuteContext.scopedNewRegion(ctx) { ctx =>
      foldConstants(ctx, ir)
    }

  def foldConstants(ctx: ExecuteContext, ir : BaseIR): BaseIR = {
    println("fold constants start")
    println(Pretty(ir))
    val constantSubTrees = Memo.empty[Unit]

    val constantRefs = Set[String]()
    visitIR(ir, constantRefs, constantSubTrees)

    val constants = ArrayBuffer[IR]()
    val bindings = ArrayBuffer[(String,IR)]()
    getConstantIRsAndRefs(ir, constantSubTrees, constants, bindings)
    val constantsIS = constants.toIndexedSeq

    val bindingsIS = bindings.toIndexedSeq
    val constantTuple = MakeTuple.ordered(constantsIS)
    val letWrapped = bindingsIS.foldRight[IR](constantTuple){ case ((name, binding), accum) => Let(name, binding, accum)}
    println("Some line")
    println(Pretty(letWrapped))
    val compiled = CompileAndEvaluate[Any](ctx, letWrapped, optimize = false)
    val rowCompiled = compiled.asInstanceOf[Row]
    val constDict = getIRConstantMapping(rowCompiled, constantsIS)
    val productIR = replaceConstantTrees(ir, constDict)
    log.info("Fold constants end")
    log.info(Pretty(productIR))
    productIR
  }

  def visitIR(baseIR: BaseIR, constantRefs: Set[String], memo: Memo[Unit]): Option[Set[String]] = {

    baseIR match {
      case Ref(name, _) => {
        if (constantRefs.contains(name)) Some(Set())
        else Some(Set(name))
      }
      case let@Let(name, value, body) => {
        val valueDeps = visitIR(value, constantRefs, memo)
        val bodyConstantRefs = if (!valueDeps.isEmpty) {
          if (valueDeps.get.isEmpty) {
            constantRefs + name
          }
          else constantRefs
        }
        else constantRefs

        val bodyDeps = visitIR(body, bodyConstantRefs, memo)

        val nodeDeps = bodyDeps.flatMap(bD =>
          valueDeps.map(vD => vD ++ (bD - name)))
//          if (valueDeps.isEmpty)  bD
//          else valueDeps.get ++ (bD - name)
        nodeDeps.foreach(nD => if (nD.isEmpty) memo.bind(let, ()))
        nodeDeps
      }
      case _ => {
        val childrenDeps = baseIR.children.map {
          child =>
            visitIR(child, constantRefs, memo)
        }
        val allConstantChildren = childrenDeps.forall(child => !child.isEmpty)
        if (!allConstantChildren) return None
        val constChildrenDeps = childrenDeps.map(child => child.get)
        val nodeDeps = baseIR.children.zip(constChildrenDeps).zipWithIndex.map { case ((child, childDep), index) =>
          childDep -- Bindings.apply(baseIR, index).map(ref => ref._1)
        }.foldLeft(Set[String]())((accum, elem) => elem ++ accum)
        baseIR match {
          case ir: IR =>
            if (nodeDeps.isEmpty && ir.typ.isRealizable && !badIRs(ir) && !IsConstant(ir)) {
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
      baseIR.isInstanceOf[TailLoop] || baseIR.typ == TVoid || baseIR.isInstanceOf[InitOp] ||
      baseIR.isInstanceOf[SeqOp] || baseIR.isInstanceOf[CombOp] || baseIR.isInstanceOf[ResultOp] ||
      baseIR.isInstanceOf[CombOpValue] || baseIR.isInstanceOf[AggStateValue] ||
      baseIR.isInstanceOf[InitFromSerializedValue] || baseIR.isInstanceOf[SerializeAggs] ||
      baseIR.isInstanceOf[DeserializeAggs]
  }

  def getConstantIRsAndRefs(ir: BaseIR, constantSubTrees: Memo[Unit], constants : ArrayBuffer[IR],
                                  refs : ArrayBuffer[(String,IR)]) : Unit  = {
    ir match {
      case ir: IR if constantSubTrees.contains(ir) => constants += ir

      case let@Let(name, value, body) => {
        if (constantSubTrees.contains(value)) {
          refs += ((name, value))
          constants += value
        }
        else getConstantIRsAndRefs(value, constantSubTrees, constants, refs)
        getConstantIRsAndRefs(body, constantSubTrees, constants, refs)
      }
      case _ => ir.children.foreach(child => getConstantIRsAndRefs(child, constantSubTrees, constants, refs))
    }
  }
  def getIRConstantMapping(constantsCompiled: Row, constantTrees: IndexedSeq[IR]): Memo[IR] = {
    val constDict = Memo.empty[IR]
    val constantCompiledSeq = (0 until constantsCompiled.length).map(idx => constantsCompiled(idx))
    constantTrees.zip(constantCompiledSeq).foreach { case (constantTree, constantCompiled) =>
      constDict.bind(constantTree, Literal.coerce(constantTree.typ, constantCompiled))
    }
    constDict
  }


  def replaceConstantTrees(baseIR: BaseIR, constDict: Memo[IR]): BaseIR = {
    if (constDict.contains(baseIR)) (constDict.get(baseIR).get)
    else baseIR.mapChildren{child =>
      if (constDict.contains(child)) constDict.get(child).get
      else replaceConstantTrees(child, constDict)
    }
  }
//  def replaceConstantSubTrees(baseIR: BaseIR, constDict: Memo[IR]): BaseIR = {
//    val replaceConstantSubTreeHelper = (child : BaseIR, constDict: Memo[IR]) => {
//      if (constDict.contains(child)) return (constDict.get(baseIR).get)
//      else baseIR.children.foreach(child => child.mapChildren(replaceConstantSubTreeHelper))
//    }
//    if (constDict.contains(baseIR)) return (constDict.get(baseIR).get)
//    baseIR.mapChildren(replaceConstantSubTreeHelper)
//
//    }


}

