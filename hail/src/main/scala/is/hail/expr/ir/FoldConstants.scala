package is.hail.expr.ir

import is.hail.types.virtual.TVoid
import is.hail.utils._
import org.apache.spark.sql.Row

import scala.collection.mutable.ArrayBuffer

object FoldConstants {
  def apply(ctx: ExecuteContext, ir: BaseIR): BaseIR =
    ExecuteContext.scopedNewRegion(ctx) { ctx =>
      foldConstants(ctx, ir)
    }

  def foldConstants(ctx: ExecuteContext, ir : BaseIR): BaseIR = {
    val constantSubTrees = Memo.empty[Unit]
    val constantRefs = Set[String]()
    visitIR(ir, constantRefs, constantSubTrees)
    val constants = ArrayBuffer[IR]()
    val bindings = ArrayBuffer[(String,IR)]()
    getConstantIRsAndRefs(ir, constantSubTrees, constants, bindings)
    val constantsIS = constants.toIndexedSeq
    assert(constants.forall(x => x.typ.isRealizable))
    val bindingsIS = bindings.toIndexedSeq
    val constantTuple = MakeTuple.ordered(constantsIS)
    val letWrapped = bindingsIS.foldRight[IR](constantTuple){ case ((name, binding), accum) => Let(name, binding, accum)}
    val productIR = try {
      val compiled = CompileAndEvaluate[Any](ctx, letWrapped, optimize = false)
      val rowCompiled = compiled.asInstanceOf[Row]
      val constDict = getIRConstantMapping(rowCompiled, constantsIS)
      replaceConstantTrees(ir, constDict)
    }
    catch {
      case _: HailException | _: NumberFormatException => {
        log.info("Error raised during fold constants, aborting")
        ir
      }
    }
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
        nodeDeps.foreach(nD => if (nD.isEmpty && let.typ.isRealizable) memo.bind(let, ()))
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
            if (nodeDeps.isEmpty && ir.typ.isRealizable && !badIRs(ir)) {
              memo.bind(ir, ())
               Some(nodeDeps)
            }
            else if (!badIRs(ir)) Some(nodeDeps)
            else  None

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
      baseIR.isInstanceOf[DeserializeAggs] || baseIR.isInstanceOf[Die]|| baseIR.isInstanceOf[AggLet] ||
      baseIR.isInstanceOf[ApplyAggOp] || baseIR.isInstanceOf[ApplyScanOp] || baseIR.isInstanceOf[RelationalRef] ||
      baseIR.isInstanceOf[RelationalLet] || baseIR.isInstanceOf[WriteValue]  || baseIR.isInstanceOf[WritePartition] ||
      baseIR.isInstanceOf[WriteMetadata]

  }

  def getConstantIRsAndRefs(ir: BaseIR, constantSubTrees: Memo[Unit], constants : ArrayBuffer[IR],
                                  refs : ArrayBuffer[(String,IR)]) : Unit  = {
    ir match {
      case ir: IR if constantSubTrees.contains(ir) && !IsConstant(ir) => constants += ir

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
    else baseIR.mapChildren{child => replaceConstantTrees(child, constDict)
    }
  }
}

