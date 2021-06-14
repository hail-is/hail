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
    val constantSubTrees = Memo.empty[Unit]
    val usesAndDefs = ComputeUsesAndDefs(ir)
    findConstantSubTreesHelper(ir, constantSubTrees, usesAndDefs)
    val parents = ParentPointers(ir)
    removeUnrealizableConstants(ir, constantSubTrees, parents, usesAndDefs)
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
           _: Trap |
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
    findConstantSubTreesHelper(baseIR, constantSubTrees, usesAndDefs)
    constantSubTrees
  }
  def findConstantSubTreesHelper(ir: BaseIR, memo: Memo[Unit], usesAndDefs: UsesAndDefs): Unit = {
    def recur(ir: BaseIR): Unit = findConstantSubTreesHelper(ir, memo, usesAndDefs)
    def bindRefRecur(args: IndexedSeq[IR], body: IR, names : Option[IndexedSeq[String]] = None): Unit = {
      args.foreach( arg => recur(arg))
      if (names.isEmpty) bindRefs()
      else {
        val refNames = names.get
        stringIRSeqBind(refNames.zip(args))
      }
      recur(body)
    }
    def bindRefs(name: Option[String] = None): Unit = {

      val refs = if (!name.isEmpty)  usesAndDefs.uses(ir).filter(ref => ref.t.name == name.get)
                 else usesAndDefs.uses(ir)
      refs.foreach(ref => memo.bind(ref, ()))
    }
    def stringIRSeqBind(seq: IndexedSeq[(String, IR)]): Unit = {
     seq.foreach { case (name, streamIr) => if (memo.contains(streamIr)) bindRefs(Some(name))}
   }

     if (IsConstant(ir)) {
       memo.bind(ir, ())
     }
    else if (ir.isInstanceOf[Ref]) {}
    else {
      ir match {
        case Let(name, value, body) => bindRefRecur(IndexedSeq(value), body)
        case StreamMap(a, name, body) => bindRefRecur(IndexedSeq(a), body)
        case StreamZip(as, names, body, _) => bindRefRecur(as, body, Some(names))
        case StreamZipJoin(as, key, curKey, curVals, body) => {
          as.foreach(seq => recur(seq))
          val allConstant = as.forall(streamIR => memo.contains(streamIR))
          if (allConstant) bindRefs()
          recur(body)
        }
        case StreamFlatMap(a, name, body) => bindRefRecur(IndexedSeq(a), body)
        case StreamFilter(a, name, body) => bindRefRecur(IndexedSeq(a), body)
        case StreamFold(a, zero, accumName, valueName, body) =>
          bindRefRecur(IndexedSeq(a, zero), body, Some(IndexedSeq(valueName, accumName)))
        case StreamFold2(a, accum, valueName, seq, result) => {
          recur(a)
          accum.map(item => item._2).foreach(streamIr => recur(streamIr))
          if(memo.contains(a)) bindRefs(Some(valueName))
          stringIRSeqBind(accum)
          seq.foreach(seqIR => recur(seqIR))
          recur(result)   //prob not right
        }
        case RunAggScan(a, name, _, _, _, _) => ???
        case StreamScan(a, zero, valueName, accumName, body) =>
          bindRefRecur(IndexedSeq(a, zero), body, Some(IndexedSeq(valueName, accumName)))
        case StreamAggScan(a, name, body) => bindRefRecur(IndexedSeq(a), body)
        case StreamJoinRightDistinct(ll, rr, lkeys, rkeys, l, r, body, _) =>
          bindRefRecur(IndexedSeq(ll, rr), body)
        case ArraySort(a, left, right, body) => bindRefRecur(IndexedSeq(a), body)
        case AggArrayPerElement(a, element, indexName, aggBody, knownLength, _) => ???
        case NDArrayMap(nd, name, body) => bindRefRecur(IndexedSeq(nd), body)
        case NDArrayMap2(l, r, lName, rName, body) =>
          bindRefRecur(IndexedSeq(l, r), body, Some(IndexedSeq(lName, rName)))
        case _ =>
          ir.children.foreach(child => {
            findConstantSubTreesHelper(child, memo, usesAndDefs)
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
      baseIR.isInstanceOf[TailLoop] || baseIR.typ == TVoid
  }

//  def removeUnrealizableConstants(ir: BaseIR, constantSubTrees: Memo[Unit], usesAndDefs: UsesAndDefs) : Unit = {
//    val parents = ParentPointers(ir)
//    removeUnrealizableConstants(ir, constantSubTrees, parents, usesAndDefs)
//
//  }

  def removeUnrealizableConstants(ir : BaseIR, constantSubTrees: Memo[Unit], parents: Memo[BaseIR],
                                  usesAndDefs: UsesAndDefs): Unit = {
    if (!constantSubTrees.contains(ir)) {
      ir.children.foreach(child => removeUnrealizableConstants(child, constantSubTrees,
        ParentPointers(child), usesAndDefs))
    }
    else {
      ir match {
        case ir : IR =>
          if (!ir.typ.isRealizable) {
            constantSubTrees.delete(ir)
            usesAndDefs.uses(ir).foreach(ref => {
              var current: BaseIR = ref.t
              while (constantSubTrees.contains(current)) {
                constantSubTrees.delete(ref)
                current = parents.get(current).getOrElse(current)
              }
            })
          }
      }
    }
  }
//  def getConstantIRsAndRefs(ir: BaseIR, constantSubTrees: Memo[Unit]): (IndexedSeq[IR], IndexedSeq[(String, IR)]) = {
//    val constants = ArrayBuffer[IR]()
//    val refs = ArrayBuffer[(String,IR)]()
//    val results = getConstantIRsAndRefsHelper(ir, constantSubTrees, constants, refs)
//    (constants.toIndexedSeq , refs.toIndexedSeq)
//
//  }

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
