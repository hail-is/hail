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
    visitIR(ir, constantSubTrees)
    val constants = ArrayBuffer[IR]()
    getConstantIRs(ir, constantSubTrees, constants)
    val constantsIS = constants.toIndexedSeq
    assert(constants.forall(x => x.typ.isRealizable))
    val constantsTrapTuple = MakeTuple.ordered(constantsIS.map(constIR => Trap(constIR)))
    val compiled = CompileAndEvaluate[Any](ctx, constantsTrapTuple, optimize = false)
    val rowCompiled = compiled.asInstanceOf[Row]
    val constDict = getIRConstantMapping(rowCompiled, constantsIS)
    replaceConstantTrees(ir, constDict)
  }

  def visitIR(baseIR: BaseIR, memo: Memo[Unit]): Option[Set[String]] = {
    baseIR match {
      case Ref(name, _) => Some(Set(name))
      case _ =>
        val childrenDeps = baseIR.children.map {
          child =>
            visitIR(child, memo)
        }
        val allConstantChildren = childrenDeps.forall(child => !child.isEmpty)
        if (!allConstantChildren) return None
        val constChildrenDeps = childrenDeps.map(child => child.get)
        val nodeDeps = baseIR.children.zip(constChildrenDeps).zipWithIndex.map {
          case ((child, childDep), index) =>
            childDep -- Bindings.apply(baseIR, index).map(ref => ref._1)
        }.foldLeft(Set[String]())((accum, elem) => elem ++ accum)
        baseIR match {
          case ir: IR =>
            if (nodeDeps.isEmpty && ir.typ.isRealizable && !neverConstantIRs(ir)) {
              memo.bind(ir, ())
              Some(nodeDeps)
            }
            else if (!neverConstantIRs(ir)) Some(nodeDeps)
            else None

          case _ => None
        }
    }
  }

  def neverConstantIRs(baseIR: BaseIR): Boolean = {
    baseIR match {
      case _: ApplySeeded |
          _: UUID4 |
          _: In |
          _: TailLoop |
          _: InitOp |
          _: SeqOp |
          _: CombOp |
          _: ResultOp |
          _: CombOpValue |
          _: AggStateValue |
          _: InitFromSerializedValue |
          _: SerializeAggs |
          _: DeserializeAggs |
          _: Die |
          _: AggLet |
          _: ApplyAggOp |
          _: ApplyScanOp |
          _: RelationalRef |
          _: RelationalLet |
          _: WriteValue |
          _: WritePartition |
          _: ReadPartition |
          _: CollectDistributedArray |
          _: WriteMetadata |
          _: ShuffleRead |
          _: ShuffleWith |
          _: ShuffleWrite |
          _: ShufflePartitionBounds |
          _: Begin => true
      case ir: IR if ir.typ == TVoid => true
      case _ => false
    }

  }

  def getConstantIRs(ir: BaseIR, constantSubTrees: Memo[Unit], constants : ArrayBuffer[IR]) : Unit  = {
    ir match {
      case ir: IR if constantSubTrees.contains(ir) && !IsConstant(ir) => constants += ir
      case _ => ir.children.foreach(child => getConstantIRs(child, constantSubTrees, constants))
    }
  }

  def getIRConstantMapping(constantsCompiled: Row, constantTrees: IndexedSeq[IR]): Memo[IR] = {
    val constDict = Memo.empty[IR]
    val constantCompiledSeq = (0 until constantsCompiled.length).map(idx => constantsCompiled(idx))
    constantTrees.zip(constantCompiledSeq).foreach { case (constantTree, constantCompiled) =>
      constantCompiled match {
        case Row(error, value) =>
          error match {
            case Row(msg: String, id: Int) =>
              constDict.bind(constantTree, new Die(Str(msg), constantTree.typ, id))
            case _ => constDict.bind(constantTree, Literal.coerce(constantTree.typ, value))
          }
      }
    }
    constDict
  }

  def replaceConstantTrees(baseIR: BaseIR, constDict: Memo[IR]): BaseIR = {
    if (constDict.contains(baseIR)) (constDict.get(baseIR).get)
    else baseIR.mapChildren{child => replaceConstantTrees(child, constDict)}
  }
}