package is.hail.expr.ir.lowering

import is.hail.expr.ir._
import is.hail.expr.ir.functions.GetElement
import is.hail.expr.types.virtual._
import is.hail.utils.FastIndexedSeq

case class BlockMatrixStage(
  ctxRef: Ref,
  blockContexts: Map[(Int, Int), IR],
  globalVals: Array[(String, IR)], //needed for relational lets
  body: IR) {
  def ctxName: String = ctxRef.name
  def ctxType: Type = ctxRef.typ
  def toIR(bodyTransform: IR => IR, ordering: Option[Array[(Int, Int)]]): IR = {
    val ctxs = MakeArray(
      ordering.map[Array[IR]](idxs => idxs.map(blockContexts(_)))
        .getOrElse[Array[IR]](blockContexts.values.toArray),
      TArray(ctxRef.typ))

    val blockResult = bodyTransform(body)
    val bcFields = globalVals.filter { case (f, _) => Mentions(blockResult, f) }
    val bcVals = MakeStruct(bcFields.map { case (f, v) => f -> Ref(f, v.typ) })
    val bcRef = Ref(genUID(), bcVals.typ)
    val wrappedBody = bcFields.foldLeft(blockResult) { case (accum, (f, _)) =>
      Let(f, GetField(bcRef, f), accum)
    }
    val collect = CollectDistributedArray(ctxs, bcVals, ctxRef.name, bcRef.name, wrappedBody)
    globalVals.foldRight[IR](collect) { case ((f, v), accum) => Let(f, v, accum) }
  }
}

object LowerBlockMatrixIR {

  def unimplemented[T](node: BaseIR): T =
    throw new LowererUnsupportedOperation(s"unimplemented: \n${ Pretty(node) }")

  def lower(node: IR): IR = node match {
//    case BlockMatrixCollect(child: BlockMatrixIR) =>
    case BlockMatrixToValueApply(child, GetElement(index)) => unimplemented(node)
    case BlockMatrixWrite(child, writer) => unimplemented(node)
    case BlockMatrixMultiWrite(blockMatrices, writer) => unimplemented(node)
    case node if node.children.exists( _.isInstanceOf[BlockMatrixIR] ) =>
      throw new LowererUnsupportedOperation(s"IR nodes with BlockMatrixIR children need explicit rules: \n${ Pretty(node) }")

    case node =>
      throw new LowererUnsupportedOperation(s"Value IRs with no BlockMatrixIR children must be lowered through LowerIR: \n${ Pretty(node) }")
  }

  def lower(bmir: BlockMatrixIR): BlockMatrixStage = bmir match {
    case BlockMatrixRead(reader) => unimplemented(bmir)
    case ValueToBlockMatrix(child, shape, blockSize) => unimplemented(bmir)
    case x: BlockMatrixLiteral => unimplemented(bmir)
    case BlockMatrixMap(child, eltName, f) => unimplemented(bmir)
    case BlockMatrixMap2(left, right, lname, rname, f) => unimplemented(bmir)
    case BlockMatrixBroadcast(child, inIndexExpr, shape, blockSize) => unimplemented(bmir)
    case BlockMatrixAgg(child, outIndexExpr) => unimplemented(bmir)
    case BlockMatrixFilter(child, keep) => unimplemented(bmir)
    case BlockMatrixSlice(child, slices) => unimplemented(bmir)
    case BlockMatrixDensify(child) => unimplemented(bmir)
    case BlockMatrixSparsify(child, sparsifier) => unimplemented(bmir)
    case RelationalLetBlockMatrix(name, value, body) => unimplemented(bmir)
    case x@BlockMatrixDot(leftIR, rightIR) =>
      val left = lower(leftIR)
      val right = lower(rightIR)
      val (_, n) = leftIR.typ.defaultBlockShape

      val newCtxType = TArray(TStruct(
        left.ctxName -> TArray(left.ctxType),
        right.ctxName -> TArray(right.ctxType)))

      // group blocks for multiply
      val newContexts = x.typ.allBlocks.map { case (i, j) =>
        (i -> j, MakeArray(Array.tabulate[Option[IR]](n) { k =>
          left.blockContexts.get(i -> k).flatMap { leftCtx =>
            right.blockContexts.get(k -> j).map { rightCtx =>
              MakeStruct(FastIndexedSeq(
                left.ctxName -> leftCtx,
                right.ctxName -> rightCtx))
            }
          }
        }.flatten[IR], newCtxType))
      }.toMap

      val wrapMultiply = { ctxElt: IR =>
        Let(left.ctxName, GetField(ctxElt, left.ctxName),
          Let(right.ctxName, GetField(ctxElt, right.ctxName),
            NDArrayMatMul(left.body, right.body)))
      }

      // computation for multiply
      val ctxRef = Ref(genUID(), newCtxType)
      val zero = wrapMultiply(ArrayRef(ctxRef, 0))
      val tail = invoke("[*:]", newCtxType, ctxRef, 1)
      val elt = Ref(genUID(), newCtxType.elementType)
      val accum = Ref(genUID(), zero.typ)
      val l = Ref(genUID(), TFloat64())
      val r = Ref(genUID(), TFloat64())
      val newBody = ArrayFold(tail, zero, accum.name, elt.name,
        NDArrayMap2(accum, wrapMultiply(elt), l.name, r.name, ApplyBinaryPrimOp(Add(), l, r)))

      BlockMatrixStage(
        ctxRef,
        newContexts,
        left.globalVals ++ right.globalVals,
        newBody)
  }
}
