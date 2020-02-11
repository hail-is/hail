package is.hail.expr.ir.lowering

import is.hail.expr.ir._
import is.hail.expr.ir.functions.GetElement
import is.hail.expr.types.virtual._
import is.hail.utils.FastIndexedSeq

case class BlockMatrixStage(
  ctxRef: Ref,
  blockContexts: Map[(Int, Int), IR],
  broadcastVals: Map[String, IR], //needed for relational lets
  body: IR) {
  def ctxName: String = ctxRef.name
  def ctxType: Type = ctxRef.typ
  def toIR(bodyTransform: IR => IR, ordering: Option[Array[(Int, Int)]]): IR = {
    val bc = MakeStruct(broadcastVals.toArray)
    val bcRef = Ref(genUID(), bc.typ)
    val ctxs = ordering.map { idxs =>
      idxs.map { case (i, j) => blockContexts(i -> j) }
    }.getOrElse { blockContexts.values.toArray }
    val bcFields = coerce[TStruct](bc.typ).fieldNames
    val wrappedBody = bcFields.foldLeft(bodyTransform(body)) { (accum, f) =>
      Let(f, GetField(bcRef, f), accum)
    }
    CollectDistributedArray(MakeArray(ctxs, TArray(ctxRef.typ)), bc, ctxRef.name, bcRef.name, wrappedBody)
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
    case BlockMatrixSparsify(child, value, sparsifier) => unimplemented(bmir)
    case RelationalLetBlockMatrix(name, value, body) => unimplemented(bmir)
    case x@BlockMatrixDot(leftIR, rightIR) =>
      val left = lower(leftIR)
      val right = lower(rightIR)
      val (_, n) = leftIR.typ.defaultBlockShape

      val newCtxType = TArray(TStruct(
        left.ctxName -> TArray(left.ctxType),
        right.ctxName -> TArray(right.ctxType)))

      // group blocks for multiply
      // the contexts that we're parallelizing, *in general*, are going to involve little-to-no computation so duplicating across nodes seems fine for now.
      val newContexts = x.typ.allBlocks.map { case (i, j) =>
        (i -> j, MakeArray(Array.tabulate[Option[IR]](n) { k =>
          left.blockContexts.get(i -> k).flatMap { leftCtx =>
            right.blockContexts.get(k -> j).map { rightCtx =>
              MakeStruct(FastIndexedSeq(
                left.ctxName -> leftCtx,
                right.ctxName -> rightCtx))
            }
          }
        }.flatten, newCtxType))
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
        left.broadcastVals ++ right.broadcastVals,
        newBody)
  }
}
