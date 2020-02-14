package is.hail.expr.ir.lowering

import is.hail.expr.Nat
import is.hail.expr.ir._
import is.hail.expr.ir.functions.GetElement
import is.hail.expr.types.virtual._
import is.hail.utils.{FastIndexedSeq, FastSeq}

object BlockMatrixStage {
  def apply(blockContexts: Map[(Int, Int), IR], globalVals: Array[(String, IR)])(body: Ref => IR): BlockMatrixStage = {
    assert(blockContexts.nonEmpty)
    assert(blockContexts.values.reduce((c1, c2) => c1.typ.isOfType(c2.typ)))
    val ctxRef = Ref(genUID(), blockContexts.values.head.typ)
    BlockMatrixStage(ctxRef, blockContexts, globalVals, body(ctxRef))
  }

  def empty(eltType: Type): BlockMatrixStage = {
    BlockMatrixStage(
      Ref(genUID(), TInt32()),
      Map.empty,
      Array.empty,
      NA(TNDArray(eltType, Nat(2))))
  }

}

case class BlockMatrixStage(
  ctxRef: Ref,
  blockContexts: Map[(Int, Int), IR],
  globalVals: Array[(String, IR)], //needed for relational lets
  body: IR) {
  def ctxName: String = ctxRef.name
  def ctxType: Type = ctxRef.typ
  def toIR(bodyTransform: IR => IR, ordering: Option[Array[(Int, Int)]]): IR = {
    if (blockContexts.isEmpty)
      MakeArray(FastSeq(), TArray(bodyTransform(body).typ))
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
    case BlockMatrixCollect(child) =>
      val bm = lower(child)
      val nRowBlocks = child.typ.nRowBlocks
      val nColBlocks = child.typ.nColBlocks
      val r = Ref(genUID(), bm.body.typ)
      val c = Ref(genUID(), bm.body.typ)
      val blocks = Ref(genUID(), TArray(bm.body.typ))
      if (child.typ.isSparse) {
        val zeros = MakeNDArray(
          ArrayMap(ArrayRange(0, child.typ.blockSize ^ 2, 1), genUID(), F64(0.0)),
          MakeTuple.ordered(FastSeq(I64(child.typ.blockSize), I64(child.typ.blockSize))),
          True())
        val order = bm.blockContexts.keys.toArray
        val map = order.zipWithIndex.toMap
        val cda = bm.toIR(b => b, Some(order))
        Let(blocks.name, cda,
          NDArrayConcat(
            MakeArray(Array.tabulate(nRowBlocks) { i =>
              NDArrayConcat(
                MakeArray(Array.tabulate(nColBlocks) { j =>
                  map.get(i -> j)
                    .map[IR](idx => ArrayRef(blocks, idx))
                    .getOrElse(zeros)
                }, cda.typ), 1)
            }, cda.typ), 0))

      } else {
        val rowMajor = Array.range(0, nRowBlocks)
          .flatMap(i => Array.tabulate(nColBlocks)(j => i -> j))
        val cda = bm.toIR(b => b, Some(rowMajor))
        Let(blocks.name, cda,
          NDArrayConcat(
            ArrayMap(
              ArrayRange(0, nRowBlocks, 1),
              r.name,
              NDArrayConcat(
                ArrayMap(ArrayRange(0, nColBlocks, 1), c.name,
                  ArrayRef(blocks, (r * nColBlocks) + c)
                ), 1)), 0))
      }

    case BlockMatrixToValueApply(child, GetElement(index)) => unimplemented(node)
    case BlockMatrixWrite(child, writer) => unimplemented(node)
    case BlockMatrixMultiWrite(blockMatrices, writer) => unimplemented(node)
    case node if node.children.exists( _.isInstanceOf[BlockMatrixIR] ) =>
      throw new LowererUnsupportedOperation(s"IR nodes with BlockMatrixIR children need explicit rules: \n${ Pretty(node) }")

    case node =>
      throw new LowererUnsupportedOperation(s"Value IRs with no BlockMatrixIR children must be lowered through LowerIR: \n${ Pretty(node) }")
  }

  def lower(bmir: BlockMatrixIR): BlockMatrixStage = {
    if (bmir.typ.nDefinedBlocks == 0)
      BlockMatrixStage.empty(bmir.typ.elementType)
    else lowerNonEmpty(bmir)
  }

  def lowerNonEmpty(bmir: BlockMatrixIR): BlockMatrixStage = bmir match {
    case BlockMatrixRead(reader) => unimplemented(bmir)
    case x: BlockMatrixLiteral => unimplemented(bmir)
    case BlockMatrixMap(child, eltName, f, needsDense) => unimplemented(bmir)
    case BlockMatrixMap2(left, right, lname, rname, f, sparsityStrategy) => unimplemented(bmir)
    case BlockMatrixBroadcast(child, inIndexExpr, shape, blockSize) => unimplemented(bmir)
    case BlockMatrixAgg(child, outIndexExpr) => unimplemented(bmir)
    case BlockMatrixFilter(child, keep) => unimplemented(bmir)
    case BlockMatrixSlice(child, slices) => unimplemented(bmir)
    case BlockMatrixDensify(child) => unimplemented(bmir)
    case BlockMatrixSparsify(child, sparsifier) => unimplemented(bmir)
    case RelationalLetBlockMatrix(name, value, body) => unimplemented(bmir)
    case ValueToBlockMatrix(child, shape, blockSize) if !child.typ.isInstanceOf[TArray] =>
      throw new LowererUnsupportedOperation("use explicit broadcast for scalars!")
    case x@ValueToBlockMatrix(child, _, blockSize) => // row major or scalar
      val v = Ref(genUID(), child.typ)
      val ctxs = x.typ.allBlocks.map { case (i, j) =>
        val slice = NDArraySlice(v,
          MakeTuple.ordered(FastSeq(
            MakeTuple.ordered(FastSeq[IR](i.toLong * blockSize, (i.toLong + 1) * blockSize, 1L)),
            MakeTuple.ordered(FastSeq[IR](j.toLong * blockSize, (j.toLong + 1) * blockSize, 1L)))))
        (i -> j, slice)
      }.toMap
      BlockMatrixStage(ctxs, Array(v.name -> child)) { ref: Ref => ref }
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
      val l = Ref(genUID(), leftIR.typ.elementType)
      val r = Ref(genUID(), rightIR.typ.elementType)
      val newBody = ArrayFold(tail, zero, accum.name, elt.name,
        NDArrayMap2(accum, wrapMultiply(elt), l.name, r.name, l + r))

      BlockMatrixStage(
        ctxRef,
        newContexts,
        left.globalVals ++ right.globalVals,
        newBody)
  }
}
