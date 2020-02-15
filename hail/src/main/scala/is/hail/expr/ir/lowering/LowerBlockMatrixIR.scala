package is.hail.expr.ir.lowering

import is.hail.expr.Nat
import is.hail.expr.ir._
import is.hail.expr.ir.functions.GetElement
import is.hail.expr.types.BlockMatrixSparsity
import is.hail.expr.types.virtual._
import is.hail.utils.{FastIndexedSeq, FastSeq}

object BlockMatrixStage {
  def empty(eltType: Type, nr: Int, nc: Int): BlockMatrixStage =
    EmptyBlockMatrixStage(nr, nc, eltType)
}

case class EmptyBlockMatrixStage(override val nRowBlocks: Int, override val nColBlocks: Int, eltType: Type) extends BlockMatrixStage(
  nRowBlocks, nColBlocks,
  BlockMatrixSparsity(Some(FastIndexedSeq())),
  Array(), TInt32()) {
  def blockContext(idx: (Int, Int)): IR =
    throw new LowererUnsupportedOperation("empty stage has no block contexts!")
  def blockBody(ctxRef: Ref): IR = NA(TNDArray(eltType, Nat(2)))
  override def collectBlocks(f: IR => IR, blocksToCollect: Array[(Int, Int)]): IR = {
    assert(blocksToCollect.isEmpty)
    MakeArray(FastSeq(), TArray(f(blockBody(Ref("x", ctxType))).typ))
  }
}

abstract class BlockMatrixStage(
  val nRowBlocks: Int, val nColBlocks: Int,
  val sparsity: BlockMatrixSparsity,
  val globalVals: Array[(String, IR)],
  val ctxType: Type
) {
  def blockContext(idx: (Int, Int)): IR
  def blockBody(ctxRef: Ref): IR
  lazy val blocks: IndexedSeq[(Int, Int)] = sparsity.allBlocks(nRowBlocks, nColBlocks)

  def defines(idx: (Int, Int)): Boolean = sparsity.hasBlock(idx)
  def collectBlocks(f: IR => IR, blocksToCollect: Array[(Int, Int)]): IR = {
    assert(blocksToCollect.forall(b => defines(b)))
    val ctxRef = Ref(genUID(), ctxType)
    val body = f(blockBody(ctxRef))
    val ctxs = MakeArray(blocksToCollect.map(idx => blockContext(idx)), TArray(ctxRef.typ))
    val bcFields = globalVals.filter { case (f, _) => Mentions(body, f) }
    val bcVals = MakeStruct(bcFields.map { case (f, v) => f -> Ref(f, v.typ) })
    val bcRef = Ref(genUID(), bcVals.typ)
    val wrappedBody = bcFields.foldLeft(body) { case (accum, (f, _)) =>
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
      val blocksRowMajor = Array.range(0, bm.nRowBlocks).flatMap { i =>
        Array.range(0, bm.nColBlocks).flatMap { j => Some(i -> j).filter(bm.defines) }
      }
      val cda = bm.collectBlocks(b => b, blocksRowMajor)
      val blockResults = Ref(genUID(), cda.typ)

      val rows = if (bm.sparsity.isSparse) {
        val blockMap = blocksRowMajor.zipWithIndex.toMap
        MakeArray(Array.tabulate[IR](bm.nRowBlocks) { i =>
          NDArrayConcat(MakeArray(Array.tabulate[IR](bm.nColBlocks) { j =>
            if (blockMap.contains(i -> j))
              ArrayRef(blockResults, i * bm.nColBlocks + j)
            else {
              val (nRows, nCols) = child.typ.blockShape(i, j)
              MakeNDArray.fill(zero(child.typ.elementType), FastIndexedSeq(nRows, nCols), True())
            }
          }, coerce[TArray](cda.typ)), 1)
        }, coerce[TArray](cda.typ))
      } else {
        val i = Ref(genUID(), TInt32())
        val j = Ref(genUID(), TInt32())
        val cols = ArrayMap(ArrayRange(0, bm.nColBlocks, 1), j.name, ArrayRef(blockResults, i * bm.nColBlocks + j))
        ArrayMap(ArrayRange(0, bm.nRowBlocks, 1), i.name, NDArrayConcat(cols, 1))
      }
      Let(blockResults.name, cda, NDArrayConcat(rows, 0))
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
      BlockMatrixStage.empty(bmir.typ.elementType, bmir.typ.nRowBlocks, bmir.typ.nColBlocks)
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
      val nd = MakeNDArray(child, MakeTuple.ordered(FastSeq(I64(x.typ.nRows), I64(x.typ.nCols))), True())
      val v = Ref(genUID(), nd.typ)
      new BlockMatrixStage(
        x.typ.nRowBlocks, x.typ.nColBlocks,
        x.typ.sparsity,
        Array(v.name -> nd),
        nd.typ) {
        def blockContext(idx: (Int, Int)): IR = {
          val (r, c) = idx
          NDArraySlice(v, MakeTuple.ordered(FastSeq(
            MakeTuple.ordered(FastSeq(I64(r.toLong * blockSize), I64(java.lang.Math.min((r.toLong + 1) * blockSize, x.typ.nRows)), I64(1))),
            MakeTuple.ordered(FastSeq(I64(c.toLong * blockSize), I64(java.lang.Math.min((c.toLong + 1) * blockSize, x.typ.nCols)), I64(1))))))
        }
        def blockBody(ctxRef: Ref): IR = ctxRef
      }
    case x@BlockMatrixDot(leftIR, rightIR) =>
      val left = lower(leftIR)
      val right = lower(rightIR)
      val newCtxType = TArray(TTuple(left.ctxType, right.ctxType))
      new BlockMatrixStage(
        x.typ.nRowBlocks, x.typ.nColBlocks,
        x.typ.sparsity,
        left.globalVals ++ right.globalVals,
        newCtxType) {
        def blockContext(idx: (Int, Int)): IR = {
          val (i, j) = idx
          MakeArray(Array.tabulate[Option[IR]](left.nColBlocks) { k =>
            if (left.defines(i -> k) && right.defines(k -> j))
              Some(MakeTuple.ordered(FastSeq(
                left.blockContext(i -> k), right.blockContext(k -> j))))
            else None
          }.flatten[IR], newCtxType)
        }

        def blockBody(ctxRef: Ref): IR = {
          val ctxEltRef = Ref(genUID(), newCtxType.elementType)
          val leftRef = Ref(genUID(), left.ctxType)
          val rightRef = Ref(genUID(), right.ctxType)

          val blockMultiply = Let(leftRef.name, GetTupleElement(ctxEltRef, 0),
            Let(rightRef.name, GetTupleElement(ctxEltRef, 1),
              NDArrayMatMul(left.blockBody(leftRef), right.blockBody(rightRef))))

          val sumRef = Ref(genUID(), blockMultiply.typ)
          ArrayFold(invoke("[*:]", ctxType, ctxRef, I32(1)),
            Let(ctxEltRef.name, ArrayRef(ctxRef, 0), blockMultiply),
            sumRef.name,
            ctxEltRef.name,
            NDArrayMap2(sumRef, blockMultiply, "l", "r",
              Ref("l", x.typ.elementType) + Ref("r", x.typ.elementType)))
        }
      }
  }
}
