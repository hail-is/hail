package is.hail.expr.ir.lowering

import is.hail.expr.ir._
import is.hail.expr.types.BlockMatrixType
import is.hail.expr.types.virtual._
import is.hail.utils.{FastIndexedSeq, FastSeq}

case class BlockMatrixStage(
  ctxName: String,
  ctxType: Type,
  blockContexts: Array[Array[Option[IR]]],
  broadcastVals: Map[String, IR], //optional, for a very specific type of broadcast.
  body: IR) {

  private lazy val ctxID: String = genUID()
  private lazy val broadcastID: String = genUID()

  private lazy val bcs = MakeStruct(broadcastVals.toIndexedSeq)

  def toIR(bodyTransform: IR => IR, ordering: Option[Array[(Int, Int)]]): IR = {
    val ctxs = ordering.map { idxs =>
      idxs.map { case (i, j) => blockContexts(i)(j).get }
    }.getOrElse {
      blockContexts.flatMap(rows => rows.flatten)
    }
    val wrappedBody = bcs.fields.foldLeft(bodyTransform(body)) { case (accum, (name, _)) =>
        Let(name, GetField(Ref(broadcastID, bcs.typ), name), accum)
    }
    CollectDistributedArray(MakeArray(ctxs, TArray(ctxType)), bcs, ctxID, broadcastID, wrappedBody)
  }
}


object LowerBlockMatrixIR {

  def lower(node: IR): IR = node match {
//    case BlockMatrixCollect(child: BlockMatrixIR) =>
//      val bm = lower(child)
//      val lowered = bm.toIR(nd => nd, Some(child.typ.presentBlocks))
//      if (child.typ.isSparse) {
//        // insert dense blocks
//        throw new LowererUnsupportedOperation(s"unimplemented: \n${ Pretty(node) }")
//      } else {
//        throw new LowererUnsupportedOperation(s"unimplemented: \n${ Pretty(node) }")
//      }

    case BlockMatrixMultiWrite(blockMatrices, writer) =>
      throw new LowererUnsupportedOperation(s"unimplemented: \n${ Pretty(node) }")
    case BlockMatrixToValueApply(child, function) =>
      throw new LowererUnsupportedOperation(s"unimplemented: \n${ Pretty(node) }")
    case BlockMatrixWrite(child, writer) =>
      throw new LowererUnsupportedOperation(s"unimplemented: \n${ Pretty(node) }")

    case node if node.children.exists( _.isInstanceOf[BlockMatrixIR] ) =>
      throw new LowererUnsupportedOperation(s"IR nodes with BlockMatrixIR children must be defined explicitly: \n${ Pretty(node) }")

    case node =>
      throw new LowererUnsupportedOperation(s"Value IRs with no BlockMatrixIR children must be lowered through LowerIR: \n${ Pretty(node) }")
  }

  def lower(bmir: BlockMatrixIR): BlockMatrixStage = bmir match {
    case BlockMatrixRead(reader) =>
      throw new LowererUnsupportedOperation(s"unimplemented: \n${ Pretty(bmir) }")
    case BlockMatrixMap(child, eltName, f) =>
      throw new LowererUnsupportedOperation(s"unimplemented: \n${ Pretty(bmir) }")
    case x@BlockMatrixDot(leftIR, rightIR) =>
      val left = lower(leftIR)
      val right = lower(rightIR)
      val (_, n) = leftIR.typ.defaultBlockShape

      val newCtxType = TArray(TStruct(
        left.ctxName -> TArray(left.ctxType),
        right.ctxName -> TArray(right.ctxType)))
      val newCtx = Ref(genUID(), newCtxType)

      // group blocks for multiply
      // the contexts that we're parallelizing, *in general*, are going to involve little-to-no computation so duplicating across nodes seems fine for now.
      val newContexts = Array.tabulate(x.typ.nRowBlocks) { i =>
        Array.tabulate[Option[IR]](x.typ.nColBlocks) { j =>
          if (!x.typ.definedBlocks(i)(j)) None else {
            Some(MakeArray(Array.tabulate[Option[IR]](n) { k =>
              left.blockContexts(i)(k).flatMap { leftCtx =>
                right.blockContexts(k)(j).map { rightCtx =>
                  MakeStruct(FastIndexedSeq(
                    left.ctxName -> leftCtx,
                    right.ctxName -> rightCtx))
                }
              }
            }.flatten, newCtxType))
          }
        }
      }

      val wrapMultiply = { ctxElt: IR =>
        Let(left.ctxName, GetField(ctxElt, left.ctxName),
          Let(right.ctxName, GetField(ctxElt, right.ctxName),
            NDArrayMatMul(left.body, right.body)))
      }

      // computation for multiply
      val zero = wrapMultiply(ArrayRef(newCtx, 0))
      val tail = invoke("[*:]", newCtxType, newCtx, 1)
      val elt = Ref(genUID(), newCtxType.elementType)
      val accum = Ref(genUID(), zero.typ)
      val l = Ref(genUID(), TFloat64())
      val r = Ref(genUID(), TFloat64())
      val newBody = ArrayFold(tail, zero, accum.name, elt.name,
        NDArrayMap2(accum, wrapMultiply(elt), l.name, r.name, ApplyBinaryPrimOp(Add(), l, r)))

      BlockMatrixStage(
        newCtx.name, newCtx.typ,
        newContexts,
        left.broadcastVals ++ right.broadcastVals,
        newBody)
  }

}
