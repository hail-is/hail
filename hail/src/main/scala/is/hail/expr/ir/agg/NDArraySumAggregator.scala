package is.hail.expr.ir.agg

import is.hail.annotations.StagedRegionValueBuilder
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, IR}
import is.hail.types.physical.{PCode, PNDArray, PNDArrayCode, PNDArrayValue, PType}

class NDArraySumAggregator (typ: PNDArray, knownShape: Option[IR]) extends StagedAggregator {
  override type State = TypedRegionBackedAggState

  override def resultType: PType = typ // Input and output same type seems fine.

  override def initOpTypes: Seq[PType] = ???

  override def seqOpTypes: Seq[PType] = ???

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = ???

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = ???

  override protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    // Need to:
    // 1. Check the shapes match and the strides match.
    // 2. If they match, add.
    val leftCode = PCode(typ, state.off).asInstanceOf[PNDArrayCode]
    val rightCode = PCode(typ, other.off).asInstanceOf[PNDArrayCode]
    val leftValue = cb.memoize(leftCode, "left_pndarraycode").asInstanceOf[PNDArrayValue]
    val rightValue = cb.memoize(rightCode, "right_pndarraycode").asInstanceOf[PNDArrayValue]
    val comparator = typ.codeOrdering(cb.emb, typ)
    
    ???
  }

  override protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit = ???
}
