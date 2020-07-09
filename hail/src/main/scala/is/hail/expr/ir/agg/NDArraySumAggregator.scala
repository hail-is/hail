package is.hail.expr.ir.agg

import is.hail.annotations.StagedRegionValueBuilder
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, IR}
import is.hail.types.physical.{PCanonicalTuple, PCode, PNDArray, PNDArrayCode, PNDArrayValue, PType}

object NDArraySumAggregator {

}

class NDArraySumAggregator (ndTyp: PNDArray, knownShape: Option[IR]) extends StagedAggregator {
  override type State = TypedRegionBackedAggState

  // State needs:
  // 1. Initialized or not
  // 2.
  val stateType = PCanonicalTuple(true, ???, ndTyp)

  override def resultType: PType = ndTyp

  override def initOpTypes: Seq[PType] = ???

  override def seqOpTypes: Seq[PType] = ???

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {

    ???
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = ???

  override protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    // Need to:
    // 1. Check the shapes match and the strides match.
    // 2. If they match, add.
    val leftCode = PCode(ndTyp, state.off).asInstanceOf[PNDArrayCode]
    val rightCode = PCode(ndTyp, other.off).asInstanceOf[PNDArrayCode]
    val leftValue = cb.memoize(leftCode, "left_pndarraycode").asInstanceOf[PNDArrayValue]
    val rightValue = cb.memoize(rightCode, "right_pndarraycode").asInstanceOf[PNDArrayValue]
    val comparator = ndTyp.codeOrdering(cb.emb, ndTyp)

    ???
  }

  override protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit = ???
}
