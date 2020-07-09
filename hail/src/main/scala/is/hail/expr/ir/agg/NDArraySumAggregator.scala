package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.Code
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, IR}
import is.hail.types.physical.{PBoolean, PCanonicalTuple, PCode, PNDArray, PNDArrayCode, PNDArrayValue, PType}

object NDArraySumAggregator {
}

class NDArraySumAggregator (ndTyp: PNDArray, knownShape: Option[IR]) extends StagedAggregator {
  override type State = TypedRegionBackedAggState

  // State needs:
  // 1. Initialized or not
  // 2. The ndarray itself.
  val stateType = PCanonicalTuple(true, PBoolean(), ndTyp)

  override def resultType: PType = ndTyp

  override def initOpTypes: Seq[PType] = ???

  override def seqOpTypes: Seq[PType] = ???

  def isInitialized(state: State): Code[Boolean] = {
    Region.loadBoolean(stateType.fieldOffset(state.off, 0))
  }

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    cb.append(Code(
      state.off := stateType.allocate(state.region),
      Region.storeBoolean(stateType.fieldOffset(state.off, 0), false)
    ))
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    // Check if the thing is initialized. If it is, then check shapes match. If they do, add ndarrays.
    val Array(nextNDArray) = seq
    isInitialized(state).mux(
      ???,
      // TODO Is this really safe, does it copy the initial array?
      ndTyp.constructAtAddress(cb.emb, stateType.fieldOffset(state.off, 1), state.region, nextNDArray.pt, nextNDArray.value[Long], true)
    )
    ???
  }

  override protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    // Need to:
    // 1. Check the shapes match and the strides match.
    // 2. If they match, add.
    val leftCode = PCode(ndTyp, state.off).asInstanceOf[PNDArrayCode]
    val rightCode = PCode(ndTyp, other.off).asInstanceOf[PNDArrayCode]
    val leftValue = cb.memoize(leftCode, "left_pndarraycode").asInstanceOf[PNDArrayValue]
    val rightValue = cb.memoize(rightCode, "right_pndarraycode").asInstanceOf[PNDArrayValue]


    ???
  }

  override protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit = ???
}
