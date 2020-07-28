package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, IR}
import is.hail.types.physical.{PBaseStructCode, PBoolean, PCanonicalTuple, PCode, PNDArray, PNDArrayCode, PNDArrayValue, PType}

object NDArraySumAggregator {
}

class NDArraySumAggregator (ndTyp: PNDArray, knownShape: Option[IR]) extends StagedAggregator {
  override type State = TypedRegionBackedAggState

  // State needs:
  // 1. Initialized or not
  // 2. The ndarray itself.
  val stateType = PCanonicalTuple(true, PBoolean(), ndTyp)

  override def resultType: PType = ndTyp

  override def initOpTypes: Seq[PType] = Array[PType]()//Array(ndTyp.shape.pType)

  override def seqOpTypes: Seq[PType] = Array(ndTyp)

  def isInitialized(state: State): Code[Boolean] = {
    Region.loadBoolean(stateType.fieldOffset(state.off, 0))
  }

  def ndArrayPointer(state: State): Code[Long] = {
    Region.loadLong(stateType.fieldOffset(state.off, 1))
  }

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    cb.append(Code(
      state.off := stateType.allocate(state.region),
      Region.storeBoolean(stateType.fieldOffset(state.off, 0), false)
    ))
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    // Check if the thing is initialized. If it is, then check shapes match. If they do, add ndarrays.
    val Array(nextNDCode) = seq
    nextNDCode.toI(cb).consume(cb, {}, {case nextNDPCode: PNDArrayCode =>
      val nextNDPValue = nextNDPCode.memoize(cb, "ndarray_sum_seqop_next")
      cb.ifx(isInitialized(state),
        {
          val currentNDPValue = PCode(ndTyp, ndArrayPointer(state)).asNDArray.memoize(cb, "ndarray_sum_seqop_current")

          addValues(cb, currentNDPValue, nextNDPValue)
        },
        {
          // TODO Is this really safe, does it copy the initial array?
          ndTyp.constructAtAddress(cb.emb, stateType.fieldOffset(state.off, 1), state.region, nextNDCode.pt, nextNDPValue.get.tcode[Long], true)
        }
      )
    })

  }

  override protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    // Need to:
    // 1. Check the shapes match.
    // 2. If they match, add.
    val leftValue = PCode(stateType, state.off).asBaseStruct.memoize(cb, "left_state_ndarray_sum_agg")
    val rightValue = PCode(stateType, other.off).asBaseStruct.memoize(cb, "right_state_ndarray_sum_agg")
    val leftNdValue = leftValue.loadField(cb, 1).pc.asNDArray.memoize(cb, "left_ndarray_sum_agg")
    val rightNdValue = rightValue.loadField(cb, 1).pc.asNDArray.memoize(cb, "right_ndarray_sum_agg")

    addValues(cb, leftNdValue, rightNdValue)

  }

  private def addValues(cb: EmitCodeBuilder, leftNdValue: PNDArrayValue, rightNdValue: PNDArrayValue): Unit = {
    val sameShape = leftNdValue.sameShape(rightNdValue, cb.emb)

    val idxVars = Array.tabulate(ndTyp.nDims) { _ => cb.emb.genFieldThisRef[Long]() }

    val body = ndTyp.mutateElement(
      idxVars,
      leftNdValue.value.asInstanceOf[Value[Long]],
      ndTyp.loadElementToIRIntermediate(idxVars, rightNdValue.value.asInstanceOf[Value[Long]], cb.emb),
      cb.emb
    )

    val leftNdShape = PCode.apply(ndTyp.shape.pType, ndTyp.shape.load(leftNdValue.value.asInstanceOf[Value[Long]])).asBaseStruct.memoize(cb, "left_nd_shape")

    val columnMajorLoops = idxVars.zipWithIndex.foldLeft(body) { case (innerLoops, (dimVar, dimIdx)) =>
      Code(
        dimVar := 0L,
        Code.whileLoop(dimVar < leftNdShape(dimIdx),
          innerLoops,
          dimVar := dimVar + 1L
        )
      )
    }

    cb.append(sameShape.mux(
      columnMajorLoops,
      Code._fatal[Unit]("Can't sum ndarrays of different shapes.")
    ))
  }

  override protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit = ???
}
