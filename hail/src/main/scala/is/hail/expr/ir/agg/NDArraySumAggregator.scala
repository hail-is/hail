package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{CodeParamType, EmitCode, EmitCodeBuilder, EmitParamType, coerce}
import is.hail.types.physical.{PCanonicalTuple, PNDArray, PNDArrayCode, PNDArrayValue, PNumeric, PType}
import is.hail.utils._

class NDArraySumAggregator (ndTyp: PNDArray) extends StagedAggregator {
  override type State = TypedRegionBackedAggState

  override def resultType: PType = ndTyp

  val stateType = PCanonicalTuple(true, ndTyp)

  override def initOpTypes: Seq[PType] = Array[PType]()

  override def seqOpTypes: Seq[PType] = Array(ndTyp)

  val ndarrayFieldNumber = 0

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val initMethod = cb.emb.genEmitMethod[Unit]("ndarray_sum_aggregator_init_op")
    initMethod.voidWithBuilder(cb =>
      cb.append(stateType.setFieldMissing(state.off, ndarrayFieldNumber))
    )
    cb.invokeVoid(initMethod)
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(nextNDCode) = seq
    val seqOpMethod = cb.emb.genEmitMethod("ndarray_sum_aggregator_seq_op", FastIndexedSeq(EmitParamType(nextNDCode.pt)), CodeParamType(UnitInfo))

    seqOpMethod.voidWithBuilder(cb => {
      val nextNDInput = seqOpMethod.getEmitParam(1)
      val statePV = stateType.getPointerTo(cb, state.off).asBaseStruct.memoize(cb, "ndarray_sum_seq_op_state")
      nextNDInput.toI(cb).consume(cb, {}, {case nextNDArrayPCode: PNDArrayCode =>
        val nextNDPV = nextNDArrayPCode.memoize(cb, "ndarray_sum_seqop_next")
        statePV.loadField(cb, ndarrayFieldNumber).consume(cb,
          {
            cb.append(state.region.getNewRegion(Region.TINY))
            cb.append(stateType.setFieldPresent(state.off, ndarrayFieldNumber))
            ndTyp.storeAtAddress(
              cb,
              stateType.fieldOffset(state.off, ndarrayFieldNumber),
              state.region,
              nextNDPV,
              true)
          },
          { currentNDPCode =>
            val currentNDPValue = currentNDPCode.asNDArray.memoize(cb, "ndarray_sum_seqop_current")
            addValues(cb, currentNDPValue, nextNDPV)
          }
        )
      })
    })
    cb.invokeVoid(seqOpMethod, nextNDCode)
  }

  override protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    val combOpMethod = cb.emb.genEmitMethod[Unit]("ndarray_sum_aggregator_comb_op")

    combOpMethod.voidWithBuilder(cb => {
      val rightPV = stateType.getPointerTo(cb, other.off).asBaseStruct.memoize(cb, "ndarray_sum_comb_op_right")
      rightPV.loadField(cb, ndarrayFieldNumber).consume(cb, {},
        { rightNDPC =>
          val leftPV = stateType.getPointerTo(cb, state.off).asBaseStruct.memoize(cb, "ndarray_sum_comb_op_left")
          leftPV.loadField(cb, ndarrayFieldNumber).consume(cb,
            {
              cb.append(state.storeNonmissing(other.off))
            },
            { leftNDPC =>
              val leftNdValue = leftNDPC.asNDArray.memoize(cb, "left_ndarray_sum_agg")
              val rightNdValue = rightNDPC.asNDArray.memoize(cb, "right_ndarray_sum_agg")
              addValues(cb, leftNdValue, rightNdValue)
            })
        }
      )
    })

    cb.invokeVoid(combOpMethod)
  }

  private def addValues(cb: EmitCodeBuilder, leftNdValue: PNDArrayValue, rightNdValue: PNDArrayValue): Unit = {
    val sameShape = leftNdValue.sameShape(rightNdValue, cb)

    val idxVars = Array.tabulate(ndTyp.nDims) { _ => cb.emb.genFieldThisRef[Long]() }

    def loadElement(ndValue: PNDArrayValue) = {
      ndTyp.loadElementToIRIntermediate(idxVars, ndValue.value.asInstanceOf[Value[Long]], cb.emb)
    }

    val newElement = coerce[PNumeric](ndTyp.elementType).add(loadElement(leftNdValue), loadElement(rightNdValue))

    val body = ndTyp.setElement(
      idxVars,
      leftNdValue.value.asInstanceOf[Value[Long]],
      newElement,
      cb.emb
    )

    val leftNdShape = leftNdValue.shapes(cb)

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

  override protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit = {
    val t = state.get()
    cb.append(t.setup)
    cb.append(
      stateType.isFieldDefined(state.off, ndarrayFieldNumber).mux(
        srvb.addWithDeepCopy(resultType, stateType.loadField(state.off, ndarrayFieldNumber)),
        srvb.setMissing()
    ))
  }
}
