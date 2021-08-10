package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{CodeParamType, EmitCode, EmitCodeBuilder, EmitParamType, SCodeEmitParamType}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.SNDArrayPointerSettable
import is.hail.types.physical.stypes.interfaces.{SNDArray, SNDArrayCode, SNDArrayValue}
import is.hail.types.physical.{PCanonicalNDArray, PType}
import is.hail.types.virtual.Type
import is.hail.utils._

class NDArraySumAggregator(ndVTyp: VirtualTypeWithReq) extends StagedAggregator {
  private val ndTyp = ndVTyp.canonicalPType.setRequired(false).asInstanceOf[PCanonicalNDArray]

  override type State = TypedRegionBackedAggState

  override def resultType: PType = ndTyp

  override def initOpTypes: Seq[Type] = Array[Type]()

  override def seqOpTypes: Seq[Type] = Array(ndVTyp.t)

  val ndarrayFieldNumber = 0

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val initMethod = cb.emb.genEmitMethod[Unit]("ndarray_sum_aggregator_init_op")
    initMethod.voidWithBuilder(cb =>
      state.storeMissing(cb)
    )
    cb.invokeVoid(initMethod)
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(nextNDCode) = seq
    val seqOpMethod = cb.emb.genEmitMethod("ndarray_sum_aggregator_seq_op", FastIndexedSeq(nextNDCode.emitParamType), CodeParamType(UnitInfo))

    seqOpMethod.voidWithBuilder { cb =>
      val nextNDInput = seqOpMethod.getEmitParam(1, null) // no streams here
      nextNDInput.toI(cb).consume(cb, {}, { case nextNDArrayPCode: SNDArrayCode =>
        val nextNDPV = nextNDArrayPCode.memoize(cb, "ndarray_sum_seqop_next")
        val statePV = state.storageType.loadCheapSCode(cb, state.off).asBaseStruct.memoize(cb, "ndarray_sum_seq_op_state")
        statePV.loadField(cb, ndarrayFieldNumber).consume(cb,
          {
            cb += (state.region.getNewRegion(Region.TINY))
            cb += state.storageType.setFieldPresent(state.off, ndarrayFieldNumber)
            state.storeNonmissing(cb, nextNDPV)
          },
          { currentNDPCode =>
            val currentNDPValue = currentNDPCode.asNDArray.memoize(cb, "ndarray_sum_seqop_current")
            addValues(cb, state.region, currentNDPValue, nextNDPV)
          }
        )
      })
    }
    cb.invokeVoid(seqOpMethod, nextNDCode)
  }

  override protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    val combOpMethod = cb.emb.genEmitMethod[Unit]("ndarray_sum_aggregator_comb_op")

    combOpMethod.voidWithBuilder { cb =>
      val rightPV = other.storageType.loadCheapSCode(cb, other.off).asBaseStruct.memoize(cb, "ndarray_sum_comb_op_right")
      rightPV.loadField(cb, ndarrayFieldNumber).consume(cb, {},
        { rightNDPC =>
          val rightNdValue = rightNDPC.asNDArray.memoize(cb, "right_ndarray_sum_agg")
          val leftPV = state.storageType.loadCheapSCode(cb, state.off).asBaseStruct.memoize(cb, "ndarray_sum_comb_op_left")
          leftPV.loadField(cb, ndarrayFieldNumber).consume(cb,
            {
              state.storeNonmissing(cb, rightNdValue)
            },
            { leftNDPC =>
              val leftNdValue = leftNDPC.asNDArray.memoize(cb, "left_ndarray_sum_agg")
              addValues(cb, state.region, leftNdValue, rightNdValue)
            })
        }
      )
    }

    cb.invokeVoid(combOpMethod)
  }

  private def addValues(cb: EmitCodeBuilder, region: Value[Region], leftNdValue: SNDArrayValue, rightNdValue: SNDArrayValue): Unit = {

    cb.ifx(!leftNdValue.sameShape(rightNdValue, cb),
      cb += Code._fatal[Unit]("Can't sum ndarrays of different shapes."))

    SNDArray.coiterate(cb, region, FastIndexedSeq((leftNdValue.get, "left"), (rightNdValue.get, "right")), {
      case Seq(l, r) =>
        val newElement = SCode.add(cb, l, r, true)
        cb.assign(l, newElement.copyToRegion(cb, region, leftNdValue.st.elementType))
    })
  }

  protected def _storeResult(cb: EmitCodeBuilder, state: State, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit = {
    state.get(cb).consume(cb,
      ifMissing(cb),
      { sc => pt.storeAtAddress(cb, addr, region, sc, deepCopy = true) })
  }
}
