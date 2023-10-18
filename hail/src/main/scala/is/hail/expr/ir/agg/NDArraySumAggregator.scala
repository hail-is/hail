package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{CodeParamType, EmitCode, EmitCodeBuilder, IEmitCode}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical.PCanonicalNDArray
import is.hail.types.physical.stypes.interfaces.SNDArrayValue
import is.hail.types.physical.stypes.{EmitType, SCode}
import is.hail.types.virtual.Type
import is.hail.utils._

class NDArraySumAggregator(ndVTyp: VirtualTypeWithReq) extends StagedAggregator {
  override type State = TypedRegionBackedAggState

  override def resultEmitType: EmitType = ndVTyp.canonicalEmitType
  private val ndTyp = resultEmitType.storageType.asInstanceOf[PCanonicalNDArray] // TODO: Set required false?

  override def initOpTypes: Seq[Type] = Array[Type]()

  override def seqOpTypes: Seq[Type] = Array(ndVTyp.t)

  val ndarrayFieldNumber = 0

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val initMethod = cb.emb.genEmitMethod[Unit]("ndarray_sum_aggregator_init_op")
    initMethod.voidWithBuilder(cb =>
      state.storeMissing(cb)
    )
    cb.invokeVoid(initMethod, cb._this)
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(nextNDCode) = seq
    val seqOpMethod = cb.emb.genEmitMethod("ndarray_sum_aggregator_seq_op", FastSeq(nextNDCode.emitParamType), CodeParamType(UnitInfo))

    seqOpMethod.voidWithBuilder { cb =>
      val nextNDInput = seqOpMethod.getEmitParam(cb, 1)
      nextNDInput.toI(cb).consume(cb, {}, { case nextNDPV: SNDArrayValue =>
        val statePV = state.storageType.loadCheapSCode(cb, state.off).asBaseStruct
        statePV.loadField(cb, ndarrayFieldNumber).consume(cb,
          {
            cb += state.region.getNewRegion(Region.TINY)
            state.storageType.setFieldPresent(cb, state.off, ndarrayFieldNumber)
            val tempRegionForCreation = cb.newLocal[Region]("ndarray_sum_agg_temp_region", Region.stagedCreate(Region.REGULAR, cb.emb.ecb.pool()))
            val fullyCopiedNDArray = ndTyp.constructByActuallyCopyingData(nextNDPV, cb, tempRegionForCreation)
            state.storeNonmissing(cb, fullyCopiedNDArray)
            cb += tempRegionForCreation.clearRegion()
          },
          { currentND =>
            NDArraySumAggregator.addValues(cb, state.region, currentND.asNDArray, nextNDPV)
          }
        )
      })
    }
    cb.invokeVoid(seqOpMethod, cb._this, nextNDCode)
  }

  override protected def _combOp(ctx: ExecuteContext, cb: EmitCodeBuilder, state: TypedRegionBackedAggState, other: TypedRegionBackedAggState): Unit = {
    val combOpMethod = cb.emb.genEmitMethod[Unit]("ndarray_sum_aggregator_comb_op")

    combOpMethod.voidWithBuilder { cb =>
      val rightPV = other.storageType.loadCheapSCode(cb, other.off).asBaseStruct
      rightPV.loadField(cb, ndarrayFieldNumber).consume(cb, {},
        { case rightNdValue: SNDArrayValue =>
          val leftPV = state.storageType.loadCheapSCode(cb, state.off).asBaseStruct
          leftPV.loadField(cb, ndarrayFieldNumber).consume(cb,
            {
              state.storeNonmissing(cb, rightNdValue)
            },
            { case leftNdValue: SNDArrayValue =>
              NDArraySumAggregator.addValues(cb, state.region, leftNdValue, rightNdValue)
            })
        }
      )
    }
    cb.invokeVoid(combOpMethod, cb._this)
  }

  protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region]): IEmitCode = {
    state.get(cb).map(cb)(sv => sv.copyToRegion(cb, region, sv.st))
  }
}

object NDArraySumAggregator {

  def addValues(cb: EmitCodeBuilder, region: Value[Region], leftNdValue: SNDArrayValue, rightNdValue: SNDArrayValue): Unit = {
    cb.if_(!leftNdValue.sameShape(cb, rightNdValue),
      cb += Code._fatal[Unit]("Can't sum ndarrays of different shapes."))

    leftNdValue.coiterateMutate(cb, region, (rightNdValue, "right")) {
      case Seq(l, r) =>
        val newElement = SCode.add(cb, l, r, true)
        newElement.copyToRegion(cb, region, leftNdValue.st.elementType)
    }
  }
}
