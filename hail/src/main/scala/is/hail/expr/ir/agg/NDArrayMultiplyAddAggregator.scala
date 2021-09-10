package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s.{UnitInfo, Value}
import is.hail.expr.ir.{CodeParamType, EmitCode, EmitCodeBuilder}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical.stypes.interfaces.{SBaseStructCode, SNDArray}
import is.hail.types.physical.{PCanonicalNDArray, PType}
import is.hail.types.virtual.Type
import is.hail.utils.{FastIndexedSeq, valueToRichCodeRegion}
import is.hail.asm4s.{Code, _}

class NDArrayMultiplyAddAggregator(ndVTyp: VirtualTypeWithReq) extends StagedAggregator {
  private val ndTyp = ndVTyp.canonicalPType.setRequired(false).asInstanceOf[PCanonicalNDArray]
  override type State = TypedRegionBackedAggState

  override def resultType: PType = ndTyp

  override def initOpTypes: Seq[Type] = Array[Type]()

  override def seqOpTypes: Seq[Type] = Array(ndVTyp.t)

  val ndarrayFieldNumber = 0

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val initMethod = cb.emb.genEmitMethod[Unit]("ndarray_multiply_add_aggregator_init_op")
    initMethod.voidWithBuilder(cb =>
      state.storeMissing(cb)
    )
    cb.invokeVoid(initMethod)
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(nextNDTupleCode) = seq
    val seqOpMethod = cb.emb.genEmitMethod("ndarray_add_multiply_aggregator_seq_op", FastIndexedSeq(nextNDTupleCode.emitParamType), CodeParamType(UnitInfo))
    seqOpMethod.voidWithBuilder {
      val nextNDTupleInput = seqOpMethod.getEmitParam(1, null)
      nextNDTupleInput.toI(cb).consume(cb, {}, { case nextNDArrayTuplePCode: SBaseStructCode =>
        val nextNDTV = nextNDArrayTuplePCode.memoize(cb, "ndarry_add_multiply_seqop_next")
        val NDArrayA = nextNDTV.loadField(cb, 1).get(cb).asNDArray.memoize(cb, "ndarray_add_mutiply_seqop_A")
        val NDArrayB = nextNDTV.loadField(cb, 2).get(cb).asNDArray.memoize(cb, "ndarray_add_mutiply_seqop_B")
        val statePV = state.storageType.loadCheapSCode(cb, state.off).asBaseStruct.memoize(cb, "mdarray_add_multiply_seqop_state")
        statePV.loadField(cb, ndarrayFieldNumber).consume(cb,
          {
            cb += state.region.getNewRegion(Region.TINY)
            cb += state.storageType.setFieldPresent(state.off, ndarrayFieldNumber)
            val tempRegionForCreation = cb.newLocal[Region]("ndarray_add_multiply_agg_temp_region", Region.stagedCreate(Region.REGULAR, cb.emb.ecb.pool()))
            val uninitializedNDArray = ndTyp.constructUnintialized(NDArrayA.shapes, NDArrayA.strides, cb, tempRegionForCreation).memoize(cb, "ndarray_sum_seq_op_uninitialized")
            state.storeNonmissing(cb, uninitializedNDArray)
            cb += tempRegionForCreation.clearRegion()
            SNDArray.gemm(cb, "N", "N", const(1.0), NDArrayA.get, NDArrayB.get, const(0.0), uninitializedNDArray)
          },
          { currentNDPCode =>
            val currentNDPValue = currentNDPCode.asNDArray.memoize(cb, "ndarray_add_multiply_current")
            SNDArray.gemm(cb, "N", "N", NDArrayA.get, NDArrayB.get, currentNDPValue)
          }
        )
      })
    }
    cb.invokeVoid(seqOpMethod, nextNDTupleCode)
  }
  override protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    val combOpMethod = cb.emb.genEmitMethod[Unit]("ndarraymutiply_add_agg_comb_op")

    combOpMethod.voidWithBuilder { cb =>
      val rightPV = other.storageType.loadCheapSCode(cb, other.off).asBaseStruct.memoize(cb, "ndarray_mutiply_add_comb_op_right")
      rightPV.loadField(cb, ndarrayFieldNumber).consume(cb, {},
        { rightNDPC =>
          val rightNdValue = rightNDPC.asNDArray.memoize(cb, "right_ndarray_mutiply_add_agg")
          val leftPV = state.storageType.loadCheapSCode(cb, state.off).asBaseStruct.memoize(cb, "ndarray_mutiply_add_comb_op_left")
          leftPV.loadField(cb, ndarrayFieldNumber).consume(cb,
            {
              state.storeNonmissing(cb, rightNdValue)
            },
            { leftNDPC =>
              val leftNdValue = leftNDPC.asNDArray.memoize(cb, "left_ndarray_mutiply_add_agg")
              NDArraySumAggregator.addValues(cb, state.region, leftNdValue, rightNdValue)
            })
        }
      )
    }
    cb.invokeVoid(combOpMethod)
  }
  override protected def _storeResult(cb: EmitCodeBuilder, state: State, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit = {
    state.get(cb).consume(cb,
      ifMissing(cb),
      { sc =>
        val lastNDInAggState = sc.asNDArray.memoize(cb, "ndarray_multiply_add_agg_last_state")
        pt.storeAtAddress(cb, addr, region, lastNDInAggState, deepCopy = true)
      })
  }
}
