package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s.{UnitInfo, Value}
import is.hail.expr.ir.{CodeParamType, EmitCode, EmitCodeBuilder}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical.stypes.interfaces.{SBaseStructCode, SNDArray, SNDArrayCode}
import is.hail.types.physical.{PCanonicalBaseStruct, PCanonicalNDArray, PCanonicalTuple, PType}
import is.hail.types.virtual.Type
import is.hail.utils.{FastIndexedSeq, valueToRichCodeRegion}
import is.hail.asm4s.{Code, _}
import is.hail.linalg.LinalgCodeUtils

class NDArrayMultiplyAddAggregator(nDVTyp: VirtualTypeWithReq) extends StagedAggregator {
  private val ndTyp = nDVTyp.canonicalPType.setRequired(false).asInstanceOf[PCanonicalNDArray]

  override type State = TypedRegionBackedAggState

  override def resultType: PType = ndTyp

  override def initOpTypes: Seq[Type] = Array[Type]()

  override def seqOpTypes: Seq[Type] = Array(ndTyp.virtualType, ndTyp.virtualType)

  val ndarrayFieldNumber = 0

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val initMethod = cb.emb.genEmitMethod[Unit]("ndarray_multiply_add_aggregator_init_op")
    initMethod.voidWithBuilder(cb =>
      state.storeMissing(cb)
    )
    cb.invokeVoid(initMethod)
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(nextNDArrayACode, nextNDArrayBCode) = seq
    val seqOpMethod = cb.emb.genEmitMethod("ndarray_add_multiply_aggregator_seq_op",
      FastIndexedSeq(nextNDArrayACode.emitParamType, nextNDArrayBCode.emitParamType), CodeParamType(UnitInfo))
    seqOpMethod.voidWithBuilder { cb =>
      val ndArrayAEmitCode = seqOpMethod.getEmitParam(cb, 1, null)
      ndArrayAEmitCode.toI(cb).consume(cb, {}, { case nextNDArrayAPCode: SNDArrayCode =>
        val ndArrayBEmitCode = seqOpMethod.getEmitParam(cb, 2, null)
        ndArrayBEmitCode.toI(cb).consume(cb, {}, { case nextNDArrayBPCode: SNDArrayCode =>
          val checkA = nextNDArrayAPCode.memoize(cb, "ndarray_add_mutiply_seqop_A")
          val checkB = nextNDArrayBPCode.memoize(cb, "ndarray_add_mutiply_seqop_B")
          val tempRegionForCreation = cb.newLocal[Region]("ndarray_add_multily_agg_temp_region", Region.stagedCreate(Region.REGULAR, cb.emb.ecb.pool()))
          val NDArrayA = LinalgCodeUtils.checkColMajorAndCopyIfNeeded(checkA, cb, tempRegionForCreation)
          val NDArrayB = LinalgCodeUtils.checkColMajorAndCopyIfNeeded(checkB, cb, tempRegionForCreation)
          val statePV = state.storageType.loadCheapSCode(cb, state.off).asBaseStruct.memoize(cb, "ndarray_add_multiply_seqop_state")
          statePV.loadField(cb, ndarrayFieldNumber).consume(cb,
            {
              cb += state.region.getNewRegion(Region.REGULAR)
              cb += state.storageType.setFieldPresent(state.off, ndarrayFieldNumber)
              val shape = IndexedSeq(NDArrayA.shapes(0), NDArrayB.shapes(1))
              val uninitializedNDArray = ndTyp.constructUnintialized(shape, ndTyp.makeColumnMajorStrides(shape, tempRegionForCreation, cb), cb, tempRegionForCreation)
              state.storeNonmissing(cb, uninitializedNDArray.get)
              SNDArray.gemm(cb, "N", "N", const(1.0), NDArrayA.get, NDArrayB.get, const(0.0), uninitializedNDArray.get)
            },
            { currentNDPCode =>
              val currentNDPValue = currentNDPCode.asNDArray.memoize(cb, "ndarray_add_multiply_current")
              SNDArray.gemm(cb, "N", "N", NDArrayA.get, NDArrayB.get, currentNDPValue.get)
            }
          )
          cb += tempRegionForCreation.clearRegion()
        })
      })
    }
    cb.invokeVoid(seqOpMethod, nextNDArrayACode, nextNDArrayBCode)
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
