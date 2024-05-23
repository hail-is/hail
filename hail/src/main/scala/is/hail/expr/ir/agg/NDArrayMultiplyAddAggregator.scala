package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{CodeParamType, EmitCode, EmitCodeBuilder, IEmitCode}
import is.hail.linalg.LinalgCodeUtils
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical.PCanonicalNDArray
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.interfaces.{SNDArray, SNDArrayValue}
import is.hail.types.virtual.Type
import is.hail.utils.{valueToRichCodeRegion, FastSeq}

class NDArrayMultiplyAddAggregator(ndVTyp: VirtualTypeWithReq) extends StagedAggregator {
  override type State = TypedRegionBackedAggState

  override def resultEmitType: EmitType = ndVTyp.canonicalEmitType

  private val ndTyp =
    resultEmitType.storageType.asInstanceOf[PCanonicalNDArray] // TODO: Set required false?

  override def initOpTypes: Seq[Type] = Array[Type]()

  override def seqOpTypes: Seq[Type] = Array(ndTyp.virtualType, ndTyp.virtualType)

  val ndarrayFieldNumber = 0

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val initMethod = cb.emb.genEmitMethod[Unit]("ndarray_multiply_add_aggregator_init_op")
    initMethod.voidWithBuilder(cb =>
      state.storeMissing(cb)
    )
    cb.invokeVoid(initMethod, cb.this_)
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(nextNDArrayACode, nextNDArrayBCode) = seq
    val seqOpMethod = cb.emb.genEmitMethod(
      "ndarray_add_multiply_aggregator_seq_op",
      FastSeq(nextNDArrayACode.emitParamType, nextNDArrayBCode.emitParamType),
      CodeParamType(UnitInfo),
    )
    seqOpMethod.voidWithBuilder { cb =>
      val ndArrayAEmitCode = seqOpMethod.getEmitParam(cb, 1)
      ndArrayAEmitCode.toI(cb).consume(
        cb,
        {},
        { case checkA: SNDArrayValue =>
          val ndArrayBEmitCode = seqOpMethod.getEmitParam(cb, 2)
          ndArrayBEmitCode.toI(cb).consume(
            cb,
            {},
            { case checkB: SNDArrayValue =>
              val tempRegionForCreation = cb.newLocal[Region](
                "ndarray_add_multily_agg_temp_region",
                Region.stagedCreate(Region.REGULAR, cb.emb.ecb.pool()),
              )
              val NDArrayA =
                LinalgCodeUtils.checkColMajorAndCopyIfNeeded(checkA, cb, tempRegionForCreation)
              val NDArrayB =
                LinalgCodeUtils.checkColMajorAndCopyIfNeeded(checkB, cb, tempRegionForCreation)
              val statePV = state.storageType.loadCheapSCode(cb, state.off).asBaseStruct
              statePV.loadField(cb, ndarrayFieldNumber).consume(
                cb, {
                  cb += state.region.getNewRegion(Region.REGULAR)
                  state.storageType.setFieldPresent(cb, state.off.get, ndarrayFieldNumber)
                  val shape = IndexedSeq(NDArrayA.shapes(0), NDArrayB.shapes(1))
                  val uninitializedNDArray = ndTyp.constructUninitialized(
                    shape,
                    ndTyp.makeColumnMajorStrides(shape, cb),
                    cb,
                    tempRegionForCreation,
                  )
                  state.storeNonmissing(cb, uninitializedNDArray)
                  SNDArray.gemm(cb, "N", "N", NDArrayA, NDArrayB, uninitializedNDArray)
                },
                currentNDPValue =>
                  SNDArray.gemm(
                    cb,
                    "N",
                    "N",
                    const(1.0),
                    NDArrayA,
                    NDArrayB,
                    const(1.0),
                    currentNDPValue.asNDArray,
                  ),
              )
              cb += tempRegionForCreation.clearRegion()
            },
          )
        },
      )
    }
    cb.invokeVoid(seqOpMethod, cb.this_, nextNDArrayACode, nextNDArrayBCode)
  }

  override protected def _combOp(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    region: Value[Region],
    state: TypedRegionBackedAggState,
    other: TypedRegionBackedAggState,
  ): Unit = {
    val combOpMethod = cb.emb.genEmitMethod[Unit]("ndarraymutiply_add_agg_comb_op")

    combOpMethod.voidWithBuilder { cb =>
      val rightPV = other.storageType.loadCheapSCode(cb, other.off).asBaseStruct
      rightPV.loadField(cb, ndarrayFieldNumber).consume(
        cb,
        {},
        { case rightNdValue: SNDArrayValue =>
          val leftPV = state.storageType.loadCheapSCode(cb, state.off).asBaseStruct
          leftPV.loadField(cb, ndarrayFieldNumber).consume(
            cb,
            state.storeNonmissing(cb, rightNdValue),
            { case leftNdValue: SNDArrayValue =>
              NDArraySumAggregator.addValues(cb, state.region, leftNdValue, rightNdValue)
            },
          )
        },
      )
    }
    cb.invokeVoid(combOpMethod, cb.this_)
  }

  override protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region])
    : IEmitCode =
    state.get(cb).map(cb)(sv => sv.copyToRegion(cb, region, sv.st))
}
