package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{CodeParamType, EmitCode, EmitCodeBuilder, EmitParamType, coerce}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical.{PCanonicalNDArray, PNDArrayCode, PNDArrayValue, PNumeric, PType}
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
    val seqOpMethod = cb.emb.genEmitMethod("ndarray_sum_aggregator_seq_op", FastIndexedSeq(EmitParamType(nextNDCode.pt)), CodeParamType(UnitInfo))

    seqOpMethod.voidWithBuilder { cb =>
      val nextNDInput = seqOpMethod.getEmitParam(1)
      nextNDInput.toI(cb).consume(cb, {}, { case nextNDArrayPCode: PNDArrayCode =>
        val nextNDPV = nextNDArrayPCode.memoize(cb, "ndarray_sum_seqop_next")
        val statePV = state.storageType.loadCheapPCode(cb, state.off).asBaseStruct.memoize(cb, "ndarray_sum_seq_op_state")
        statePV.loadField(cb, ndarrayFieldNumber).consume(cb,
          {
            cb += (state.region.getNewRegion(Region.TINY))
            cb += state.storageType.setFieldPresent(state.off, ndarrayFieldNumber)
            state.storeNonmissing(cb, nextNDPV)
          },
          { currentNDPCode =>
            val currentNDPValue = currentNDPCode.asNDArray.memoize(cb, "ndarray_sum_seqop_current")
            addValues(cb, currentNDPValue, nextNDPV)
          }
        )
      })
    }
    cb.invokeVoid(seqOpMethod, nextNDCode)
  }

  override protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    val combOpMethod = cb.emb.genEmitMethod[Unit]("ndarray_sum_aggregator_comb_op")

    combOpMethod.voidWithBuilder { cb =>
      val rightPV = other.storageType.loadCheapPCode(cb, other.off).asBaseStruct.memoize(cb, "ndarray_sum_comb_op_right")
      rightPV.loadField(cb, ndarrayFieldNumber).consume(cb, {},
        { rightNDPC =>
          val rightNdValue = rightNDPC.asNDArray.memoize(cb, "right_ndarray_sum_agg")
          val leftPV = state.storageType.loadCheapPCode(cb, state.off).asBaseStruct.memoize(cb, "ndarray_sum_comb_op_left")
          leftPV.loadField(cb, ndarrayFieldNumber).consume(cb,
            {
              state.storeNonmissing(cb, rightNdValue)
            },
            { leftNDPC =>
              val leftNdValue = leftNDPC.asNDArray.memoize(cb, "left_ndarray_sum_agg")
              addValues(cb, leftNdValue, rightNdValue)
            })
        }
      )
    }

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

    cb.ifx(!sameShape,
      cb += Code._fatal[Unit]("Can't sum ndarrays of different shapes."))
    cb += columnMajorLoops
  }

  override protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit = {
    state.get()
      .toI(cb)
      .consume(cb,
        cb += srvb.setMissing(),
        { nda => cb += srvb.addIRIntermediate(nda, deepCopy = true) })
  }
}
