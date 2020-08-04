package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.functions.StringFunctions
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitRegion, coerce}
import is.hail.types.physical.{PBooleanRequired, PCanonicalTuple, PCode, PNDArray, PNDArrayCode, PNDArrayValue, PNumeric, PType}
import is.hail.utils._

class NDArraySumAggregator (ndTyp: PNDArray) extends StagedAggregator {
  override type State = TypedRegionBackedAggState

  // State needs:
  // 1. Initialized or not
  // 2. The ndarray itself.

  override def resultType: PType = ndTyp

  val stateType = PCanonicalTuple(true, ndTyp)

  override def initOpTypes: Seq[PType] = Array[PType]()

  override def seqOpTypes: Seq[PType] = Array(ndTyp)

  val ndarrayFieldNumber = 0

  def isInitialized(state: State): Code[Boolean] = {
    stateType.isFieldDefined(state.off, ndarrayFieldNumber)
  }

  def ndArrayPointer(state: State): Code[Long] = {
    stateType.loadField(state.off, ndarrayFieldNumber)
  }

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    cb.append(stateType.setFieldMissing(state.off, ndarrayFieldNumber))
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(nextNDCode) = seq
    nextNDCode.toI(cb).consume(cb, {}, {case nextNDPCode: PNDArrayCode =>
      val nextNDPValue = nextNDPCode.memoize(cb, "ndarray_sum_seqop_next")
      cb.ifx(isInitialized(state),
        {
          val currentNDPValue = PCode(ndTyp, ndArrayPointer(state)).asNDArray.memoize(cb, "ndarray_sum_seqop_current")
          addValues(cb, currentNDPValue, nextNDPValue)
        },
        {
          cb.append(state.region.getNewRegion(Region.TINY))
          cb.append(stateType.setFieldPresent(state.off, ndarrayFieldNumber))
          cb.append(ndTyp.constructAtAddress(
            cb.emb,
            stateType.fieldOffset(state.off, ndarrayFieldNumber),
            state.region,
            nextNDCode.pt,
            nextNDPValue.get.tcode[Long],
            true)
          )
        }
      )
    })

  }

  override protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    cb.ifx(!isInitialized(other), {
      // Do nothing
    },
    {
      cb.ifx(!isInitialized(state), {
        cb.append(state.storeNonmissing(other.off))
      },
      {
        val leftValue = PCode(stateType, state.off).asBaseStruct.memoize(cb, "left_state_ndarray_sum_agg")
        val rightValue = PCode(stateType, other.off).asBaseStruct.memoize(cb, "right_state_ndarray_sum_agg")
        leftValue.loadField(cb, ndarrayFieldNumber).consume(cb, {}, { case leftNdCode: PNDArrayCode =>
          val leftNdValue = leftNdCode.memoize(cb, "left_ndarray_sum_agg")
          rightValue.loadField(cb, ndarrayFieldNumber).consume(cb, {}, { case rightNdCode: PNDArrayCode =>
            val rightNdValue = rightNdCode.memoize(cb, "right_ndarray_sum_agg")
            addValues(cb, leftNdValue, rightNdValue)
          })
        })
      })
    })
  }

  private def addValues(cb: EmitCodeBuilder, leftNdValue: PNDArrayValue, rightNdValue: PNDArrayValue): Unit = {
    val sameShape = leftNdValue.sameShape(rightNdValue, cb.emb)

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

  override protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit = {
    val t = state.get()
    cb.append(t.setup)
    cb.append(
      isInitialized(state).mux(
        srvb.addWithDeepCopy(resultType, ndArrayPointer(state)),
        srvb.setMissing()
    ))
  }
}
