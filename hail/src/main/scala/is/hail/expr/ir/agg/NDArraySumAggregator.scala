package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.functions.StringFunctions
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitRegion, coerce}
import is.hail.types.physical.{PBooleanRequired, PCanonicalTuple, PCode, PNDArray, PNDArrayCode, PNDArrayValue, PNumeric, PType}
import is.hail.utils._

object NDArraySumAggregator {
}

class NDArraySumAggregator (ndTyp: PNDArray) extends StagedAggregator {
  override type State = TypedRegionBackedAggState

  // State needs:
  // 1. Initialized or not
  // 2. The ndarray itself.
  val stateType = PCanonicalTuple(true, PBooleanRequired, ndTyp)

  override def resultType: PType = ndTyp

  override def initOpTypes: Seq[PType] = Array[PType]()

  override def seqOpTypes: Seq[PType] = Array(ndTyp)

  def isInitialized(state: State): Code[Boolean] = {
    Region.loadBoolean(stateType.fieldOffset(state.off, 0))
  }

  def ndArrayPointer(state: State): Code[Long] = {
    stateType.loadField(state.off, 1)
  }

  override protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    cb.append(Code._println("Trying to initOp"))
    cb.append(Region.storeBoolean(stateType.fieldOffset(state.off, 0), false))
  }

  override protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(nextNDCode) = seq
    cb.append(Code._println("Trying to seqop"))
    nextNDCode.toI(cb).consume(cb, {}, {case nextNDPCode: PNDArrayCode =>
      val nextNDPValue = nextNDPCode.memoize(cb, "ndarray_sum_seqop_next")
      cb.ifx(isInitialized(state),
        {
          val currentNDPValue = PCode(ndTyp, ndArrayPointer(state)).asNDArray.memoize(cb, "ndarray_sum_seqop_current")

          addValues(cb, currentNDPValue, nextNDPValue)
        },
        {
          // TODO Is this really safe, does it copy the initial array?
          cb.append(Code._println("Uninitialized, constructingAtAddress"))
          cb.append(state.region.getNewRegion(Region.TINY))
          cb.append(Region.storeBoolean(stateType.fieldOffset(state.off, 0), true))
          cb.append(ndTyp.constructAtAddress(cb.emb, stateType.fieldOffset(state.off, 1), state.region, nextNDCode.pt, nextNDPValue.get.tcode[Long], true))
        }
      )
    })

  }

  override protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    cb.append(Code._println("Trying to combOp"))
    cb.append(Code._println(const("Left value init field looks like: ").concat(Region.loadBoolean(state.off).toS)))
    cb.append(Code._println(const("Right value init field looks like: ").concat(Region.loadBoolean(other.off).toS)))
    cb.ifx(!isInitialized(other), {
      // Do nothing
    },
    {
      cb.ifx(!isInitialized(state), {
        cb.append(Code._println("combOp: state was not initialized, overwriting with other"))
        cb.append(state.storeNonmissing(other.off))
      },
      {
        val leftValue = PCode(stateType, state.off).asBaseStruct.memoize(cb, "left_state_ndarray_sum_agg")
        val rightValue = PCode(stateType, other.off).asBaseStruct.memoize(cb, "right_state_ndarray_sum_agg")
        leftValue.loadField(cb, 1).consume(cb, {}, { case leftNdCode: PNDArrayCode =>
          val leftNdValue = leftNdCode.memoize(cb, "left_ndarray_sum_agg")
          rightValue.loadField(cb, 1).consume(cb, {}, { case rightNdCode: PNDArrayCode =>
            val rightNdValue = rightNdCode.memoize(cb, "right_ndarray_sum_agg")
            cb.append(Code._println("combOp: About to addValues"))
            cb.append(Code._println(StringFunctions.boxArg(EmitRegion(cb.emb, state.region), stateType)(state.off)))
            cb.append(Code._println(StringFunctions.boxArg(EmitRegion(cb.emb, other.region), stateType)(other.off)))
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

    val body = ndTyp.mutateElement(
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
    cb.append(Code._println("Trying to write resultOp"))
    cb.append(Code._println(const("resultOp: State value init field looks like: ").concat(Region.loadBoolean(state.off).toS)))
    val t = state.get()
    assert(resultType == state.typ.asInstanceOf[PCanonicalTuple].types(1))
    cb.append(t.setup)
    cb.append(
      isInitialized(state).mux(
        Code(
          Code._println(StringFunctions.boxArg(EmitRegion(cb.emb, state.region), stateType)(state.off)),
          srvb.addWithDeepCopy(resultType, ndArrayPointer(state)),
          Code._println("Updated SRVB")
        ),
        Code(
          Code._println("Final result was missing"),
          srvb.setMissing()
        )
    ))
  }
}
