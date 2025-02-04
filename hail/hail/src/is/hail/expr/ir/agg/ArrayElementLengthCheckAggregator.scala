package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s.{coerce => _, _}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.types.physical._
import is.hail.types.physical.stypes.{EmitType, SValue}
import is.hail.types.physical.stypes.concrete.SIndexablePointer
import is.hail.types.virtual.{TInt32, TVoid, Type}
import is.hail.utils._

// initOp args: initOps for nestedAgg, length if knownLength = true
// seqOp args: array, other non-elt args for nestedAgg

class ArrayElementState(val kb: EmitClassBuilder[_], val nested: StateTuple)
    extends PointerBasedRVAState {
  val arrayType: PArray = PCanonicalArray(nested.storageType)
  private val nStates: Int = nested.nStates
  override val regionSize: Int = Region.SMALL

  val typ: PTuple = PCanonicalTuple(true, nested.storageType, arrayType)

  val lenRef: Settable[Int] = kb.genFieldThisRef[Int]("arrayrva_lenref")
  val idx: Settable[Int] = kb.genFieldThisRef[Int]("arrayrva_idx")
  private val aoff: Settable[Long] = kb.genFieldThisRef[Long]("arrayrva_aoff")

  private def regionOffset(eltIdx: Value[Int]): Value[Int] = new Value[Int] {
    def get: Code[Int] = (eltIdx + 1) * nStates
  }

  private def statesOffset(eltIdx: Value[Int]): Value[Long] = new Value[Long] {
    def get: Code[Long] = arrayType.loadElement(typ.loadField(off, 1), eltIdx)
  }

  val initContainer: TupleAggregatorState = new TupleAggregatorState(
    kb,
    nested,
    region,
    new Value[Long] {
      def get: Code[Long] = typ.loadField(off, 0)
    },
  )

  val container: TupleAggregatorState =
    new TupleAggregatorState(kb, nested, region, statesOffset(idx), regionOffset(idx))

  override def createState(cb: EmitCodeBuilder): Unit = {
    super.createState(cb)
    nested.createStates(cb)
  }

  override def load(
    cb: EmitCodeBuilder,
    regionLoader: (EmitCodeBuilder, Value[Region]) => Unit,
    src: Value[Long],
  ): Unit = {
    super.load(cb, regionLoader, src)
    cb.if_(
      off.cne(0L),
      cb.assign(
        lenRef,
        typ.isFieldMissing(cb, off, 1).mux(-1, arrayType.loadLength(typ.loadField(off, 1))),
      ),
    )
  }

  def initArray(cb: EmitCodeBuilder): Unit = {
    cb += region.setNumParents((lenRef + 1) * nStates)
    cb.assign(aoff, arrayType.allocate(region, lenRef))
    cb += Region.storeAddress(typ.fieldOffset(off, 1), aoff)
    arrayType.stagedInitialize(cb, aoff, lenRef)
    typ.setFieldPresent(cb, off, 1)
  }

  def seq(cb: EmitCodeBuilder, init: => Unit, initPerElt: => Unit, seqOp: => Unit): Unit = {
    init
    cb.assign(idx, 0)
    cb.while_(
      idx < lenRef, {
        initPerElt
        seqOp
        store(cb)
        cb.assign(idx, idx + 1)
      },
    )
  }

  def seq(cb: EmitCodeBuilder, seqOp: => Unit): Unit =
    seq(cb, initArray(cb), container.newState(cb), seqOp)

  def initLength(cb: EmitCodeBuilder, len: Code[Int]): Unit = {
    cb.assign(lenRef, len)
    seq(cb, container.copyFrom(cb, initContainer.off))
  }

  def checkLength(cb: EmitCodeBuilder, len: Code[Int]): Unit =
    cb.if_(
      lenRef.cne(len),
      cb += Code._fatal[Unit]("mismatched lengths in ArrayElementsAggregator"),
    )

  def init(cb: EmitCodeBuilder, initOp: (EmitCodeBuilder) => Unit, initLen: Boolean): Unit = {
    cb += region.setNumParents(nStates)
    cb.assign(off, region.allocate(typ.alignment, typ.byteSize))
    initContainer.newState(cb)
    initOp(cb)
    initContainer.store(cb)
    if (initLen) {
      typ.setFieldMissing(cb, off, 1)
    }
  }

  def loadInit(cb: EmitCodeBuilder): Unit = initContainer.load(cb)

  def load(cb: EmitCodeBuilder): Unit = container.load(cb)

  def store(cb: EmitCodeBuilder): Unit = container.store(cb)

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    val serializers = nested.states.map(_.serialize(codec));
    { (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) =>
      loadInit(cb)
      nested.toCodeWithArgs(cb, (cb, i, _) => serializers(i)(cb, ob))
      cb += ob.writeInt(lenRef)
      cb.assign(idx, 0)
      cb.while_(
        idx < lenRef, {
          load(cb)
          nested.toCodeWithArgs(
            cb,
            { case (cb, i, _) =>
              serializers(i)(cb, ob)
            },
          )
          cb.assign(idx, idx + 1)
        },
      )
    }
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    val deserializers = nested.states.map(_.deserialize(codec));
    { (cb: EmitCodeBuilder, ib: Value[InputBuffer]) =>
      init(
        cb,
        cb => nested.toCodeWithArgs(cb, (cb, i, _) => deserializers(i)(cb, ib)),
        initLen = false,
      )
      cb.assign(lenRef, ib.readInt())
      cb.if_(
        lenRef < 0,
        typ.setFieldMissing(cb, off, 1),
        seq(cb, nested.toCodeWithArgs(cb, (cb, i, _) => deserializers(i)(cb, ib))),
      )
    }
  }

  def copyFromAddress(cb: EmitCodeBuilder, src: Value[Long]): Unit = {
    init(cb, cb => initContainer.copyFrom(cb, cb.memoize(typ.loadField(src, 0))), initLen = false)
    cb.if_(
      typ.isFieldMissing(cb, src, 1), {
        typ.setFieldMissing(cb, off, 1)
        cb.assign(lenRef, -1)
      }, {
        cb.assign(lenRef, arrayType.loadLength(typ.loadField(src, 1)))
        seq(
          cb,
          container.copyFrom(cb, cb.memoize(arrayType.loadElement(typ.loadField(src, 1), idx))),
        )
      },
    )
  }
}

class ArrayElementLengthCheckAggregator(nestedAggs: Array[StagedAggregator], knownLength: Boolean)
    extends StagedAggregator {
  type State = ArrayElementState

  val resultEltType: PCanonicalTuple =
    PCanonicalTuple(true, nestedAggs.map(_.resultEmitType.storageType): _*)

  val resultPType: PCanonicalArray = PCanonicalArray(resultEltType)
  override def resultEmitType = EmitType(SIndexablePointer(resultPType), knownLength)

  val initOpTypes: Seq[Type] = if (knownLength) FastSeq(TInt32, TVoid) else FastSeq(TVoid)
  val seqOpTypes: Seq[Type] = FastSeq(TInt32)

  // inits all things
  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    if (knownLength) {
      val Array(len, inits) = init
      state.init(cb, cb => cb += inits.asVoid, initLen = false)
      len.toI(cb).consume(
        cb,
        cb._fatal("Array length can't be missing"),
        len => state.initLength(cb, len.asInt32.value),
      )
    } else {
      val Array(inits) = init
      state.init(cb, cb => cb += inits.asVoid, initLen = true)
      cb.assign(state.lenRef, -1)
    }
  }

  // does a length check on arrays
  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    assert(seq.length == 1)
    val len = seq.head
    len.toI(cb).consume(
      cb, {
        /* do nothing */
      },
      { len =>
        if (!knownLength) {
          val v = cb.newLocal("aelca_seqop_len", len.asInt.value)
          cb.if_(state.lenRef < 0, state.initLength(cb, v), state.checkLength(cb, v))
        } else {
          state.checkLength(cb, len.asInt.value)
        }
      },
    )
  }

  protected def _combOp(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    region: Value[Region],
    state: ArrayElementState,
    other: ArrayElementState,
  ): Unit = {
    state.seq(
      cb, {
        cb.if_(
          other.lenRef < 0,
          cb.if_(state.lenRef >= 0, other.initLength(cb, state.lenRef)), {
            if (!knownLength) {
              cb.if_(
                state.lenRef < 0,
                state.initLength(cb, other.lenRef),
                state.checkLength(cb, other.lenRef),
              )
            } else {
              state.checkLength(cb, other.lenRef)
            }
          },
        )
      }, {
        cb.assign(other.idx, state.idx)
        other.load(cb)
        state.load(cb)
      },
      state.nested.toCode((i, s) => nestedAggs(i).combOp(ctx, cb, region, s, other.nested(i))),
    )
  }

  override protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region])
    : IEmitCode = {
    val len = state.lenRef

    def resultBody(cb: EmitCodeBuilder): SValue = {
      val resultAddr = cb.newLocal[Long]("arrayagg_result_addr", resultPType.allocate(region, len))
      resultPType.stagedInitialize(cb, resultAddr, len, setMissing = false)
      val i = cb.newLocal[Int]("arrayagg_result_i", 0)

      cb.while_(
        i < len, {
          val addrAtI = cb.newLocal[Long](
            "arrayagg_result_addr_at_i",
            resultPType.elementOffset(resultAddr, len, i),
          )
          resultEltType.stagedInitialize(cb, addrAtI, setMissing = false)
          cb.assign(state.idx, i)
          state.load(cb)
          state.nested.toCode { case (nestedIdx, nestedState) =>
            val nestedAddr = cb.newLocal[Long](
              s"arrayagg_result_nested_addr_$nestedIdx",
              resultEltType.fieldOffset(addrAtI, nestedIdx),
            )
            val nestedRes = nestedAggs(nestedIdx).result(cb, nestedState, region)
            nestedRes.consume(
              cb,
              resultEltType.setFieldMissing(cb, addrAtI, nestedIdx),
              sv => resultEltType.types(nestedIdx).storeAtAddress(cb, nestedAddr, region, sv, true),
            )
          }
          state.store(cb)
          cb.assign(i, i + 1)
        },
      )
      // don't need to deep copy because that's done in nested aggregators
      resultPType.loadCheapSCode(cb, resultAddr)
    }

    if (knownLength) {
      IEmitCode.present(cb, resultBody(cb))
    } else {
      IEmitCode(cb, len < 0, resultBody(cb))
    }
  }
}

class ArrayElementwiseOpAggregator(nestedAggs: Array[StagedAggregator]) extends StagedAggregator {
  type State = ArrayElementState

  val initOpTypes: Seq[Type] = Array[Type]()
  val seqOpTypes: Seq[Type] = Array[Type](TInt32, TVoid)

  val resultPType =
    PCanonicalArray(PCanonicalTuple(false, nestedAggs.map(_.resultEmitType.storageType): _*))

  override def resultEmitType = EmitType(SIndexablePointer(resultPType), false)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit =
    throw new UnsupportedOperationException(
      "State must be initialized by ArrayElementLengthCheckAggregator."
    )

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(eltIdx, seqOps) = seq
    eltIdx.toI(cb).consume(
      cb,
      {},
      { idx =>
        cb.assign(state.idx, idx.asInt32.value)
        cb.if_(
          state.idx > state.lenRef || state.idx < 0,
          cb._fatal("element idx out of bounds"), {
            state.load(cb)
            cb += seqOps.asVoid
            state.store(cb)
          },
        )
      },
    )
  }

  protected def _combOp(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    region: Value[Region],
    state: ArrayElementState,
    other: ArrayElementState,
  ): Unit =
    throw new UnsupportedOperationException(
      "State must be combined by ArrayElementLengthCheckAggregator."
    )

  protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region]): IEmitCode =
    throw new UnsupportedOperationException(
      "Result must be defined by ArrayElementLengthCheckAggregator."
    )
}
