package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s.{coerce => _, _}
import is.hail.expr.ir._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.types.physical._
import is.hail.types.virtual.{TInt32, TVoid, Type}
import is.hail.utils._

// initOp args: initOps for nestedAgg, length if knownLength = true
// seqOp args: array, other non-elt args for nestedAgg

class ArrayElementState(val kb: EmitClassBuilder[_], val nested: StateTuple) extends PointerBasedRVAState {
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

  val initContainer: TupleAggregatorState = new TupleAggregatorState(kb, nested, region, new Value[Long] {
    def get: Code[Long] = typ.loadField(off, 0)
  })
  val container: TupleAggregatorState = new TupleAggregatorState(kb, nested, region, statesOffset(idx), regionOffset(idx))

  override def createState(cb: EmitCodeBuilder): Unit = {
    super.createState(cb)
    nested.createStates(cb)
  }

  override def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, srcc: Code[Long]): Unit = {
    super.load(cb, regionLoader, srcc)
    cb.ifx(off.cne(0L),
      {
        cb.assign(lenRef, typ.isFieldMissing(off, 1).mux(-1,
          arrayType.loadLength(typ.loadField(off, 1))))
      })
  }

  def initArray(cb: EmitCodeBuilder): Unit = {
    cb += region.setNumParents((lenRef + 1) * nStates)
    cb.assign(aoff, arrayType.allocate(region, lenRef))
    cb += Region.storeAddress(typ.fieldOffset(off, 1), aoff)
    cb += arrayType.stagedInitialize(aoff, lenRef)
    cb += typ.setFieldPresent(off, 1)
  }

  def seq(cb: EmitCodeBuilder, init: => Unit, initPerElt: => Unit, seqOp: => Unit): Unit = {
    init
    cb.assign(idx, 0)
    cb.whileLoop(idx < lenRef, {
      initPerElt
      seqOp
      store(cb)
      cb.assign(idx, idx + 1)
    })
  }


  def seq(cb: EmitCodeBuilder, seqOp: => Unit): Unit =
    seq(cb, {
      initArray(cb)
    }, container.newState(cb), seqOp)

  def initLength(cb: EmitCodeBuilder, len: Code[Int]): Unit = {
    cb.assign(lenRef, len)
    seq(cb, container.copyFrom(cb, initContainer.off))
  }

  def checkLength(cb: EmitCodeBuilder, len: Code[Int]): Unit = {
    cb.ifx(lenRef.cne(len), cb += Code._fatal[Unit]("mismatched lengths in ArrayElementsAggregator"))
  }

  def init(cb: EmitCodeBuilder, initOp: (EmitCodeBuilder) => Unit, initLen: Boolean): Unit = {
    cb += region.setNumParents(nStates)
    cb.assign(off, region.allocate(typ.alignment, typ.byteSize))
    initContainer.newState(cb)
    initOp(cb)
    initContainer.store(cb)
    if (initLen) {
      cb += typ.setFieldMissing(off, 1)
    }
  }

  def loadInit(cb: EmitCodeBuilder): Unit = initContainer.load(cb)

  def load(cb: EmitCodeBuilder): Unit = container.load(cb)

  def store(cb: EmitCodeBuilder): Unit = container.store(cb)

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    val serializers = nested.states.map(_.serialize(codec));
    { (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) =>
      loadInit(cb)
      nested.toCodeWithArgs(cb,
        FastIndexedSeq(ob),
        { (cb, i, _, args) =>
          val ob = cb.newLocal("aelca_ser_init_ob", coerce[OutputBuffer](args.head))
          serializers(i)(cb, ob)
        })
      cb += ob.writeInt(lenRef)
      cb.assign(idx, 0)
      cb.whileLoop(idx < lenRef, {
        load(cb)
        nested.toCodeWithArgs(cb,
          FastIndexedSeq(ob),
          { case (cb, i, _, args) =>
            val ob = cb.newLocal("aelca_ser_ob", coerce[OutputBuffer](args.head))
            serializers(i)(cb, ob)
          })
        cb.assign(idx, idx + 1)
      })
    }
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    val deserializers = nested.states.map(_.deserialize(codec));
    { (cb: EmitCodeBuilder, ib: Value[InputBuffer]) =>
      init(cb, cb => nested.toCodeWithArgs(cb,
        FastIndexedSeq(ib),
        { (cb, i, _, args) =>
          val ib = cb.newLocal("aelca_deser_init_ib", coerce[InputBuffer](args.head))
          deserializers(i)(cb, ib)
        }),
        initLen = false)
      cb.assign(lenRef, ib.readInt())
      cb.ifx(lenRef < 0, {
        cb += typ.setFieldMissing(off, 1)
      }, {
        seq(cb, {
          nested.toCodeWithArgs(cb,
            FastIndexedSeq(ib),
            { (cb, i, _, args) =>
              val ib = cb.newLocal("aelca_deser_ib", coerce[InputBuffer](args.head))
              deserializers(i)(cb, ib)
            })
        })
      })
    }
  }

  def copyFromAddress(cb: EmitCodeBuilder, src: Code[Long]): Unit = {
    val srcOff = cb.newField("aelca_copyfromaddr_srcoff", src)
    val initOffset = typ.loadField(srcOff, 0)
    val eltOffset = arrayType.loadElement(typ.loadField(srcOff, 1), idx)

    init(cb, cb => initContainer.copyFrom(cb, initOffset), initLen = false)
    cb.ifx(typ.isFieldMissing(srcOff, 1), {
      cb += typ.setFieldMissing(off, 1)
      cb.assign(lenRef, -1)
    }, {
      cb.assign(lenRef, arrayType.loadLength(typ.loadField(srcOff, 1)))
      seq(cb, container.copyFrom(cb, eltOffset))
    })
  }
}

class ArrayElementLengthCheckAggregator(nestedAggs: Array[StagedAggregator], knownLength: Boolean) extends StagedAggregator {
  type State = ArrayElementState

  val resultEltType: PCanonicalTuple = PCanonicalTuple(true, nestedAggs.map(_.resultType): _*)
  val resultType: PCanonicalArray = PCanonicalArray(resultEltType, required = knownLength)

  val initOpTypes: Seq[Type] = if (knownLength) FastSeq(TInt32, TVoid) else FastSeq(TVoid)
  val seqOpTypes: Seq[Type] = FastSeq(TInt32)

  // inits all things
  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    if (knownLength) {
      val Array(len, inits) = init
      state.init(cb, cb => cb += inits.asVoid, initLen = false)
      len.toI(cb).consume(cb, cb._fatal("Array length can't be missing"),
        len => state.initLength(cb, len.asInt32.intCode(cb)))
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
    len.toI(cb).consume(cb, {
      /* do nothing */
    }, { len =>
      if (!knownLength) {
        val v = cb.newLocal("aelca_seqop_len", len.asInt.intCode(cb))
        cb.ifx(state.lenRef < 0, state.initLength(cb, v), state.checkLength(cb, v))
      } else {
        state.checkLength(cb, len.asInt.intCode(cb))
      }
    })
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    state.seq(cb, {
      cb.ifx(other.lenRef < 0, {
        cb.ifx(state.lenRef >= 0, {
          other.initLength(cb, state.lenRef)
        })
      }, {
        if (!knownLength) {
          cb.ifx(state.lenRef < 0, {
            state.initLength(cb, other.lenRef)
          }, {
            state.checkLength(cb, other.lenRef)
          })
        } else {
          state.checkLength(cb, other.lenRef)
        }
      })
    }, {
      cb.assign(other.idx, state.idx)
      other.load(cb)
      state.load(cb)
    }, {
      state.nested.toCode((i, s) => nestedAggs(i).combOp(cb, s, other.nested(i)))
    })
  }

  protected def _storeResult(cb: EmitCodeBuilder, state: State, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit = {
    assert(pt == resultType)

    val len = state.lenRef
    cb.ifx(len < 0,
      ifMissing(cb),
      {
        val resultAddr = cb.newLocal[Long]("arrayagg_result_addr", resultType.allocate(region, len))
        cb += resultType.stagedInitialize(resultAddr, len, setMissing = false)
        val i = cb.newLocal[Int]("arrayagg_result_i", 0)

        cb.whileLoop(i < len, {
          val addrAtI = cb.newLocal[Long]("arrayagg_result_addr_at_i", resultType.elementOffset(resultAddr, len, i))
          cb += resultEltType.stagedInitialize(addrAtI, setMissing = false)
          cb.assign(state.idx, i)
          state.load(cb)
          state.nested.toCode { case (nestedIdx, nestedState) =>
            val nestedAddr = cb.newLocal[Long](s"arrayagg_result_nested_addr_$nestedIdx", resultEltType.fieldOffset(addrAtI, nestedIdx))
            nestedAggs(nestedIdx).storeResult(cb, nestedState, resultEltType.types(nestedIdx), nestedAddr, region,
              (cb: EmitCodeBuilder) => cb += resultEltType.setFieldMissing(addrAtI, nestedIdx))
          }
          state.store(cb)
          cb.assign(i, i + 1)
        })
        // don't need to deep copy because that's done in nested aggregators
        pt.storeAtAddress(cb, addr, region, resultType.loadCheapPCode(cb, resultAddr), deepCopy = false)

      }
    )
  }
}

class ArrayElementwiseOpAggregator(nestedAggs: Array[StagedAggregator]) extends StagedAggregator {
  type State = ArrayElementState

  val initOpTypes: Seq[Type] = Array[Type]()
  val seqOpTypes: Seq[Type] = Array[Type](TInt32, TVoid)

  def resultType: PType = PCanonicalArray(PCanonicalTuple(false, nestedAggs.map(_.resultType): _*))

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit =
    throw new UnsupportedOperationException("State must be initialized by ArrayElementLengthCheckAggregator.")

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(eltIdx, seqOps) = seq
    eltIdx.toI(cb).consume(cb, {}, { idx =>
      cb.assign(state.idx, idx.asInt32.intCode(cb))
      cb.ifx(state.idx > state.lenRef || state.idx < 0, {
        cb._fatal("element idx out of bounds")
      }, {
        state.load(cb)
        cb += seqOps.asVoid
        state.store(cb)
      })
    })
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit =
    throw new UnsupportedOperationException("State must be combined by ArrayElementLengthCheckAggregator.")

  protected def _storeResult(cb: EmitCodeBuilder, state: State, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit =
    throw new UnsupportedOperationException("Result must be defined by ArrayElementLengthCheckAggregator.")
}
