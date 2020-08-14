package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.{coerce => _, _}
import is.hail.expr.ir._
import is.hail.types.physical._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
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

  val initContainer: TupleAggregatorState = new TupleAggregatorState(kb, nested, region, new Value[Long]{
    def get: Code[Long] = typ.loadField(off, 0)
  })
  val container: TupleAggregatorState = new TupleAggregatorState(kb, nested, region, statesOffset(idx), regionOffset(idx))

  override def createState(cb: EmitCodeBuilder): Unit = {
    super.createState(cb)
    nested.createStates(cb)
  }

  override def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long]): Code[Unit] = {
    Code(super.load(regionLoader, src),
      off.ceq(0L).mux(Code._empty,
        lenRef := typ.isFieldMissing(off, 1).mux(-1,
          arrayType.loadLength(typ.loadField(off, 1)))))
  }

  def initArray(): Code[Unit] =
    Code(
      region.setNumParents((lenRef + 1) * nStates),
      aoff := arrayType.allocate(region, lenRef),
      Region.storeAddress(typ.fieldOffset(off, 1), aoff),
      arrayType.stagedInitialize(aoff, lenRef),
      typ.setFieldPresent(off, 1))

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
    seq(cb, {cb += initArray()}, container.newState(cb), seqOp)

  def initLength(cb: EmitCodeBuilder, len: Code[Int]): Unit = {
    cb.assign(lenRef, len)
    seq(cb, container.copyFrom(cb, initContainer.off))
  }

  def checkLength(len: Code[Int]): Code[Unit] = {
    lenRef.ceq(len).mux(Code._empty,
      Code._fatal[Unit]("mismatched lengths in ArrayElementsAggregator "))
  }

  def init(cb: EmitCodeBuilder, initOp: => Unit, initLen: Boolean): Unit = {
    cb += region.setNumParents(nStates)
    cb.assign(off, region.allocate(typ.alignment, typ.byteSize))
    initContainer.newState(cb)
    initOp
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
      init(cb, nested.toCodeWithArgs(cb,
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

    init(cb, initContainer.copyFrom(cb, initOffset), initLen = false)
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

  val resultEltType: PTuple = PCanonicalTuple(true, nestedAggs.map(_.resultType): _*)
  val resultType: PArray = PCanonicalArray(resultEltType, required = knownLength)

  val initOpTypes: Seq[PType] = if (knownLength) FastSeq(PInt32(true), PVoid) else FastSeq(PVoid)
  val seqOpTypes: Seq[PType] = FastSeq(PInt32())

  // inits all things
  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    if (knownLength) {
      val Array(len, inits) = init
      state.init(cb, cb += inits.setup, initLen = false)
      cb += len.setup
      state.initLength(cb, len.m.mux(Code._fatal[Int]("Array length can't be missing"), len.value[Int]))
    } else {
      val Array(inits) = init
      state.init(cb, cb += inits.setup, initLen = true)
      cb.assign(state.lenRef, -1)
    }
  }

  // does a length check on arrays
  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    assert(seq.length == 1)
    val len = seq.head
    len.toI(cb).consume(cb, { /* do nothing */ }, { len =>
      if (!knownLength) {
        val v = cb.newLocal("aelca_seqop_len", len.tcode[Int])
        cb.ifx(state.lenRef < 0, state.initLength(cb, v), cb += state.checkLength(v))
      } else {
        cb += state.checkLength(len.tcode[Int])
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
            cb += state.checkLength(other.lenRef)
          })
        } else {
          cb += state.checkLength(other.lenRef)
        }
      })
    }, {
      cb.assign(other.idx, state.idx)
      other.load(cb)
      state.load(cb)
    }, {
      state.nested.toCode(cb, (cb, i, s) => nestedAggs(i).combOp(cb, s, other.nested(i)))
    })
  }

  protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit =
    cb.ifx(state.lenRef < 0, { cb += srvb.setMissing() }, {
      cb += srvb.addArray(resultType, { sab =>
        EmitCodeBuilder.scopedVoid(sab.mb) { cb =>
          cb += sab.start(state.lenRef)
          cb.whileLoop(sab.arrayIdx < state.lenRef, {
            cb += sab.addBaseStruct(resultEltType, { ssb =>
              EmitCodeBuilder.scopedVoid(ssb.mb) { cb =>
                cb += ssb.start()
                cb.assign(state.idx, sab.arrayIdx)
                state.load(cb)
                state.nested.toCode(cb, { (cb, i, s) =>
                  nestedAggs(i).result(cb, s, ssb)
                  cb += ssb.advance()
                })
                state.store(cb)
              }
            })
            cb += sab.advance()
          })
        }
      })
    })
}

class ArrayElementwiseOpAggregator(nestedAggs: Array[StagedAggregator]) extends StagedAggregator {
  type State = ArrayElementState

  val initOpTypes: Seq[PType] = Array[PType]()
  val seqOpTypes: Seq[PType] = Array[PType](PInt32(), PVoid)

  def resultType: PType = PCanonicalArray(PCanonicalTuple(false, nestedAggs.map(_.resultType): _*))

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit =
    throw new UnsupportedOperationException("State must be initialized by ArrayElementLengthCheckAggregator.")

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(eltIdx, seqOps) = seq
    cb += eltIdx.setup
    cb.ifx(!eltIdx.m, {
      cb.assign(state.idx, eltIdx.value[Int])
      cb.ifx(state.idx > state.lenRef || state.idx < 0, {
        cb._fatal("element idx out of bounds")
      }, {
        state.load(cb)
        cb += seqOps.setup
        state.store(cb)
      })
    })
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit =
    throw new UnsupportedOperationException("State must be combined by ArrayElementLengthCheckAggregator.")

  protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit =
    throw new UnsupportedOperationException("Result must be defined by ArrayElementLengthCheckAggregator.")
}
