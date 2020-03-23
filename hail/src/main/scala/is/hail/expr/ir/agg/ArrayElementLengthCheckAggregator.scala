package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.{coerce => _, _} // use ir coerce
import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.utils._

// initOp args: initOps for nestedAgg, length if knownLength = true
// seqOp args: array, other non-elt args for nestedAgg

class ArrayElementState(val cb: EmitClassBuilder[_], val nested: StateTuple) extends PointerBasedRVAState {
  val arrayType: PArray = PArray(nested.storageType)
  private val nStates: Int = nested.nStates
  override val regionSize: Int = Region.SMALL

  val typ: PTuple = PTuple(nested.storageType, arrayType)

  val lenRef: Settable[Int] = cb.genFieldThisRef[Int]("arrayrva_lenref")
  val idx: Settable[Int] = cb.genFieldThisRef[Int]("arrayrva_idx")
  private val aoff: Settable[Long] = cb.genFieldThisRef[Long]("arrayrva_aoff")

  private def regionOffset(eltIdx: Value[Int]): Value[Int] = new Value[Int] {
    def get: Code[Int] = (eltIdx + 1) * nStates
  }

  private def statesOffset(eltIdx: Value[Int]): Value[Long] = new Value[Long] {
    def get: Code[Long] = arrayType.loadElement(typ.loadField(off, 1), eltIdx)
  }



  val initContainer: TupleAggregatorState = new TupleAggregatorState(cb, nested, region, new Value[Long]{
    def get: Code[Long] = typ.loadField(off, 0)
  })
  val container: TupleAggregatorState = new TupleAggregatorState(cb, nested, region, statesOffset(idx), regionOffset(idx))

  override def createState: Code[Unit] = Code(
    super.createState, nested.createStates(cb))

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

  def seq(init: Code[Unit], initPerElt: Code[Unit], seqOp: Code[Unit]): Code[Unit] =
    Code(
      init,
      idx := 0,
      Code.whileLoop(idx < lenRef,
        initPerElt,
        seqOp,
        store,
        idx := idx + 1))

  def seq(seqOp: Code[Unit]): Code[Unit] =
    seq(initArray(), container.newState, seqOp)

  def initLength(len: Code[Int]): Code[Unit] = {
    Code(lenRef := len, seq(container.copyFrom(initContainer.off)))
  }

  def checkLength(len: Code[Int]): Code[Unit] = {
    lenRef.ceq(len).mux(Code._empty,
      Code._fatal[Unit]("mismatched lengths in ArrayElementsAggregator "))
  }

  def init(initOp: Code[Unit], initLen: Boolean): Code[Unit] = {
    Code(
      region.setNumParents(nStates),
      off := region.allocate(typ.alignment, typ.byteSize),
      initContainer.newState,
      initOp,
      initContainer.store,
      if (initLen) typ.setFieldMissing(off, 1) else Code._empty)
  }

  def loadInit: Code[Unit] = initContainer.load

  def load: Code[Unit] =
    container.load

  def store: Code[Unit] =
    container.store

  def serialize(codec: BufferSpec): Value[OutputBuffer] => Code[Unit] = {
    val serializers = nested.states.map(_.serialize(codec));
    { ob: Value[OutputBuffer] =>
      Code(
        loadInit,
        nested.toCodeWithArgs(cb, "array_nested_serialize_init", Array[TypeInfo[_]](classInfo[OutputBuffer]),
          FastIndexedSeq(ob),
          { (i, _, args) =>
            Code.memoize(coerce[OutputBuffer](args.head), "aelca_ser_init_ob") { ob => serializers(i)(ob) }
          }),
        ob.writeInt(lenRef),
        idx := 0,
        Code.whileLoop(idx < lenRef,
          load,
          nested.toCodeWithArgs(cb, "array_nested_serialize", Array[TypeInfo[_]](classInfo[OutputBuffer]),
            FastIndexedSeq(ob),
            { case (i, _, args) =>
              Code.memoize(coerce[OutputBuffer](args.head), "aelca_ser_ob") { ob => serializers(i)(ob) }
            }),
          idx := idx + 1))
    }
  }

  def deserialize(codec: BufferSpec): Value[InputBuffer] => Code[Unit] = {
    val deserializers = nested.states.map(_.deserialize(codec));
    { ib: Value[InputBuffer] =>
      Code(
        init(nested.toCodeWithArgs(cb, "array_nested_deserialize_init", Array[TypeInfo[_]](classInfo[InputBuffer]),
          FastIndexedSeq(ib),
          { (i, _, args) =>
            Code.memoize(coerce[InputBuffer](args.head), "aelca_deser_init_ib") { ib =>
              deserializers(i)(ib)
            }
          }),
          initLen = false),
        lenRef := ib.readInt(),
        (lenRef < 0).mux(
          typ.setFieldMissing(off, 1),
          seq(nested.toCodeWithArgs(cb, "array_nested_deserialize", Array[TypeInfo[_]](classInfo[InputBuffer]),
            FastIndexedSeq(ib),
            { (i, _, args) =>
              Code.memoize(coerce[InputBuffer](args.head), "aelca_deser_ib") { ib =>
                deserializers(i)(ib)
              }
            }))))
    }
  }

  def copyFromAddress(src: Code[Long]): Code[Unit] = {
    val srcOff = cb.genFieldThisRef[Long]()
    val initOffset = typ.loadField(srcOff, 0)
    val eltOffset = arrayType.loadElement(typ.loadField(srcOff, 1), idx)

    Code(
      srcOff := src,
      init(initContainer.copyFrom(initOffset), initLen = false),
      typ.isFieldMissing(srcOff, 1).mux(
        Code(typ.setFieldMissing(off, 1),
          lenRef := -1),
        Code(
          lenRef := arrayType.loadLength(typ.loadField(srcOff, 1)),
          seq(container.copyFrom(eltOffset)))))
  }
}

class ArrayElementLengthCheckAggregator(nestedAggs: Array[StagedAggregator], knownLength: Boolean) extends StagedAggregator {
  type State = ArrayElementState

  val resultEltType: PTuple = PTuple(nestedAggs.map(_.resultType): _*)
  val resultType: PArray = PArray(resultEltType)

  def createState(cb: EmitClassBuilder[_]): State = new ArrayElementState(cb, StateTuple(nestedAggs.map(_.createState(cb))))

  // inits all things
  def initOp(state: State, init: Array[EmitCode], dummy: Boolean): Code[Unit] = {
    if (knownLength) {
      val Array(len, inits) = init
      Code(state.init(inits.setup, initLen = false), len.setup,
        state.initLength(len.m.mux(Code._fatal[Int]("Array length can't be missing"), len.value[Int])))
    } else {
      val Array(inits) = init
      Code(state.init(inits.setup, initLen = true), state.lenRef := -1)
    }
  }

  // does a length check on arrays
  def seqOp(state: State, seq: Array[EmitCode], dummy: Boolean): Code[Unit] = {
    assert(seq.length == 1)
    val len = seq.head
    Code(
      len.setup,
      len.m.mux(
        Code._empty,
        Code.memoize(len.value[Int], "aelca_seqop_len") { v =>
          var check = state.checkLength(v)
          if (!knownLength)
            check = (state.lenRef < 0).mux(state.initLength(v), check)
          check
        }))
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    var check = state.checkLength(other.lenRef)
    if (!knownLength)
      check = (state.lenRef < 0).mux(state.initLength(other.lenRef), check)

    state.seq((other.lenRef < 0).mux(
      (state.lenRef < 0).mux(
        Code._empty,
        other.initLength(state.lenRef)),
      check),
      Code(other.idx := state.idx, other.load, state.load),
      state.nested.toCode(state.cb, "array_nested_comb", (i, s) => nestedAggs(i).combOp(s, other.nested(i))))
  }

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] =
    (state.lenRef < 0).mux(
      srvb.setMissing(),
      srvb.addArray(resultType, { sab =>
        Code(
          sab.start(state.lenRef),
          Code.whileLoop(sab.arrayIdx < state.lenRef,
            sab.addBaseStruct(resultEltType, { ssb =>
              Code(
                ssb.start(),
                state.idx := sab.arrayIdx,
                state.load,
                state.nested.toCode(state.cb, "array_nested_result", { (i, s) =>
                  Code(nestedAggs(i).result(s, ssb), ssb.advance())
                }))
            }),
            sab.advance()))
      })
    )
}

class ArrayElementwiseOpAggregator(nestedAggs: Array[StagedAggregator]) extends StagedAggregator {
  type State = ArrayElementState

  def initOpTypes: Array[PType] = Array()

  def seqOpTypes: Array[PType] = Array(PInt32(), PVoid)

  def resultType: PType = PArray(PTuple(nestedAggs.map(_.resultType): _*))

  def createState(cb: EmitClassBuilder[_]): State =
    throw new UnsupportedOperationException(s"State must be created by ArrayElementLengthCheckAggregator")

  def initOp(state: State, init: Array[EmitCode], dummy: Boolean): Code[Unit] =
    throw new UnsupportedOperationException("State must be initialized by ArrayElementLengthCheckAggregator.")

  def seqOp(state: State, seq: Array[EmitCode], dummy: Boolean): Code[Unit] = {
    val Array(eltIdx, seqOps) = seq
    Code(
      eltIdx.setup,
      eltIdx.m.mux(
        Code._empty,
        Code(
          state.idx := eltIdx.value[Int],
          (state.idx > state.lenRef || state.idx < 0).mux(
            Code._fatal[Unit]("element idx out of bounds"),
            Code(
              state.load,
              seqOps.setup,
              state.store)))))
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] =
    throw new UnsupportedOperationException("State must be combined by ArrayElementLengthCheckAggregator.")

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] =
    throw new UnsupportedOperationException("Result must be defined by ArrayElementLengthCheckAggregator.")
}