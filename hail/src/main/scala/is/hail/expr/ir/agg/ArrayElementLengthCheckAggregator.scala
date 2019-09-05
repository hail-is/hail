package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.io.{BufferSpec, CodecSpec, CodecSpec2, InputBuffer, OutputBuffer}
import is.hail.utils._

// initOp args: initOps for nestedAgg, length if knownLength = true
// seqOp args: array, other non-elt args for nestedAgg

class ArrayElementState(val fb: EmitFunctionBuilder[_], val nested: Array[AggregatorState]) extends PointerBasedRVAState {
  val container: StateContainer = StateContainer(nested, region)
  val arrayType: PArray = PArray(container.typ)
  private val nStates: Int = nested.length
  override val regionSize: Int = Region.SMALL

  val typ: PTuple = PTuple(container.typ, arrayType)

  val lenRef: ClassFieldRef[Int] = fb.newField[Int]("arrayrva_lenref")
  val idx: ClassFieldRef[Int] = fb.newField[Int]("arrayrva_idx")
  private val aoff: ClassFieldRef[Long] = fb.newField[Long]("arrayrva_aoff")

  private def regionOffset(eltIdx: Code[Int]): Code[Int] = (eltIdx + 1) * nStates

  private val initStatesOffset = typ.loadField(region, off, 0)
  private def initStateOffset(idx: Int): Code[Long] = container.getStateOffset(initStatesOffset, idx)

  private def statesOffset(eltIdx: Code[Int]): Code[Long] = arrayType.loadElement(region, typ.loadField(region, off, 1), eltIdx)

  override def createState: Code[Unit] = Code(
    super.createState,
    container.toCode((_, s) => s.createState))

  override def load(regionLoader: Code[Region] => Code[Unit], src: Code[Long]): Code[Unit] = {
    Code(super.load(regionLoader, src),
      off.ceq(0L).mux(Code._empty,
        lenRef := typ.isFieldMissing(off, 1).mux(-1,
          arrayType.loadLength(region, typ.loadField(region, off, 1)))))
  }

  private val initArray: Code[Unit] =
    Code(
      region.setNumParents((lenRef + 1) * nStates),
      aoff := region.allocate(arrayType.contentsAlignment, arrayType.contentsByteSize(lenRef)),
      region.storeAddress(typ.fieldOffset(off, 1), aoff),
      arrayType.stagedInitialize(aoff, lenRef),
      typ.setFieldPresent(region, off, 1))

  def seq(init: Code[Unit], initPerElt: Code[Unit], seqOp: (Int, AggregatorState) => Code[Unit]): Code[Unit] =
    Code(
      init,
      idx := 0,
      Code.whileLoop(idx < lenRef,
        initPerElt,
        container.toCode(seqOp),
        store(idx),
        idx := idx + 1))

  def seq(seqOp: (Int, AggregatorState) => Code[Unit]): Code[Unit] =
    seq(initArray, container.newStates, seqOp)

  def initLength(len: Code[Int]): Code[Unit] = {
    Code(lenRef := len, seq((i, s) => s.copyFrom(initStateOffset(i))))
  }

  def checkLength(len: Code[Int]): Code[Unit] = {
    lenRef.ceq(len).mux(Code._empty,
      Code._fatal("mismatched lengths in ArrayElementsAggregator "))
  }

  def init(initOp: Code[Unit], initLen: Boolean): Code[Unit] = {
      Code(
        region.setNumParents(nStates),
        off := region.allocate(typ.alignment, typ.byteSize),
        container.newStates,
        initOp,
        container.store(0, initStatesOffset),
        if (initLen) typ.setFieldMissing(off, 1) else Code._empty)
  }

  def loadInit: Code[Unit] =
    container.load(0, initStatesOffset)

  def load(eltIdx: Code[Int]): Code[Unit] =
    container.load(regionOffset(eltIdx), statesOffset(eltIdx))

  def store(eltIdx: Code[Int]): Code[Unit] =
    container.store(regionOffset(eltIdx), statesOffset(eltIdx))

  def serialize(codec: BufferSpec): Code[OutputBuffer] => Code[Unit] = {
    val serializers = nested.map(_.serialize(codec));
    { ob: Code[OutputBuffer] =>
      Code(
        loadInit,
        container.toCode((i, _) => serializers(i)(ob)),
        ob.writeInt(lenRef),
        idx := 0,
        Code.whileLoop(idx < lenRef,
          load(idx),
          container.toCode((i, _) => serializers(i)(ob)),
          idx := idx + 1))
    }
  }

  def deserialize(codec: BufferSpec): Code[InputBuffer] => Code[Unit] = {
    val deserializers = nested.map(_.deserialize(codec));
    { ib: Code[InputBuffer] =>
        Code(
          init(container.toCode((i, _) => deserializers(i)(ib)), initLen = false),
          lenRef := ib.readInt(),
          (lenRef < 0).mux(
            typ.setFieldMissing(off, 1),
            seq((i, _) => deserializers(i)(ib))))
    }
  }

  def copyFromAddress(src: Code[Long]): Code[Unit] = {
    val srcOff = fb.newField[Long]
    val initOffset = typ.loadField(srcOff, 0)
    val eltOffset = arrayType.loadElement(typ.loadField(srcOff, 1), idx)

    Code(
      srcOff := src,
      init(container.toCode((i, s) => s.copyFrom(container.getStateOffset(initOffset, i))), initLen = false),
      typ.isFieldMissing(srcOff, 1).mux(
        Code(typ.setFieldMissing(off, 1),
          lenRef := -1),
        Code(
          lenRef := arrayType.loadLength(typ.loadField(srcOff, 1)),
          seq((i, s) => s.copyFrom(container.getStateOffset(eltOffset, i))))))
  }
}

class ArrayElementLengthCheckAggregator(nestedAggs: Array[StagedAggregator], knownLength: Boolean) extends StagedAggregator {
  type State = ArrayElementState

  val resultEltType: PTuple = PTuple(nestedAggs.map(_.resultType): _*)
  val resultType: PArray = PArray(resultEltType)

  def createState(fb: EmitFunctionBuilder[_]): State = new ArrayElementState(fb, nestedAggs.map(_.createState(fb)))

  // inits all things
  def initOp(state: State, init: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    if (knownLength) {
      val Array(len, inits) = init
      Code(state.init(inits.setup, initLen = false), len.setup,
        state.initLength(len.m.mux(Code._fatal("Array length can't be missing"), len.value[Int])))
    } else {
      val Array(inits) = init
      Code(state.init(inits.setup, initLen = true), state.lenRef := -1)
    }
  }

  //does a length check on arrays
  def seqOp(state: State, seq: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    val Array(len) = seq
    var check = state.checkLength(len.value[Int])
    if (!knownLength)
      check = (state.lenRef < 0).mux(state.initLength(len.value[Int]), check)
    Code(len.setup, len.m.mux(Code._empty, check))
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
      Code(other.load(state.idx), state.load(state.idx)),
      (i, s) => nestedAggs(i).combOp(s, other.nested(i)))
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
                state.load(sab.arrayIdx),
                state.container.toCode { (i, s) =>
                  Code(nestedAggs(i).result(s, ssb), ssb.advance())
                })
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

  def createState(fb: EmitFunctionBuilder[_]): State =
    throw new UnsupportedOperationException(s"State must be created by ArrayElementLengthCheckAggregator")

  def initOp(state: State, init: Array[EmitTriplet], dummy: Boolean): Code[Unit] =
    throw new UnsupportedOperationException("State must be initialized by ArrayElementLengthCheckAggregator.")

  def seqOp(state: State, seq: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    val Array(eltIdx, seqOps) = seq
    val eltIdxV = state.fb.newField[Int]
    Code(
      eltIdx.setup,
      eltIdx.m.mux(
        Code._empty,
        Code(
          eltIdxV := eltIdx.value[Int],
          (eltIdxV > state.lenRef || eltIdxV < 0).mux(
            Code._fatal("element idx out of bounds"),
            Code(
              state.load(eltIdxV),
              seqOps.setup,
              state.store(eltIdxV))))))
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] =
    throw new UnsupportedOperationException("State must be combined by ArrayElementLengthCheckAggregator.")

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] =
    throw new UnsupportedOperationException("Result must be defined by ArrayElementLengthCheckAggregator.")
}