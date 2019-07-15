package is.hail.expr.ir.agg

import is.hail.annotations.{Region, RegionUtils, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.io.{CodecSpec, InputBuffer, OutputBuffer}
import is.hail.utils._
import is.hail.asm4s.coerce

// initOp args: initOps for nestedAgg, length if knownLength = true
// seqOp args: array, other non-elt args for nestedAgg

object ArrayElementState {
  def create(mb: EmitMethodBuilder, aggs: Array[StagedRegionValueAggregator], knownLength: Boolean): ArrayElementState =
    ArrayElementState(mb, aggs.map(_.createState(mb)), knownLength)
}

case class ArrayElementState(mb: EmitMethodBuilder, nested: Array[AggregatorState], knownLength: Boolean) extends PointerBasedRVAState {
  val container: StateContainer = StateContainer(nested, region)
  val arrayType: PArray = PArray(container.typ)
  private val nStates: Int = nested.length
  override val regionSize: Int = Region.small

  val typ: PTuple = PTuple(FastIndexedSeq(container.typ, arrayType))

  val lenRef: ClassFieldRef[Int] = mb.newField[Int]("arrayrva_lenref")
  val idx: ClassFieldRef[Int] = mb.newField[Int]("arrayrva_idx")

  val srvb = new StagedRegionValueBuilder(er, typ)

  private def regionOffset(eltIdx: Code[Int]): Code[Int] = (eltIdx + 1) * nStates

  private val initStatesOffset = typ.loadField(region, off, 0)
  private def initStateOffset(idx: Int): Code[Long] = container.getStateOffset(initStatesOffset, idx)

  private def statesOffset(eltIdx: Code[Int]): Code[Long] = arrayType.loadElement(region, typ.loadField(region, off, 1), eltIdx)
  private def stateAddressOffset(eltIdx: Code[Int], stateIdx: Int): Code[Long] = container.getStateOffset(statesOffset(eltIdx), stateIdx)

  override def storeRegion(topRegion: Code[Region], rIdx: Code[Int]): Code[Unit] = Code(
    super.storeRegion(topRegion, rIdx),
    container.toCode((i, s) => s.loadRegion(r => r.closeButKeepContainer())))

  override def createState: Code[Unit] = Code(
    super.createState,
    container.toCode((i, s) => s.createState))

  override def loadState(src: Code[Long]): Code[Unit] = Code(
    super.loadState(src),
    off.ceq(0L).mux(Code._empty,
      lenRef := typ.isFieldMissing(off, 1).mux(-1,
        arrayType.loadLength(region, typ.loadField(region, off, 1)))))

  def initLength(len: Code[Int]): Code[Unit] = {
    val srvb2 = new StagedRegionValueBuilder(er, arrayType)
    Code(
      lenRef := len,
      region.setNumParents((lenRef + 1) * nStates),
      srvb2.start(lenRef),
      Code.whileLoop(srvb2.arrayIdx < lenRef,
        container.loadRegions(regionOffset(srvb2.arrayIdx)),
        container.toCode((i, s) => s.copyFrom(initStateOffset(i))),
        container.addState(srvb2),
        srvb2.advance()),
      typ.setFieldPresent(region, off, 1),
      region.storeAddress(typ.fieldOffset(off, 1), srvb2.end()))
  }

  def checkLength(len: Code[Int]): Code[Unit] = {
    val check =
      lenRef.ceq(len).mux(Code._empty,
        Code._fatal("mismatched lengths in ArrayElementsAggregator"))

    if (knownLength) check else (lenRef < 0).mux(initLength(len), check)
  }

  def init(initOp: Array[Code[Unit]]): Code[Unit] = {
      val c = Code(
        region.setNumParents(nStates),
        srvb.start(),
        container.loadRegions(0),
        container.toCode((i, _) => initOp(i)),
        container.addState(srvb),
        srvb.advance())
    if (knownLength)
      Code(c, off := srvb.end())
    else
      Code(c, srvb.setMissing(), off := srvb.end())
  }

  def loadInit: Code[Unit] =
    container.load(0, initStatesOffset)


  def load(eltIdx: Code[Int]): Code[Unit] =
    container.load(regionOffset(eltIdx), statesOffset(eltIdx))

  def store(eltIdx: Code[Int]): Code[Unit] =
    container.store(regionOffset(eltIdx), statesOffset(eltIdx))

  def serialize(codec: CodecSpec): Code[OutputBuffer] => Code[Unit] = {
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

  def unserialize(codec: CodecSpec): Code[InputBuffer] => Code[Unit] = {
    val deserializers = nested.map(_.unserialize(codec));
    { ib: Code[InputBuffer] =>
        Code(
          region.setNumParents(nStates),
          srvb.start(),
          container.loadRegions(0),
          container.toCode((i, _) => deserializers(i)(ib)),
          container.addState(srvb),
          srvb.advance(),
          lenRef := ib.readInt(),
          (lenRef < 0).mux(
            srvb.setMissing(),
            Code(
              region.setNumParents((lenRef + 1) * nStates),
              srvb.addArray(arrayType, sab =>
                Code(
                  sab.start(lenRef),
                  Code.whileLoop(sab.arrayIdx < lenRef,
                    container.loadRegions(regionOffset(sab.arrayIdx)),
                    container.toCode((i, _) => deserializers(i)(ib)),
                    container.addState(sab),
                    sab.advance()))))),
          off := srvb.end())
    }
  }

  def copyFromAddress(src: Code[Long]): Code[Unit] = {
    Code(
      off := src,
      typ.isFieldMissing(region, off, 1).mux(
        Code(lenRef := -1, region.setNumParents(nStates)),
        Code(lenRef := arrayType.loadLength(region, typ.loadField(region, off, 1)),
          region.setNumParents((lenRef + 1) * nStates))),
      srvb.start(),
      container.loadRegions(0),
      container.toCode((i, s) => s.copyFrom(initStateOffset(i))),
      container.addState(srvb),
      srvb.advance(),
      (lenRef < 0).mux(
        srvb.setMissing(),
        srvb.addArray(arrayType, sab =>
          Code(
            sab.start(lenRef),
            Code.whileLoop(sab.arrayIdx < lenRef,
              container.loadRegions(regionOffset(sab.arrayIdx)),
              container.toCode { (i, s) =>
                s.copyFrom(stateAddressOffset(sab.arrayIdx, i))
              },
              container.addState(sab),
              sab.advance())))),
      off := srvb.end())
  }
}

class ArrayElementLengthCheckAggregator(nestedAggs: Array[StagedRegionValueAggregator], knownLength: Boolean) extends StagedRegionValueAggregator {
  type State = ArrayElementState
  private val nStates: Int = nestedAggs.length

  var initOpTypes: Array[PType] = nestedAggs.flatMap(_.initOpTypes)
  if (knownLength)
    initOpTypes = PInt32() +: initOpTypes
  val seqOpTypes: Array[PType] = Array(PInt32())

  val resultEltType: PTuple = PTuple(nestedAggs.map(_.resultType))
  val resultType: PArray = PArray(resultEltType)

  def createState(mb: EmitMethodBuilder): State = ArrayElementState.create(mb, nestedAggs, knownLength)

  // inits all things
  def initOp(state: State, init: Array[RVAVariable], dummy: Boolean): Code[Unit] = {
    var i = if (knownLength) 1 else 0
    val initOps = Array.tabulate(nStates) { sIdx =>
      val agg = nestedAggs(sIdx)
      val vars = init.slice(i, i + agg.initOpTypes.length)
      i += agg.initOpTypes.length
      agg.initOp(state.nested(sIdx), vars)
    }

    if (knownLength) {
      val len = init.head
      assert(len.t isOfType PInt32())
      Code(state.init(initOps), len.setup,
        state.initLength(len.m.mux(Code._fatal("Array length can't be missing"), len.v[Int])))
    } else {
      Code(state.init(initOps), state.lenRef := -1)
    }
  }

  //does a length check on arrays
  def seqOp(state: State, seq: Array[RVAVariable], dummy: Boolean): Code[Unit] = {
    val Array(len) = seq
    assert(len.t isOfType PInt32())

    Code(len.setup, len.m.mux(Code._empty, state.checkLength(len.v[Int])))
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    Code(
      (other.lenRef < 0).mux(
        (state.lenRef < 0).mux(
          Code._empty,
          other.initLength(state.lenRef)),
        state.checkLength(other.lenRef)),
      state.idx := 0,
      Code.whileLoop(state.idx < state.lenRef,
        other.load(state.idx),
        state.load(state.idx),
        state.container.toCode( (i, s) =>
          nestedAggs(i).combOp(s, other.nested(i))),
        state.store(state.idx),
        state.idx := state.idx + 1))
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

class ArrayElementwiseOpAggregator(nestedAggs: Array[StagedRegionValueAggregator]) extends StagedRegionValueAggregator {
  type State = ArrayElementState

  def initOpTypes: Array[PType] = Array()
  def seqOpTypes: Array[PType] = Array(PInt32(), PVoid)

  def resultType: PType = PArray(PTuple(nestedAggs.map(_.resultType)))

  def createState(mb: EmitMethodBuilder): State =
    throw new UnsupportedOperationException(s"State must be created by ArrayElementLengthCheckAggregator")

  def initOp(state: State, init: Array[RVAVariable], dummy: Boolean): Code[Unit] =
    throw new UnsupportedOperationException("State must be initialized by ArrayElementLengthCheckAggregator.")

  def seqOp(state: State, seq: Array[RVAVariable], dummy: Boolean): Code[Unit] = {
    val Array(eltIdx, seqOps) = seq
    assert((eltIdx.t isOfType PInt32()) && (seqOps.t == PVoid))
    val eltIdxV = state.mb.newField[Int]
    Code(
      eltIdx.setup,
      eltIdx.m.mux(
        Code._empty,
        Code(
          eltIdxV := eltIdx.v[Int],
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