package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
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
    ArrayElementState(aggs.map(_.createState(mb)), mb.newField[Region], mb.newField[Long], knownLength)
}

case class ArrayElementState(nested: Array[RVAState], r: ClassFieldRef[Region], off: ClassFieldRef[Long], knownLength: Boolean) extends RVAState {
  private val container: StateContainer = StateContainer(nested, r)
  private val arrayType: PArray = PArray(container.typ)

  private val nStates: Int = nested.length
  val mb: EmitMethodBuilder = nested.head.mb

  val typ: PTuple = PTuple(FastIndexedSeq(container.typ, arrayType))

  val lenRef: ClassFieldRef[Int] = mb.newField[Int]
  val idx: ClassFieldRef[Int] = mb.newField[Int]

  val srvb = new StagedRegionValueBuilder(er, typ)

  private def regionOffset(eltIdx: Code[Int]): Code[Int] = (eltIdx + 1) * nStates
  private def regionFromIdx(eltIdx: Code[Int], stateIdx: Int): Code[Region] = container.getRegion(regionOffset(eltIdx), stateIdx)

  private val initStatesOffset = typ.loadField(region, off, 0)
  private def initStateOffset(idx: Int): Code[Long] = container.getStateOffset(initStatesOffset, idx)
  private def initStateAddress(idx: Int): Code[Long] = container.loadStateAddress(initStatesOffset, idx)

  private def statesOffset(eltIdx: Code[Int]): Code[Long] = arrayType.loadElement(region, typ.loadField(region, off, 1), eltIdx)
  private def stateAddressOffset(eltIdx: Code[Int], stateIdx: Int): Code[Long] = container.getStateOffset(statesOffset(eltIdx), stateIdx)
  private def eltState(eltIdx: Code[Int], stateIdx: Int): Code[Long] = container.loadStateAddress(statesOffset(eltIdx), stateIdx)

  def initLength(len: Code[Int]): Code[Unit] = {
    val srvb2 = new StagedRegionValueBuilder(er, arrayType)
    Code(
      lenRef := len,
      region.setNumParents((lenRef + 1) * nStates),
      srvb2.start(lenRef),
      Code.whileLoop(srvb2.arrayIdx < lenRef,
        container.scoped(regionOffset(srvb2.arrayIdx)){ (i, s) =>
          s.copyFrom(initStateAddress(i)) },
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

  def init(initOp: Array[() => Code[Unit]]): Code[Unit] = {
      val c = Code(
        region.setNumParents(nStates),
        srvb.start(),
        container.scoped(0)((i, _) => initOp(i)()),
        container.addState(srvb),
        srvb.advance())
    if (knownLength)
      Code(c, off := srvb.end())
    else
      Code(c, srvb.setMissing(), off := srvb.end())
  }

  def scoped(eltIdx: Code[Int], f: Code[Unit]): Code[Unit] =
    container.scoped(regionOffset(eltIdx), statesOffset(eltIdx), f)

  def scoped(eltIdx: Code[Int])(f: (Int, RVAState) => Code[Unit]): Code[Unit] =
    container.scoped(regionOffset(eltIdx), statesOffset(eltIdx))(f)

  def update(eltIdx: Code[Int], f: Code[Unit]): Code[Unit] =
    container.update(regionOffset(eltIdx), statesOffset(eltIdx), f)

  def update(eltIdx: Code[Int])(f: (Int, RVAState) => Code[Unit]): Code[Unit] =
    container.update(regionOffset(eltIdx), statesOffset(eltIdx))(f)

  def serialize(codec: CodecSpec): Code[OutputBuffer] => Code[Unit] = {
    val serializers = nested.map(_.serialize(codec));
    { ob: Code[OutputBuffer] =>
      val serialize = coerce[Unit](Code(serializers.map(_(ob)): _*))
      Code(
        container.loadStateOffsets(initStatesOffset),
        serialize,
        ob.writeInt(lenRef),
        idx := 0,
        Code.whileLoop(idx < lenRef,
          container.loadStateOffsets(statesOffset(idx)),
          serialize,
          idx := idx + 1),
        region.close(),
        r := Code._null)
    }
  }

  def unserialize(codec: CodecSpec): Code[InputBuffer] => Code[Unit] = {
    val deserializers = nested.map(_.unserialize(codec));
    { ib: Code[InputBuffer] =>
        Code(
          region.setNumParents(nStates),
          srvb.start(),
          container.scoped(0)((i, _) => deserializers(i)(ib)),
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
                    container.scoped(regionOffset(sab.arrayIdx))((i, _) => deserializers(i)(ib)),
                    container.addState(sab),
                    sab.advance()))))),
          off := srvb.end())
    }
  }

  def copyFrom(src: Code[Long]): Code[Unit] = {
    Code(
      off := src,
      typ.isFieldMissing(region, off, 1).mux(
        Code(lenRef := -1, region.setNumParents(nStates)),
        Code(lenRef := arrayType.loadLength(region, typ.loadField(region, off, 1)),
          region.setNumParents((lenRef + 1) * nStates))),
      srvb.start(),
      container.scoped(0, initStatesOffset)((i, s) => s.copyFrom(initStateOffset(i))),
      container.addState(srvb),
      srvb.advance(),
      (lenRef < 0).mux(
        srvb.setMissing(),
        srvb.addArray(arrayType, sab =>
          Code(
            sab.start(lenRef),
            Code.whileLoop(sab.arrayIdx < lenRef,
              scoped(sab.arrayIdx) { (i, s) =>
                s.copyFrom(eltState(sab.arrayIdx, i))
              }),
            container.addState(sab),
            sab.advance()))),
      off := srvb.end())
  }
}

class ArrayElementRegionValueAggregator(nestedAggs: Array[StagedRegionValueAggregator], knownLength: Boolean) extends StagedRegionValueAggregator {
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
      () => agg.initOp(state.nested(sIdx), vars)
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
        other.scoped(state.idx,
          state.update(state.idx)( (i, s) =>
            nestedAggs(i).combOp(s, other.nested(i)))),
        state.idx := state.idx + 1))
  }

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] =
    srvb.addArray(resultType, { sab =>
      Code(
        sab.start(state.lenRef),
        Code.whileLoop(sab.arrayIdx < state.lenRef,
          sab.addBaseStruct(resultEltType, { ssb =>
            Code(
              ssb.start(),
              state.scoped(sab.arrayIdx) { (i, s) =>
                Code(nestedAggs(i).result(s, ssb), ssb.advance())
              })
          }),
          sab.advance()))
    })
}

class ArrayElementRegionValueAggregator2(nestedAggs: Array[StagedRegionValueAggregator]) extends StagedRegionValueAggregator {
  type State = ArrayElementState

  def initOpTypes: Array[PType] = Array()
  def seqOpTypes: Array[PType] = Array(PInt32(), PVoid)

  def resultType: PType = PArray(PTuple(nestedAggs.map(_.resultType)))

  def createState(mb: EmitMethodBuilder): State =
    throw new UnsupportedOperationException("foo")

  def initOp(state: State, init: Array[RVAVariable], dummy: Boolean): Code[Unit] =
    throw new UnsupportedOperationException("foo")

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
            state.update(eltIdxV, seqOps.setup)))))
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] =
    throw new UnsupportedOperationException("foo")

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] =
    throw new UnsupportedOperationException("foo")
}