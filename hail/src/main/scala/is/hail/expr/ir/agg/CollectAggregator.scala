package is.hail.expr.ir.agg

import is.hail.asm4s._
import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.utils._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}

class CollectAggregator(val elemType: PType) extends StagedAggregator {

  assert(elemType.isCanonical)
  val resultType = PCanonicalArray(elemType, required = true)

  class State(val cb: EmitClassBuilder[_]) extends AggregatorState {
    val r = cb.genFieldThisRef[Region]()
    val region: Value[Region] = r
    val bll = new StagedBlockLinkedList(elemType, cb)

    def storageType = bll.storageType
    override def regionSize: Region.Size = Region.REGULAR

    def createState: Code[Unit] =
      region.isNull.orEmpty(Code(
        r := Region.stagedCreate(regionSize),
        region.invalidate()))

    def newState(off: Code[Long]): Code[Unit] =
      region.getNewRegion(regionSize)

    def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long]): Code[Unit] =
      Code(
        regionLoader(region),
        bll.load(src))

    def store(regionStorer: Value[Region] => Code[Unit], dst: Code[Long]): Code[Unit] =
      region.isValid.orEmpty(Code(
        regionStorer(region),
        bll.store(dst),
        region.invalidate()))

    def copyFrom(src: Code[Long]): Code[Unit] = {
      val copyBll = new StagedBlockLinkedList(elemType, cb)
      Code(
        copyBll.load(src),
        bll.initWithDeepCopy(region, copyBll))
    }

    def serialize(codec: BufferSpec): Value[OutputBuffer] => Code[Unit] =
      bll.serialize(region, _)

    def deserialize(codec: BufferSpec): Value[InputBuffer] => Code[Unit] = {
      { ib => Code(bll.init(region), bll.deserialize(region, ib)) }
    }
  }

  def createState(cb: EmitClassBuilder[_]): State =
    new State(cb)

  def initOp(state: State, args: Array[EmitCode], dummy: Boolean): Code[Unit] = {
    assert(args.isEmpty)
    state.bll.init(state.region)
  }

  def seqOp(state: State, seq: Array[EmitCode], dummy: Boolean): Code[Unit] =
    state.bll.push(state.region, seq(0))

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] =
    state.bll.append(state.region, other.bll)

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] =
    srvb.addArray(resultType, state.bll.writeToSRVB(_))
}
