package is.hail.expr.ir.agg

import is.hail.asm4s._
import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.utils._
import is.hail.io.{CodecSpec, InputBuffer, OutputBuffer, EmitPackEncoder}

class CollectAggregator(val elemType: PType) extends StagedAggregator {

  val resultType = PArray(elemType, required = true)

  class State(val fb: EmitFunctionBuilder[_]) extends AggregatorState {
    override def regionSize: Int = Region.REGULAR

    val r = fb.newField[Region]
    val region = r.load
    val bll = new StagedBlockLinkedList(elemType, fb)

    val isLoaded: Code[Boolean] = region.isValid

    def createState: Code[Unit] =
      region.isNull.orEmpty(Code(
        r := Code.newInstance[Region, Int](regionSize),
        region.invalidate()))

    def newState: Code[Unit] =
      region.getNewRegion(regionSize)

    val storageType = bll.storageType

    def load(regionLoader: Code[Region] => Code[Unit], src: Code[Long]): Code[Unit] =
      Code(
        regionLoader(region),
        bll.load(src))

    def store(regionStorer: Code[Region] => Code[Unit], dst: Code[Long]): Code[Unit] =
      region.isValid.orEmpty(Code(
        regionStorer(region),
        bll.store(dst),
        region.invalidate()))

    def copyFrom(src: Code[Long]): Code[Unit] = {
      val copyBll = new StagedBlockLinkedList(elemType, fb)
      Code(
        copyBll.load(src),
        bll.initWithDeepCopy(region, copyBll))
    }

    def serialize(codec: CodecSpec): Code[OutputBuffer] => Code[Unit] =
      bll.serialize(region, _)

    def deserialize(codec: CodecSpec): Code[InputBuffer] => Code[Unit] = {
      { ob => Code(bll.init(region), bll.deserialize(region, ob)) }
    }
  }

  def createState(fb: EmitFunctionBuilder[_]) =
    new State(fb)

  def initOp(state: State, args: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    assert(args.isEmpty)
    state.bll.init(state.region)
  }

  def seqOp(state: State, seq: Array[EmitTriplet], dummy: Boolean): Code[Unit] =
    state.bll.push(state.region, seq(0))

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] =
    state.bll.append(state.region, other.bll)

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] =
    srvb.addArray(resultType, state.bll.writeToSRVB(_))

}
