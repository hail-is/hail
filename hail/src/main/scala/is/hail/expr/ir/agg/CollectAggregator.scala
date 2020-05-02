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

  class State(val kb: EmitClassBuilder[_]) extends AggregatorState {
    val r = kb.genFieldThisRef[Region]()
    val region: Value[Region] = r
    val bll = new StagedBlockLinkedList(elemType, kb)

    def storageType = bll.storageType
    override def regionSize: Region.Size = Region.REGULAR

    def createState(cb: EmitCodeBuilder): Unit =
      cb.ifx(region.isNull, {
        cb.assign(r, region.stagedCreate(regionSize))
        cb += region.invalidate()
      })

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
      val copyBll = new StagedBlockLinkedList(elemType, kb)
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

  def createState(cb: EmitCodeBuilder): State = new State(cb.emb.ecb)

  protected def _initOp(state: State, args: Array[EmitCode]): Code[Unit] = {
    assert(args.isEmpty)
    state.bll.init(state.region)
  }

  protected def _seqOp(state: State, seq: Array[EmitCode]): Code[Unit] =
    state.bll.push(state.region, seq(0))

  protected def _combOp(state: State, other: State): Code[Unit] =
    state.bll.append(state.region, other.bll)

  protected def _result(state: State, srvb: StagedRegionValueBuilder): Code[Unit] =
    srvb.addArray(resultType, state.bll.writeToSRVB(_))
}
