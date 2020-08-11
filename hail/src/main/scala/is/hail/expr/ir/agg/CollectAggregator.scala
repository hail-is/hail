package is.hail.expr.ir.agg

import is.hail.asm4s._
import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.expr.ir._
import is.hail.types.physical._
import is.hail.utils._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}

class CollectAggState(val elemType: PType, val kb: EmitClassBuilder[_]) extends AggregatorState {
  val r = kb.genFieldThisRef[Region]()
  val region: Value[Region] = r
  val bll = new StagedBlockLinkedList(elemType, kb)

  def storageType = bll.storageType
  override def regionSize: Region.Size = Region.REGULAR

  def createState(cb: EmitCodeBuilder): Unit =
    cb.ifx(region.isNull, {
      cb.assign(r, Region.stagedCreate(regionSize))
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

  def copyFrom(cb: EmitCodeBuilder, src: Code[Long]): Unit = {
    val copyBll = new StagedBlockLinkedList(elemType, kb)
    cb += Code(
      copyBll.load(src),
      bll.initWithDeepCopy(region, copyBll))
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    { (cb, ib) => cb += bll.serialize(region, ib) }
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    { (cb, ib) => cb += Code(bll.init(region), bll.deserialize(region, ib)) }
  }
}

class CollectAggregator(val elemType: PType) extends StagedAggregator {
  type State = CollectAggState

  assert(elemType.isCanonical)
  val resultType = PCanonicalArray(elemType, required = true)
  val initOpTypes: Seq[PType] = Array[PType]()
  val seqOpTypes: Seq[PType] = Array[PType](elemType)

  protected def _initOp(cb: EmitCodeBuilder, state: State, args: Array[EmitCode]): Unit = {
    assert(args.isEmpty)
    cb += state.bll.init(state.region)
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit =
    cb += state.bll.push(state.region, seq(0))

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit =
    cb += state.bll.append(state.region, other.bll)

  protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit =
    cb += srvb.addArray(resultType, state.bll.writeToSRVB(cb.emb, _))
}
