package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical._
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.concrete.SIndexablePointer
import is.hail.types.virtual.Type
import is.hail.utils._

class CollectAggState(val elemVType: VirtualTypeWithReq, val kb: EmitClassBuilder[_])
    extends AggregatorState {
  private val elemType = elemVType.canonicalPType

  val r = kb.genFieldThisRef[Region]()
  val region: Value[Region] = r
  val bll = new StagedBlockLinkedList(elemType, kb)

  def storageType = bll.storageType

  override def regionSize: Region.Size = Region.REGULAR

  def createState(cb: EmitCodeBuilder): Unit =
    cb.if_(
      region.isNull, {
        cb.assign(r, Region.stagedCreate(regionSize, kb.pool()))
        cb += region.invalidate()
      },
    )

  def newState(cb: EmitCodeBuilder, off: Value[Long]): Unit = cb += region.getNewRegion(regionSize)

  override def load(
    cb: EmitCodeBuilder,
    regionLoader: (EmitCodeBuilder, Value[Region]) => Unit,
    src: Value[Long],
  ): Unit = {
    regionLoader(cb, region)
    bll.load(cb, src)
  }

  override def store(
    cb: EmitCodeBuilder,
    regionStorer: (EmitCodeBuilder, Value[Region]) => Unit,
    dest: Value[Long],
  ): Unit = {
    cb.if_(
      region.isValid, {
        regionStorer(cb, region)
        bll.store(cb, dest)
        cb += region.invalidate()
      },
    )
  }

  def copyFrom(cb: EmitCodeBuilder, src: Value[Long]): Unit = {
    val copyBll = new StagedBlockLinkedList(elemType, kb)
    copyBll.load(cb, src)
    bll.initWithDeepCopy(cb, region, copyBll)
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    (cb, ib) => bll.serialize(cb, region, ib)
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    (cb, ib) =>
      bll.init(cb, region)
      bll.deserialize(cb, region, ib)
  }
}

class CollectAggregator(val elemType: VirtualTypeWithReq) extends StagedAggregator {
  type State = CollectAggState

  val sArrayType = SIndexablePointer(PCanonicalArray(elemType.canonicalPType))
  val resultEmitType = EmitType(sArrayType, true)
  val initOpTypes: Seq[Type] = Array[Type]()
  val seqOpTypes: Seq[Type] = Array[Type](elemType.t)

  protected def _initOp(cb: EmitCodeBuilder, state: State, args: Array[EmitCode]): Unit = {
    assert(args.isEmpty)
    state.bll.init(cb, state.region)
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit =
    state.bll.push(cb, state.region, seq(0))

  protected def _combOp(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    region: Value[Region],
    state: CollectAggState,
    other: CollectAggState,
  ): Unit =
    state.bll.append(cb, state.region, other.bll)

  protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region]): IEmitCode =
    // deepCopy is handled by the blocked linked list
    IEmitCode.present(
      cb,
      state.bll.resultArray(cb, region, resultEmitType.storageType.asInstanceOf[PCanonicalArray]),
    )
}
