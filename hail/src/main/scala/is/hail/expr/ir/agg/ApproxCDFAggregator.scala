package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitContext, IEmitCode}
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.types.physical._
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.concrete.{SBaseStructPointer, SBaseStructPointerValue}
import is.hail.types.virtual.{TFloat64, TInt32, Type}
import is.hail.utils._

class ApproxCDFState(val kb: EmitClassBuilder[_]) extends AggregatorState {
  override val regionSize: Region.Size = Region.TINIER

  private val r: Settable[Region] = kb.genFieldThisRef[Region]()
  val region: Value[Region] = r

  val storageType: PStruct = PCanonicalStruct(
    true,
    ("id", PInt32Required),
    ("initialized", PBooleanRequired),
    ("k", PInt32Required),
  )

  private val aggr = kb.genFieldThisRef[ApproxCDFStateManager]("aggr")

  private val initialized = kb.genFieldThisRef[Boolean]("initialized")
  private val initializedOffset: Code[Long] => Code[Long] = storageType.loadField(_, "initialized")

  private val id = kb.genFieldThisRef[Int]("id")
  private val idOffset: Code[Long] => Code[Long] = storageType.loadField(_, "id")

  private val k = kb.genFieldThisRef[Int]("k")
  private val kOffset: Code[Long] => Code[Long] = storageType.loadField(_, "k")

  def init(cb: EmitCodeBuilder, k: Code[Int]): Unit = {
    cb.assign(this.k, k)
    cb.assign(
      aggr,
      Code.invokeScalaObject1[Int, ApproxCDFStateManager](
        ApproxCDFStateManager.getClass,
        "apply",
        this.k,
      ),
    )
    cb.assign(id, region.storeJavaObject(aggr))
    cb.assign(this.initialized, true)
  }

  def seq(cb: EmitCodeBuilder, x: Code[Double]): Unit =
    cb += aggr.invoke[Double, Unit]("seqOp", x)

  def comb(cb: EmitCodeBuilder, other: ApproxCDFState): Unit =
    cb += aggr.invoke[ApproxCDFStateManager, Unit]("combOp", other.aggr)

  def result(cb: EmitCodeBuilder, region: Value[Region]): SBaseStructPointerValue =
    QuantilesAggregator.resultPType.loadCheapSCode(
      cb,
      aggr.invoke[Region, Long]("rvResult", region),
    )

  def newState(cb: EmitCodeBuilder, off: Value[Long]): Unit = cb += region.getNewRegion(regionSize)

  def createState(cb: EmitCodeBuilder): Unit =
    cb.if_(region.isNull, cb.assign(r, Region.stagedCreate(regionSize, kb.pool())))

  override def load(
    cb: EmitCodeBuilder,
    regionLoader: (EmitCodeBuilder, Value[Region]) => Unit,
    src: Value[Long],
  ): Unit = {
    regionLoader(cb, r)
    cb.assign(id, Region.loadInt(idOffset(src)))
    cb.assign(initialized, Region.loadBoolean(initializedOffset(src)))
    cb.if_(
      initialized, {
        cb.assign(aggr, Code.checkcast[ApproxCDFStateManager](region.lookupJavaObject(id)))
        cb.assign(k, Region.loadInt(kOffset(src)))
      },
    )
  }

  override def store(
    cb: EmitCodeBuilder,
    regionStorer: (EmitCodeBuilder, Value[Region]) => Unit,
    dest: Value[Long],
  ): Unit = {
    cb.if_(
      region.isValid, {
        regionStorer(cb, region)
        cb += region.invalidate()
        cb += Region.storeInt(idOffset(dest), id)
        cb += Region.storeInt(kOffset(dest), k)
        cb += Region.storeBoolean(initializedOffset(dest), initialized)
      },
    )
  }

  override def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    (cb, ob: Value[OutputBuffer]) =>
      cb += ob.writeBoolean(initialized)
      cb += ob.writeInt(k)
      cb.if_(initialized, cb += aggr.invoke[OutputBuffer, Unit]("serializeTo", ob))
  }

  override def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    (cb, ib: Value[InputBuffer]) =>
      cb.assign(initialized, ib.readBoolean())
      cb.assign(k, ib.readInt())
      cb.if_(
        initialized, {
          cb.assign(
            aggr,
            Code.invokeScalaObject2[Int, InputBuffer, ApproxCDFStateManager](
              ApproxCDFStateManager.getClass,
              "deserializeFrom",
              k,
              ib,
            ),
          )

          cb.assign(id, region.storeJavaObject(aggr))
        },
      )
  }

  override def copyFrom(cb: EmitCodeBuilder, src: Value[Long]): Unit = {
    cb.assign(k, Region.loadInt(kOffset(src)))
    cb.assign(
      aggr,
      Code.invokeScalaObject1[Int, ApproxCDFStateManager](
        ApproxCDFStateManager.getClass,
        "apply",
        this.k,
      ),
    )
    cb.assign(id, region.storeJavaObject(aggr))
    cb.assign(this.initialized, true)
  }
}

class ApproxCDFAggregator extends StagedAggregator {
  type State = ApproxCDFState

  def resultEmitType: EmitType = EmitType(SBaseStructPointer(QuantilesAggregator.resultPType), true)
  val initOpTypes: Seq[Type] = FastSeq(TInt32)
  val seqOpTypes: Seq[Type] = FastSeq(TFloat64)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val Array(k) = init
    k.toI(cb)
      .consume(
        cb,
        cb += Code._fatal[Unit]("approx_cdf: 'k' may not be missing"),
        pv => state.init(cb, pv.asInt.value),
      )
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(x) = seq
    x.toI(cb)
      .consume(
        cb,
        {},
        pv => state.seq(cb, pv.asDouble.value),
      )
  }

  protected def _combOp(
    ctx: ExecuteContext,
    cb: EmitCodeBuilder,
    region: Value[Region],
    state: ApproxCDFState,
    other: ApproxCDFState,
  ): Unit =
    state.comb(cb, other)

  protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region]): IEmitCode =
    IEmitCode.present(cb, state.result(cb, region))
}
