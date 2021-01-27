package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder}
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.types.physical.stypes.concrete.SBaseStructPointerCode
import is.hail.types.physical.{PBooleanRequired, PCanonicalStruct, PInt32Required, PStruct, PType}
import is.hail.types.virtual.{TFloat64, TInt32, TInt64, Type}
import is.hail.utils._

class ApproxCDFState(val kb: EmitClassBuilder[_]) extends AggregatorState {
  override val regionSize: Region.Size = Region.TINIER

  private val r: Settable[Region] = kb.genFieldThisRef[Region]()
  val region: Value[Region] = r

  val storageType: PStruct = PCanonicalStruct(true, ("id", PInt32Required), ("initialized", PBooleanRequired), ("k", PInt32Required))
  private val aggr = kb.genFieldThisRef[ApproxCDFStateManager]("aggr")

  private val initialized = kb.genFieldThisRef[Boolean]("initialized")
  private val initializedOffset: Code[Long] => Code[Long] = storageType.loadField(_, "initialized")

  private val id = kb.genFieldThisRef[Int]("id")
  private val idOffset: Code[Long] => Code[Long] = storageType.loadField(_, "id")

  private val k = kb.genFieldThisRef[Int]("k")
  private val kOffset: Code[Long] => Code[Long] = storageType.loadField(_, "k")

  def init(cb: EmitCodeBuilder, k: Code[Int]): Unit = {
      cb.assign(this.k, k)
      cb.assign(aggr, Code.newInstance[ApproxCDFStateManager, Int](this.k))
      cb.assign(id, region.storeJavaObject(aggr))
      cb.assign(this.initialized, true)
  }

  def seq(cb: EmitCodeBuilder, x: Code[Double]): Unit = {
    cb += aggr.invoke[Double, Unit]("seqOp", x)
  }

  def comb(cb: EmitCodeBuilder, other: ApproxCDFState): Unit = {
    cb += aggr.invoke[ApproxCDFStateManager, Unit]("combOp", other.aggr)
  }

  def result(cb: EmitCodeBuilder, region: Value[Region]): SBaseStructPointerCode = {
    QuantilesAggregator.resultType.loadCheapPCode(cb, aggr.invoke[Region, Long]("rvResult", region))
  }

  def newState(cb: EmitCodeBuilder, off: Code[Long]): Unit = cb += region.getNewRegion(regionSize)

  def createState(cb: EmitCodeBuilder): Unit =
    cb.ifx(region.isNull, cb.assign(r, Region.stagedCreate(regionSize, kb.pool())))

  override def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, srcc: Code[Long]): Unit = {
    val src = cb.newLocal("acdfa_load_src", srcc)
    regionLoader(cb, r)
    cb.assign(id, Region.loadInt(idOffset(src)))
    cb.assign(initialized, Region.loadBoolean(initializedOffset(src)))
    cb.ifx(initialized,
      {
        cb.assign(aggr, Code.checkcast[ApproxCDFStateManager](region.lookupJavaObject(id)))
        cb.assign(k, Region.loadInt(kOffset(src)))
      })
  }

  override def store(cb: EmitCodeBuilder, regionStorer: (EmitCodeBuilder, Value[Region]) => Unit, destc: Code[Long]): Unit = {
    val dest = cb.newLocal("acdfa_store_dest", destc)
    cb.ifx(region.isValid,
      {
        regionStorer(cb, region)
        cb += region.invalidate()
        cb += Region.storeInt(idOffset(dest), id)
        cb += Region.storeInt(kOffset(dest), k)
        cb += Region.storeBoolean(initializedOffset(dest), initialized)
      })
  }

  override def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    (cb, ob: Value[OutputBuffer]) =>
      cb += Code(
        ob.writeBoolean(initialized),
        ob.writeInt(k),
        initialized.orEmpty(
          aggr.invoke[OutputBuffer, Unit]("serializeTo", ob)
        ))
  }

  override def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    (cb, ib: Value[InputBuffer]) =>
      cb += Code(
        initialized := ib.readBoolean(),
        k := ib.readInt(),
        initialized.orEmpty(
          Code(
            aggr := Code.invokeScalaObject2[Int, InputBuffer, ApproxCDFStateManager](
              ApproxCDFStateManager.getClass, "deserializeFrom", k, ib),
            id := region.storeJavaObject(aggr)
          )
        ))
  }

  override def copyFrom(cb: EmitCodeBuilder, src: Code[Long]): Unit = {
    cb += Code(
      k := Region.loadInt(kOffset(src)),
      aggr := Code.newInstance[ApproxCDFStateManager, Int](k),
      id := region.storeJavaObject(aggr),
      this.initialized := true
    )
  }
}

class ApproxCDFAggregator extends StagedAggregator {
  type State = ApproxCDFState

  def resultType: PStruct = QuantilesAggregator.resultType
  val initOpTypes: Seq[Type] = FastSeq(TInt32)
  val seqOpTypes: Seq[Type] = FastSeq(TFloat64)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val Array(k) = init
    k.toI(cb)
      .consume(cb,
        cb += Code._fatal[Unit]("approx_cdf: 'k' may not be missing"),
        pv => state.init(cb, pv.asInt.intCode(cb)))
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(x) = seq
    x.toI(cb)
      .consume(cb,
        {},
        pv => state.seq(cb, pv.asDouble.doubleCode(cb))
      )
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    state.comb(cb, other)
  }

  protected def _storeResult(cb: EmitCodeBuilder, state: State, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit = {
    assert(pt == resultType)
    pt.storeAtAddress(cb, addr, region, state.result(cb, region), deepCopy = true)
  }
}
