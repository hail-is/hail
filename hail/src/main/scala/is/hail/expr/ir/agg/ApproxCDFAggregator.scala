package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitFunctionBuilder}
import is.hail.expr.types.physical.{PBooleanRequired, PInt32Required, PStruct, PType}
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.utils._

class ApproxCDFState(val cb: EmitClassBuilder[_]) extends AggregatorState {
  override val regionSize: Region.Size = Region.TINIER

  private val r: Settable[Region] = cb.genFieldThisRef[Region]()
  val region: Value[Region] = r

  val storageType: PStruct = PStruct(true, ("id", PInt32Required), ("initialized", PBooleanRequired), ("k", PInt32Required))
  private val aggr = cb.genFieldThisRef[ApproxCDFStateManager]("aggr")

  private val initialized = cb.genFieldThisRef[Boolean]("initialized")
  private val initializedOffset: Code[Long] => Code[Long] = storageType.loadField(_, "initialized")

  private val id = cb.genFieldThisRef[Int]("id")
  private val idOffset: Code[Long] => Code[Long] = storageType.loadField(_, "id")

  private val k = cb.genFieldThisRef[Int]("k")
  private val kOffset: Code[Long] => Code[Long] = storageType.loadField(_, "k")

  def init(k: Code[Int]): Code[Unit] = {
    Code(
      this.k := k,
      aggr := Code.newInstance[ApproxCDFStateManager, Int](this.k),
      id := region.storeJavaObject(aggr),
      this.initialized := true
    )
  }

  def seq(x: Code[Double]): Code[Unit] = {
    aggr.invoke[Double, Unit]("seqOp", x)
  }

  def comb(other: ApproxCDFState): Code[Unit] = {
    aggr.invoke[ApproxCDFStateManager, Unit]("combOp", other.aggr)
  }

  def result(srvb: StagedRegionValueBuilder): Code[Unit] = {
    srvb.addIRIntermediate(QuantilesAggregator.resultType)(aggr.invoke[Region, Long]("rvResult", srvb.region))
  }

  def newState(off: Code[Long]): Code[Unit] = region.getNewRegion(regionSize)

  def createState: Code[Unit] = region.isNull.mux(r := Region.stagedCreate(regionSize), Code._empty)

  override def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long]): Code[Unit] =
    Code.memoize(src, "acdfa_load_src") { src =>
      Code(
        regionLoader(r),
        id := Region.loadInt(idOffset(src)),
        initialized := Region.loadBoolean(initializedOffset(src)),
        initialized.orEmpty(Code(
          aggr := Code.checkcast[ApproxCDFStateManager](region.lookupJavaObject(id)),
          k := Region.loadInt(kOffset(src)))))
    }

  override def store(regionStorer: Value[Region] => Code[Unit], dest: Code[Long]): Code[Unit] =
    Code.memoize(dest, "acdfa_store_dest") { dest =>
      region.isValid.orEmpty(
        Code(
          regionStorer(region),
          region.invalidate(),
          Region.storeInt(idOffset(dest), id),
          Region.storeInt(kOffset(dest), k),
          Region.storeBoolean(initializedOffset(dest), initialized)))
    }

  override def serialize(codec: BufferSpec): Value[OutputBuffer] => Code[Unit] = {
    (ob: Value[OutputBuffer]) =>
      Code(
        ob.writeBoolean(initialized),
        ob.writeInt(k),
        initialized.orEmpty(
          aggr.invoke[OutputBuffer, Unit]("serializeTo", ob)
        ))
  }

  override def deserialize(codec: BufferSpec): Value[InputBuffer] => Code[Unit] = {
    (ib: Value[InputBuffer]) =>
      Code(
        initialized := ib.readBoolean(),
        k := ib.readInt(),
        initialized.orEmpty(
          Code(
            aggr := Code.invokeScalaObject[Int, InputBuffer, ApproxCDFStateManager](
              ApproxCDFStateManager.getClass, "deserializeFrom", k, ib),
            id := region.storeJavaObject(aggr)
          )
        ))
  }

  override def copyFrom(src: Code[Long]): Code[Unit] = {
    Code(
      k := Region.loadInt(kOffset(src)),
      aggr := Code.newInstance[ApproxCDFStateManager, Int](k),
      id := region.storeJavaObject(aggr),
      this.initialized := true
    )
  }
}

class ApproxCDFAggregator extends StagedAggregator {
  type State = ApproxCDFState

  def resultType: PStruct = PType.canonical(QuantilesAggregator.resultType).asInstanceOf[PStruct]

  def createState(cb: EmitClassBuilder[_]): State = new ApproxCDFState(cb)

  def initOp(state: State, init: Array[EmitCode], dummy: Boolean): Code[Unit] = {
    val Array(k) = init
    Code(
      k.setup,
      k.m.mux(
        Code._fatal[Unit]("approx_cdf: 'k' may not be missing"),
        state.init(k.v.asInstanceOf[Code[Int]])
      ))
  }

  def seqOp(state: State, seq: Array[EmitCode], dummy: Boolean): Code[Unit] = {
    val Array(x) = seq
    Code(
      x.setup,
      x.m.mux(
        Code._empty,
        state.seq(x.v.asInstanceOf[Code[Double]])
      ))
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    state.comb(other)
  }


  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] = {
    state.result(srvb)
  }
}
