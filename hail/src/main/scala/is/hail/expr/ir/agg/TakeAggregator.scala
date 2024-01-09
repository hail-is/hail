package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s.{Code, _}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder, IEmitCode}
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical._
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SIndexablePointerValue}
import is.hail.types.virtual.{TInt32, Type}
import is.hail.utils._

class TakeRVAS(val eltType: VirtualTypeWithReq, val kb: EmitClassBuilder[_])
    extends AggregatorState {
  val eltPType = eltType.canonicalPType

  private val r: ThisFieldRef[Region] = kb.genFieldThisRef[Region]()
  val region: Value[Region] = r

  val builder = new StagedArrayBuilder(eltPType, kb, region)
  val storageType: PCanonicalTuple = PCanonicalTuple(true, PInt32Required, builder.stateType)
  private val maxSize = kb.genFieldThisRef[Int]()
  private val maxSizeOffset: Code[Long] => Code[Long] = storageType.loadField(_, 0)
  private val builderStateOffset: Code[Long] => Code[Long] = storageType.loadField(_, 1)

  def newState(cb: EmitCodeBuilder, off: Value[Long]): Unit = cb += region.getNewRegion(regionSize)

  def createState(cb: EmitCodeBuilder): Unit =
    cb.if_(region.isNull, cb.assign(r, Region.stagedCreate(regionSize, kb.pool())))

  override def load(
    cb: EmitCodeBuilder,
    regionLoader: (EmitCodeBuilder, Value[Region]) => Unit,
    src: Value[Long],
  ): Unit = {
    regionLoader(cb, r)
    cb.assign(maxSize, Region.loadInt(maxSizeOffset(src)))
    builder.loadFrom(cb, builderStateOffset(src))
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
        cb += Region.storeInt(maxSizeOffset(dest), maxSize)
        builder.storeTo(cb, builderStateOffset(dest))
      },
    )
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) =>
      cb += ob.writeInt(maxSize)
      builder.serialize(codec)(cb, ob)
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    (cb: EmitCodeBuilder, ib: Value[InputBuffer]) =>
      cb.assign(maxSize, ib.readInt())
      builder.deserialize(codec)(cb, ib)
  }

  def init(cb: EmitCodeBuilder, _maxSize: Code[Int]): Unit = {
    cb.assign(maxSize, _maxSize)
    builder.initialize(cb)
  }

  def seqOp(cb: EmitCodeBuilder, elt: EmitCode): Unit =
    cb.if_(
      builder.size < maxSize,
      elt.toI(cb)
        .consume(cb, builder.setMissing(cb), sc => builder.append(cb, sc)),
    )

  def combine(cb: EmitCodeBuilder, other: TakeRVAS): Unit = {
    val j = kb.genFieldThisRef[Int]()
    cb.assign(j, 0)
    cb.while_(
      (builder.size < maxSize) && (j < other.builder.size), {
        other.builder.loadElement(cb, j).toI(cb)
          .consume(cb, builder.setMissing(cb), sc => builder.append(cb, sc))
        cb.assign(j, j + 1)
      },
    )
  }

  def resultArray(cb: EmitCodeBuilder, region: Value[Region], resType: PCanonicalArray)
    : SIndexablePointerValue =
    resType.constructFromElements(cb, region, builder.size, deepCopy = true) { (cb, idx) =>
      builder.loadElement(cb, idx).toI(cb)
    }

  def copyFrom(cb: EmitCodeBuilder, src: Value[Long]): Unit = {
    cb.assign(maxSize, Region.loadInt(maxSizeOffset(src)))
    builder.copyFrom(cb, builderStateOffset(src))
  }
}

class TakeAggregator(typ: VirtualTypeWithReq) extends StagedAggregator {
  type State = TakeRVAS

  private val pt = typ.canonicalPType
  val resultPType: PCanonicalArray = PCanonicalArray(pt)
  val resultEmitType: EmitType = EmitType(SIndexablePointer(resultPType), true)
  val initOpTypes: Seq[Type] = Array(TInt32)
  val seqOpTypes: Seq[Type] = Array(typ.t)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 1)
    val Array(sizeTriplet) = init
    sizeTriplet.toI(cb)
      .consume(
        cb,
        cb += Code._fatal[Unit](s"argument 'n' for 'hl.agg.take' may not be missing"),
        sc => state.init(cb, sc.asInt.value),
      )
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(elt: EmitCode) = seq
    state.seqOp(cb, elt)
  }

  protected def _combOp(ctx: ExecuteContext, cb: EmitCodeBuilder, state: TakeRVAS, other: TakeRVAS)
    : Unit = state.combine(cb, other)

  protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region]): IEmitCode =
    // deepCopy is handled by state.resultArray
    IEmitCode.present(cb, state.resultArray(cb, region, resultPType))
}
