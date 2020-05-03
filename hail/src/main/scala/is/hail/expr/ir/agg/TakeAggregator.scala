package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitFunctionBuilder}
import is.hail.expr.types.physical._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.utils._

class TakeRVAS(val eltType: PType, val resultType: PArray, val kb: EmitClassBuilder[_]) extends AggregatorState {
  private val r: ThisFieldRef[Region] = kb.genFieldThisRef[Region]()
  val region: Value[Region] = r

  val builder = new StagedArrayBuilder(eltType, kb, region)
  val storageType: PCanonicalTuple = PCanonicalTuple(true, PInt32Required, builder.stateType)
  private val maxSize = kb.genFieldThisRef[Int]()
  private val maxSizeOffset: Code[Long] => Code[Long] = storageType.loadField(_, 0)
  private val builderStateOffset: Code[Long] => Code[Long] = storageType.loadField(_, 1)

  def newState(off: Code[Long]): Code[Unit] = region.getNewRegion(regionSize)

  def createState(cb: EmitCodeBuilder): Unit =
    cb.ifx(region.isNull, { cb.assign(r, Region.stagedCreate(regionSize)) })

  override def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long]): Code[Unit] =
    Code.memoize(src, "take_rvas_src") { src =>
      Code(
        regionLoader(r),
        maxSize := Region.loadInt(maxSizeOffset(src)),
        builder.loadFrom(builderStateOffset(src)))
    }

  override def store(regionStorer: Value[Region] => Code[Unit], dest: Code[Long]): Code[Unit] =
    Code.memoize(dest, "ta_store_dest") { dest =>
      region.isValid.orEmpty(
        Code(
          regionStorer(region),
          region.invalidate(),
          Region.storeInt(maxSizeOffset(dest), maxSize),
          builder.storeTo(builderStateOffset(dest))))
    }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    { (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) =>
      cb += ob.writeInt(maxSize)
      cb += builder.serialize(codec)(ob)
    }
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    { (cb: EmitCodeBuilder, ib: Value[InputBuffer]) =>
      cb.assign(maxSize, ib.readInt())
      cb += builder.deserialize(codec)(ib)
    }
  }

  def init(_maxSize: Code[Int]): Code[Unit] = {
    Code(
      maxSize := _maxSize,
      builder.initialize()
    )
  }

  def seqOp(elt: EmitCode): Code[Unit] = {
    Code(
      elt.setup,
      (builder.size < maxSize)
        .orEmpty(
          elt.m.mux(
            builder.setMissing(),
            builder.append(elt.value))
        )
    )
  }

  def combine(other: TakeRVAS): Code[Unit] = {
    val j = kb.genFieldThisRef[Int]()
    val (eltJMissing, eltJ) = other.builder.loadElement(j)

    Code(
      j := const(0),
      Code.whileLoop((builder.size < maxSize) & (j < other.builder.size),
        eltJMissing.mux(
          builder.setMissing(),
          builder.append(eltJ)
        ),
        j := j + 1
      )
    )
  }

  def result(srvb: StagedRegionValueBuilder): Code[Unit] = {
    srvb.addArray(resultType, { rvb =>
      val (eltIMissing, eltOffset) = builder.elementOffset(rvb.arrayIdx)
      Code(
        rvb.start(builder.size),
        Code.whileLoop(rvb.arrayIdx < builder.size,
          eltIMissing.mux(
            rvb.setMissing(),
            rvb.addWithDeepCopy(eltType, Region.loadIRIntermediate(eltType)(eltOffset))
          ),
          rvb.advance()))
      })
  }

  def copyFrom(cb: EmitCodeBuilder, srcCode: Code[Long]): Unit = {
    val src = cb.newLocal("takervas_copy_from_src", srcCode)
    cb.assign(maxSize, Region.loadInt(maxSizeOffset(src)))
    cb += builder.copyFrom(builderStateOffset(src))
  }
}

class TakeAggregator(typ: PType) extends StagedAggregator {
  assert(typ.isCanonical)

  type State = TakeRVAS

  val resultType: PCanonicalArray = PCanonicalArray(typ, required = true)

  def createState(cb: EmitCodeBuilder): State =
    new TakeRVAS(typ, resultType, cb.emb.ecb)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 1)
    val Array(sizeTriplet) = init
    cb += Code(
      sizeTriplet.setup,
      sizeTriplet.m.orEmpty(Code._fatal[Unit](s"argument 'n' for 'hl.agg.take' may not be missing")),
      state.init(coerce[Int](sizeTriplet.v))
    )
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(elt: EmitCode) = seq
    cb += state.seqOp(elt)
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = cb += state.combine(other)

  protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit = cb += state.result(srvb)
}
