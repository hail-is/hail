package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitFunctionBuilder}
import is.hail.expr.types.physical._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.utils._

class TakeRVAS(val eltType: PType, val resultType: PArray, val cb: EmitClassBuilder[_]) extends AggregatorState {
  private val r: ThisFieldRef[Region] = cb.genFieldThisRef[Region]()
  val region: Value[Region] = r

  val builder = new StagedArrayBuilder(eltType, cb, region)
  val storageType: PTuple = PTuple(true, PInt32Required, builder.stateType)
  private val maxSize = cb.genFieldThisRef[Int]()
  private val maxSizeOffset: Code[Long] => Code[Long] = storageType.loadField(_, 0)
  private val builderStateOffset: Code[Long] => Code[Long] = storageType.loadField(_, 1)

  def newState(off: Code[Long]): Code[Unit] = region.getNewRegion(regionSize)

  def createState: Code[Unit] = region.isNull.mux(r := Region.stagedCreate(regionSize), Code._empty)

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

  def serialize(codec: BufferSpec): Value[OutputBuffer] => Code[Unit] = {
    { ob: Value[OutputBuffer] =>
      Code(
        ob.writeInt(maxSize),
        builder.serialize(codec)(ob))
    }
  }

  def deserialize(codec: BufferSpec): Value[InputBuffer] => Code[Unit] = {
    { ib: Value[InputBuffer] =>

      Code(
        maxSize := ib.readInt(),
        builder.deserialize(codec)(ib))
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

  def combine(other: TakeRVAS, dummy: Boolean): Code[Unit] = {
    val j = cb.genFieldThisRef[Int]()
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

  def result(srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] = {
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

  def copyFrom(src: Code[Long]): Code[Unit] = {
    Code.memoize(src, "takervas_copy_from_src") { src =>
      Code(
        maxSize := Region.loadInt(maxSizeOffset(src)),
        builder.copyFrom(builderStateOffset(src)))
    }
  }
}

class TakeAggregator(typ: PType) extends StagedAggregator {

  type State = TakeRVAS

  val resultType: PArray = PArray(typ)

  def createState(cb: EmitClassBuilder[_]): State =
    new TakeRVAS(typ, resultType, cb)

  def initOp(state: State, init: Array[EmitCode], dummy: Boolean): Code[Unit] = {
    assert(init.length == 1)
    val Array(sizeTriplet) = init
    Code(
      sizeTriplet.setup,
      sizeTriplet.m.orEmpty(Code._fatal[Unit](s"argument 'n' for 'hl.agg.take' may not be missing")),
      state.init(coerce[Int](sizeTriplet.v))
    )
  }

  def seqOp(state: State, seq: Array[EmitCode], dummy: Boolean): Code[Unit] = {
    val Array(elt: EmitCode) = seq
    state.seqOp(elt)
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = state.combine(other, dummy)

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] = state.result(srvb, dummy)
}
