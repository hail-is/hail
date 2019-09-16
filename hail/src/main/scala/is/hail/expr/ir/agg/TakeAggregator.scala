package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitFunctionBuilder, EmitTriplet}
import is.hail.expr.types.physical._
import is.hail.io.{BufferSpec, CodecSpec, InputBuffer, OutputBuffer}
import is.hail.utils._

class TakeRVAS(val eltType: PType, val resultType: PArray, val fb: EmitFunctionBuilder[_]) extends AggregatorState {
  private val r: ClassFieldRef[Region] = fb.newField[Region]
  val region: Code[Region] = r.load()

  val builder = new StagedArrayBuilder(eltType, fb, region)
  val storageType: PTuple = PTuple(true, PInt32Required, builder.stateType)
  private val maxSize = fb.newField[Int]
  private val maxSizeOffset: Code[Long] => Code[Long] = storageType.loadField(_, 0)
  private val builderStateOffset: Code[Long] => Code[Long] = storageType.loadField(_, 1)

  def newState: Code[Unit] = region.getNewRegion(regionSize)

  def createState: Code[Unit] = region.isNull.mux(r := Region.stagedCreate(regionSize), Code._empty)

  override def load(regionLoader: Code[Region] => Code[Unit], src: Code[Long]): Code[Unit] =
    Code(
      regionLoader(r),
      maxSize := region.loadInt(maxSizeOffset(src)),
      builder.loadFields(builderStateOffset(src)))

  override def store(regionStorer: Code[Region] => Code[Unit], dest: Code[Long]): Code[Unit] =
    region.isValid.orEmpty(
      Code(
        regionStorer(region),
        region.invalidate(),
        Region.storeInt(maxSizeOffset(dest), maxSize),
        builder.storeFields(builderStateOffset(dest))))

  def serialize(codec: BufferSpec): Code[OutputBuffer] => Code[Unit] = {
    { ob: Code[OutputBuffer] =>
      Code(
        ob.writeInt(maxSize),
        builder.serialize(codec)(ob))
    }
  }

  def deserialize(codec: BufferSpec): Code[InputBuffer] => Code[Unit] = {
    { ib: Code[InputBuffer] =>

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

  def seqOp(elt: EmitTriplet): Code[Unit] = {
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
    val j = fb.newField[Int]
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
    Code(
      maxSize := region.loadInt(maxSizeOffset(src)),
      builder.copyFrom(builderStateOffset(src))
    )
  }
}

class TakeAggregator(typ: PType) extends StagedAggregator {

  type State = TakeRVAS

  val resultType: PArray = PArray(typ)

  def createState(fb: EmitFunctionBuilder[_]): State =
    new TakeRVAS(typ, resultType, fb)

  def initOp(state: State, init: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    assert(init.length == 1)
    val Array(sizeTriplet) = init
    Code(
      sizeTriplet.setup,
      sizeTriplet.m.orEmpty(Code._fatal(s"argument 'n' for 'hl.agg.take' may not be missing")),
      state.init(coerce[Int](sizeTriplet.v))
    )
  }

  def seqOp(state: State, seq: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    val Array(elt: EmitTriplet) = seq
    state.seqOp(elt)
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = state.combine(other, dummy)

  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] = state.result(srvb, dummy)
}
