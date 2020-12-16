package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitFunctionBuilder}
import is.hail.types.physical._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer}
import is.hail.utils._

class TakeRVAS(val eltType: PType, val resultType: PArray, val kb: EmitClassBuilder[_]) extends AggregatorState {
  private val r: ThisFieldRef[Region] = kb.genFieldThisRef[Region]()
  val region: Value[Region] = r

  val builder = new StagedArrayBuilder(eltType, kb, region)
  val storageType: PCanonicalTuple = PCanonicalTuple(true, PInt32Required, builder.stateType)
  private val maxSize = kb.genFieldThisRef[Int]()
  private def maxSizeOffset(off: Code[Long])(implicit line: LineNumber): Code[Long] =
    storageType.loadField(off, 0)
  private def builderStateOffset(off: Code[Long])(implicit line: LineNumber): Code[Long] =
    storageType.loadField(off, 1)

  def newState(off: Code[Long])(implicit line: LineNumber): Code[Unit] = region.getNewRegion(regionSize)

  def createState(cb: EmitCodeBuilder): Unit = {
    implicit val line = cb.lineNumber
    cb.ifx(region.isNull, { cb.assign(r, Region.stagedCreate(regionSize, kb.pool())) })
  }

  override def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long])(implicit line: LineNumber): Code[Unit] =
    Code.memoize(src, "take_rvas_src") { src =>
      Code(
        regionLoader(r),
        maxSize := Region.loadInt(maxSizeOffset(src)),
        builder.loadFrom(builderStateOffset(src)))
    }

  override def store(regionStorer: Value[Region] => Code[Unit], dest: Code[Long])(implicit line: LineNumber): Code[Unit] =
    Code.memoize(dest, "ta_store_dest") { dest =>
      region.isValid.orEmpty(
        Code(
          regionStorer(region),
          region.invalidate(),
          Region.storeInt(maxSizeOffset(dest), maxSize),
          builder.storeTo(builderStateOffset(dest))))
    }

  def serialize(codec: BufferSpec)(implicit line: LineNumber): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    { (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) =>
      cb += ob.writeInt(maxSize)
      cb += builder.serialize(codec, ob)
    }
  }

  def deserialize(codec: BufferSpec)(implicit line: LineNumber): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    { (cb: EmitCodeBuilder, ib: Value[InputBuffer]) =>
      cb.assign(maxSize, ib.readInt())
      cb += builder.deserialize(codec, ib)
    }
  }

  def init(_maxSize: Code[Int])(implicit line: LineNumber): Code[Unit] = {
    Code(
      maxSize := _maxSize,
      builder.initialize()
    )
  }

  def seqOp(elt: EmitCode)(implicit line: LineNumber): Code[Unit] = {
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

  def combine(cb: EmitCodeBuilder, other: TakeRVAS): Unit = {
    implicit val line = cb.lineNumber
    val j = kb.genFieldThisRef[Int]()
    val elt = other.builder.loadElement(cb, j)

    cb += Code(
      j := const(0),
      Code.whileLoop((builder.size < maxSize) & (j < other.builder.size),
        elt.setup,
        elt.m.mux(
          builder.setMissing(),
          builder.append(elt.v)
        ),
        j := j + 1
      )
    )
  }

  def result(srvb: StagedRegionValueBuilder)(implicit line: LineNumber): Code[Unit] = {
    srvb.addArray(resultType, { rvb =>
      val elt = EmitCodeBuilder.scopedEmitCode(srvb.mb)(cb => builder.loadElement(cb, rvb.arrayIdx))
      Code(
        rvb.start(builder.size),
        Code.whileLoop(rvb.arrayIdx < builder.size,
          elt.setup,
          elt.m.mux(
            rvb.setMissing(),
            rvb.addWithDeepCopy(eltType, elt.v)
          ),
          rvb.advance()))
      })
  }

  def copyFrom(cb: EmitCodeBuilder, srcCode: Code[Long]): Unit = {
    implicit val line = cb.lineNumber
    val src = cb.newLocal("takervas_copy_from_src", srcCode)
    cb.assign(maxSize, Region.loadInt(maxSizeOffset(src)))
    cb += builder.copyFrom(builderStateOffset(src))
  }
}

class TakeAggregator(typ: PType) extends StagedAggregator {
  assert(typ.isCanonical)

  type State = TakeRVAS

  val resultType: PCanonicalArray = PCanonicalArray(typ, required = true)
  val initOpTypes: Seq[PType] = Array(PInt32Required)
  val seqOpTypes: Seq[PType] = Array(typ)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    implicit val line = cb.lineNumber
    assert(init.length == 1)
    val Array(sizeTriplet) = init
    cb += Code(
      sizeTriplet.setup,
      sizeTriplet.m.orEmpty(Code._fatal[Unit](s"argument 'n' for 'hl.agg.take' may not be missing")),
      state.init(coerce[Int](sizeTriplet.v))
    )
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    implicit val line = cb.lineNumber
    val Array(elt: EmitCode) = seq
    cb += state.seqOp(elt)
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit =
    state.combine(cb, other)

  protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit = {
    implicit val line = cb.lineNumber
    cb += state.result(srvb)
  }
}
