package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder, ParamType}
import is.hail.types.physical._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.utils._

import scala.language.existentials


object CallStatsState {
  val callStatsInternalArrayType = PCanonicalArray(PInt32Required, required = true)
  val stateType: PCanonicalTuple = PCanonicalTuple(true, callStatsInternalArrayType, callStatsInternalArrayType)

  val resultType = PCanonicalStruct(required = true,
    "AC" -> PCanonicalArray(PInt32(true), required = true),
    "AF" -> PCanonicalArray(PFloat64(true), required = false),
    "AN" -> PInt32(true),
    "homozygote_count" -> PCanonicalArray(PInt32(true), required = true))

}

class CallStatsState(val kb: EmitClassBuilder[_]) extends PointerBasedRVAState {
  def alleleCountsOffset: Code[Long] = CallStatsState.stateType.fieldOffset(off, 0)
  def homCountsOffset: Code[Long] = CallStatsState.stateType.fieldOffset(off, 1)
  def alleleCounts: Code[Long] = CallStatsState.stateType.loadField(off, 0)
  def homCounts: Code[Long] = CallStatsState.stateType.loadField(off, 1)
  val nAlleles: Settable[Int] = kb.genFieldThisRef[Int]()
  private val addr = kb.genFieldThisRef[Long]()

  def loadNAlleles: Code[Unit] = nAlleles := CallStatsState.callStatsInternalArrayType.loadLength(alleleCounts)

  // unused but extremely useful for debugging if something goes wrong
  def dump(tag: String): Code[Unit] = {
    val i = kb.genFieldThisRef[Int]()
    Code(
      Code._println(s"at tag $tag"),
      i := 0,
      Code.whileLoop(i < nAlleles,
        Code._println(const("at i=").concat(i.toS).concat(", AC=").concat(alleleCountAtIndex(i, nAlleles).toS).concat(", HOM=").concat(homCountAtIndex(i, nAlleles).toS)),
        i := i + 1)
    )
  }

  override def load(regionLoader: Value[Region] => Code[Unit], src: Code[Long]): Code[Unit] = Code(
    super.load(regionLoader, src),
    loadNAlleles
  )

  def alleleCountAtIndex(idx: Code[Int], length: Code[Int]): Code[Int] =
    Region.loadInt(CallStatsState.callStatsInternalArrayType.loadElement(alleleCounts, length, idx))

  def updateAlleleCountAtIndex(idx: Code[Int], length: Code[Int], updater: Code[Int] => Code[Int]): Code[Unit] = Code(
    addr := CallStatsState.callStatsInternalArrayType.loadElement(alleleCounts, length, idx),
    Region.storeInt(addr, updater(Region.loadInt(addr)))
  )

  def homCountAtIndex(idx: Code[Int], length: Code[Int]): Code[Int] =
    Region.loadInt(CallStatsState.callStatsInternalArrayType.loadElement(homCounts, length, idx))


  def updateHomCountAtIndex(idx: Code[Int], length: Code[Int], updater: Code[Int] => Code[Int]): Code[Unit] = Code(
    addr := CallStatsState.callStatsInternalArrayType.loadElement(homCounts, length, idx),
    Region.storeInt(addr, updater(Region.loadInt(addr)))
  )

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = { (cb, ob) =>
    cb += TypedCodecSpec(CallStatsState.stateType, codec)
      .buildTypedEmitEncoderF[Long](CallStatsState.stateType, kb)(region, off, ob)
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    val (decType, dec) = TypedCodecSpec(CallStatsState.stateType, codec)
      .buildEmitDecoderF[Long](kb)
    assert(decType == CallStatsState.stateType)

    { (cb: EmitCodeBuilder, ib: Value[InputBuffer]) =>
      cb += Code(
        off := dec(region, ib),
        loadNAlleles)
    }
  }

  def copyFromAddress(cb: EmitCodeBuilder, src: Code[Long]): Unit = {
    cb += Code(
      off := StagedRegionValueBuilder.deepCopyFromOffset(kb, region, CallStatsState.stateType, src),
      loadNAlleles
    )
  }
}

class CallStatsAggregator(t: PCall) extends StagedAggregator {

  type State = CallStatsState

  def resultType: PStruct = CallStatsState.resultType
  val initOpTypes: Seq[PType] = FastSeq(PInt32(true))
  val seqOpTypes: Seq[PType] = FastSeq(t)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val Array(nAlleles) = init
    val addr = state.kb.genFieldThisRef[Long]()
    val n = state.kb.genFieldThisRef[Int]()
    val i = state.kb.genFieldThisRef[Int]()
    cb += Code(
      nAlleles.setup,
      nAlleles.m.mux(
        Code._fatal[Unit]("call_stats: n_alleles may not be missing"),
        Code(FastIndexedSeq(
          n := coerce[Int](nAlleles.v),
          state.nAlleles := n,
          state.off := state.region.allocate(CallStatsState.stateType.alignment, CallStatsState.stateType.byteSize),
          addr := CallStatsState.callStatsInternalArrayType.allocate(state.region, n),
          CallStatsState.callStatsInternalArrayType.stagedInitialize(addr, n),
          Region.storeAddress(state.alleleCountsOffset, addr),
          addr := CallStatsState.callStatsInternalArrayType.allocate(state.region, n),
          CallStatsState.callStatsInternalArrayType.stagedInitialize(addr, n),
          Region.storeAddress(state.homCountsOffset, addr),
          i := 0,
          Code.whileLoop(i < n,
            state.updateAlleleCountAtIndex(i, n, _ => 0),
            state.updateHomCountAtIndex(i, n, _ => 0),
            i := i + 1)
        )
      )
    ))
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(call) = seq
    assert(t == call.pv.pt)

    call.toI(cb).consume(cb, { /* do nothing if missing */ }, { case callc: PCallCode =>
      val call = callc.memoize(cb, "callstats_seqop_callv")
      val hom = cb.newLocal[Boolean]("hom", true)
      val lastAllele = cb.newLocal[Int]("lastAllele", -1)
      val i = cb.newLocal[Int]("i", 0)
      call.forEachAllele(cb) { allele: Value[Int] =>
        cb.ifx(allele > state.nAlleles,
          cb._fatal(const("found allele outside of expected range [0, ")
            .concat(state.nAlleles.toS).concat("]: ").concat(allele.toS)))
        cb += state.updateAlleleCountAtIndex(allele, state.nAlleles, _ + 1)
        cb.ifx(i > 0, cb.assign(hom, hom && allele.ceq(lastAllele)))
        cb.assign(lastAllele, allele)
        cb.assign(i, i + 1)
      }

      cb.ifx((i > 1) && hom, { cb += state.updateHomCountAtIndex(lastAllele, state.nAlleles, _ + 1) })
    })
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    val i = state.kb.genFieldThisRef[Int]()
    cb += Code(
      other.nAlleles.cne(state.nAlleles).orEmpty(Code._fatal[Unit]("length mismatch")),
      i := 0,
      Code.whileLoop(i < state.nAlleles,
        state.updateAlleleCountAtIndex(i, state.nAlleles, _ + other.alleleCountAtIndex(i, state.nAlleles)),
        state.updateHomCountAtIndex(i, state.nAlleles, _ + other.homCountAtIndex(i, state.nAlleles)),
        i := i + 1
      )
    )
  }


  protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit = {
    val alleleNumber = state.kb.genFieldThisRef[Int]()
    val i = state.kb.genFieldThisRef[Int]()
    val x = state.kb.genFieldThisRef[Int]()
    cb += srvb.addBaseStruct(CallStatsState.resultType, {
      srvb =>
        Code(
          srvb.start(),
          alleleNumber := 0,
          srvb.addArray(resultType.fieldType("AC").asInstanceOf[PArray], {
            srvb =>
              Code(
                srvb.start(state.nAlleles),
                i := 0,
                Code.whileLoop(i < state.nAlleles,
                  x := state.alleleCountAtIndex(i, state.nAlleles),
                  alleleNumber := alleleNumber + x,
                  srvb.addInt(x),
                  srvb.advance(),
                  i := i + 1))
          }),
          srvb.advance(),
          alleleNumber.ceq(0).mux(
            srvb.setMissing(),
            srvb.addArray(resultType.fieldType("AF").asInstanceOf[PArray], {
              srvb =>
                Code(
                  srvb.start(state.nAlleles),
                  i := 0,
                  Code.whileLoop(i < state.nAlleles,
                    x := state.alleleCountAtIndex(i, state.nAlleles),
                    srvb.addDouble(x.toD / alleleNumber.toD),
                    srvb.advance(),
                    i := i + 1))
            })),
          srvb.advance(),
          srvb.addInt(alleleNumber),
          srvb.advance(),
          srvb.addArray(resultType.fieldType("homozygote_count").asInstanceOf[PArray], {
            srvb =>
              Code(
                srvb.start(state.nAlleles),
                i := 0,
                Code.whileLoop(i < state.nAlleles,
                  x := state.homCountAtIndex(i, state.nAlleles),
                  srvb.addInt(x),
                  srvb.advance(),
                  i := i + 1))
          }))
    })
  }
}
