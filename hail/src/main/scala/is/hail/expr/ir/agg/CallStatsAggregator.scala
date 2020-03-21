package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitFunctionBuilder}
import is.hail.expr.types.physical._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.stats.CallStats
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

  assert(resultType.virtualType == CallStats.schema)
}

class CallStatsState(val cb: EmitClassBuilder[_]) extends PointerBasedRVAState {
  def alleleCountsOffset: Code[Long] = CallStatsState.stateType.fieldOffset(off, 0)
  def homCountsOffset: Code[Long] = CallStatsState.stateType.fieldOffset(off, 1)
  def alleleCounts: Code[Long] = CallStatsState.stateType.loadField(off, 0)
  def homCounts: Code[Long] = CallStatsState.stateType.loadField(off, 1)
  val nAlleles: Settable[Int] = cb.genFieldThisRef[Int]()
  private val addr = cb.genFieldThisRef[Long]()

  def loadNAlleles: Code[Unit] = nAlleles := CallStatsState.callStatsInternalArrayType.loadLength(alleleCounts)

  // unused but extremely useful for debugging if something goes wrong
  def dump(tag: String): Code[Unit] = {
    val i = cb.genFieldThisRef[Int]()
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

  def serialize(codec: BufferSpec): Value[OutputBuffer] => Code[Unit] = {
    TypedCodecSpec(CallStatsState.stateType, codec).buildEmitEncoderF[Long](CallStatsState.stateType, cb)(region, off, _)
  }

  def deserialize(codec: BufferSpec): Value[InputBuffer] => Code[Unit] = {
    val (decType, dec) = TypedCodecSpec(CallStatsState.stateType, codec)
      .buildEmitDecoderF[Long](CallStatsState.stateType.virtualType, cb)
    assert(decType == CallStatsState.stateType)

    { ib: Value[InputBuffer] =>
      Code(
        off := dec(region, ib),
        loadNAlleles)
    }
  }

  def copyFromAddress(src: Code[Long]): Code[Unit] = {
    Code(
      off := StagedRegionValueBuilder.deepCopyFromOffset(cb, region, CallStatsState.stateType, src),
      loadNAlleles
    )
  }
}

class CallStatsAggregator(t: PCall) extends StagedAggregator {

  type State = CallStatsState

  def resultType: PStruct = CallStatsState.resultType

  def createState(cb: EmitClassBuilder[_]): State = new CallStatsState(cb)

  def initOp(state: State, init: Array[EmitCode], dummy: Boolean): Code[Unit] = {
    val Array(nAlleles) = init
    val addr = state.cb.genFieldThisRef[Long]()
    val n = state.cb.genFieldThisRef[Int]()
    val i = state.cb.genFieldThisRef[Int]()
    Code(
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

  def seqOp(state: State, seq: Array[EmitCode], dummy: Boolean): Code[Unit] = {
    val Array(call) = seq
    val hom = state.cb.genFieldThisRef[Boolean]()
    val lastAllele = state.cb.genFieldThisRef[Int]()
    val i = state.cb.genFieldThisRef[Int]()

    def checkSize(a: Code[Int]): Code[Unit] =
      Code.memoize(a, "callstatsagg_check_size_a") { a =>
        (a > state.nAlleles).orEmpty(
          Code._fatal[Unit](const("found allele outside of expected range [0, ")
            .concat(state.nAlleles.toS).concat("]: ").concat(a.toS)))
      }

    Code(
      call.setup,
      call.m.mux(
        Code._empty,
        Code(
          i := 0,
          hom := true,
          lastAllele := -1,
          t.forEachAllele(state.cb, coerce[Int](call.v), { allele: Code[Int] =>
            Code.memoize(allele, "callstatsagg_seqop_allele") { allele =>
              Code(
                checkSize(allele),
                state.updateAlleleCountAtIndex(allele, state.nAlleles, _ + 1),
                (i > 0).orEmpty(hom := hom && allele.ceq(lastAllele)),
                lastAllele := allele,
                i := i + 1)
            }
          }),
          (i > 1) && hom).orEmpty(
          state.updateHomCountAtIndex(lastAllele, state.nAlleles, _ + 1))))
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    val i = state.cb.genFieldThisRef[Int]()
    Code(
      other.nAlleles.cne(state.nAlleles).orEmpty(Code._fatal[Unit]("length mismatch")),
      i := 0,
      Code.whileLoop(i < state.nAlleles,
        state.updateAlleleCountAtIndex(i, state.nAlleles, _ + other.alleleCountAtIndex(i, state.nAlleles)),
        state.updateHomCountAtIndex(i, state.nAlleles, _ + other.homCountAtIndex(i, state.nAlleles)),
        i := i + 1
      )
    )
  }


  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] = {
    val alleleNumber = state.cb.genFieldThisRef[Int]()
    val i = state.cb.genFieldThisRef[Int]()
    val x = state.cb.genFieldThisRef[Int]()
    srvb.addBaseStruct(CallStatsState.resultType, {
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
