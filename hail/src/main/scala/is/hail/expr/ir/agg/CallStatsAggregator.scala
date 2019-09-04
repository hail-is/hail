package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitFunctionBuilder, EmitMethodBuilder, EmitTriplet}
import is.hail.expr.types.physical._
import is.hail.io.{BufferSpec, CodecSpec, InputBuffer, OutputBuffer, PackCodecSpec2}
import is.hail.stats.CallStats
import is.hail.utils._

import scala.language.existentials


object CallStatsState {
  val callStatsInternalArrayType = PArray(PInt32Required, required = true)
  val stateType: PTuple = PTuple(true, callStatsInternalArrayType, callStatsInternalArrayType)

  val resultType = PStruct(
    "AC" -> PArray(PInt32()),
    "AF" -> PArray(PFloat64()),
    "AN" -> PInt32(),
    "homozygote_count" -> PArray(PInt32()))

  assert(resultType.virtualType == CallStats.schema)
}

class CallStatsState(val fb: EmitFunctionBuilder[_]) extends PointerBasedRVAState {
  val alleleCountsOffset: Code[Long] = CallStatsState.stateType.fieldOffset(off, 0)
  val homCountsOffset: Code[Long] = CallStatsState.stateType.fieldOffset(off, 1)
  val alleleCounts: Code[Long] = CallStatsState.stateType.loadField(off, 0)
  val homCounts: Code[Long] = CallStatsState.stateType.loadField(off, 1)
  val nAlleles: ClassFieldRef[Int] = fb.newField[Int]
  private val addr = fb.newField[Long]

  val loadNAlleles: Code[Unit] = nAlleles := CallStatsState.callStatsInternalArrayType.loadLength(alleleCounts)


  // unused but extremely useful for debugging if something goes wrong
  def dump(tag: String): Code[Unit] = {
    val i = fb.newField[Int]
    Code(
      Code._println(s"at tag $tag"),
      i := 0,
      Code.whileLoop(i < nAlleles,
        Code._println(const("at i=").concat(i.toS).concat(", AC=").concat(alleleCountAtIndex(i, nAlleles).toS).concat(", HOM=").concat(homCountAtIndex(i, nAlleles).toS)),
        i := i + 1)
    )
  }

  override def load(regionLoader: Code[Region] => Code[Unit], src: Code[Long]): Code[Unit] = Code(
    super.load(regionLoader, src),
    loadNAlleles
  )

  def alleleCountAtIndex(idx: Code[Int], length: Code[Int]): Code[Int] =
    Region.loadInt(CallStatsState.callStatsInternalArrayType.loadElement(region, alleleCounts, length, idx))

  def updateAlleleCountAtIndex(idx: Code[Int], length: Code[Int], updater: Code[Int] => Code[Int]): Code[Unit] = Code(
    addr := CallStatsState.callStatsInternalArrayType.loadElement(region, alleleCounts, length, idx),
    Region.storeInt(addr, updater(Region.loadInt(addr)))
  )

  def homCountAtIndex(idx: Code[Int], length: Code[Int]): Code[Int] =
    Region.loadInt(CallStatsState.callStatsInternalArrayType.loadElement(region, homCounts, length, idx))


  def updateHomCountAtIndex(idx: Code[Int], length: Code[Int], updater: Code[Int] => Code[Int]): Code[Unit] = Code(
    addr := CallStatsState.callStatsInternalArrayType.loadElement(region, homCounts, length, idx),
    Region.storeInt(addr, updater(Region.loadInt(addr)))
  )

  def serialize(codec: BufferSpec): Code[OutputBuffer] => Code[Unit] = {
    PackCodecSpec2(CallStatsState.stateType, codec).buildEmitEncoderF[Long](CallStatsState.stateType, fb)(region, off, _)
  }

  def deserialize(codec: BufferSpec): Code[InputBuffer] => Code[Unit] = {
    val (decType, dec) = PackCodecSpec2(CallStatsState.stateType, codec)
      .buildEmitDecoderF[Long](CallStatsState.stateType.virtualType, fb)
    assert(decType == CallStatsState.stateType)

    { ib: Code[InputBuffer] =>
      Code(
        off := dec(region, ib),
        loadNAlleles)
    }
  }

  def copyFromAddress(src: Code[Long]): Code[Unit] = {
    Code(
      off := StagedRegionValueBuilder.deepCopy(fb, region, CallStatsState.stateType, src),
      loadNAlleles
    )
  }
}

class CallStatsAggregator(t: PCall) extends StagedAggregator {

  type State = CallStatsState

  def resultType: PStruct = CallStatsState.resultType

  def createState(fb: EmitFunctionBuilder[_]): State = new CallStatsState(fb)

  def initOp(state: State, init: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    val Array(nAlleles) = init
    val addr = state.fb.newField[Long]
    val n = state.fb.newField[Int]
    val i = state.fb.newField[Int]
    Code(
      nAlleles.setup,
      nAlleles.m.mux(
        Code._fatal("call_stats: n_alleles may not be missing"),
        Code(
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
    ).asInstanceOf[Code[Unit]]
  }

  def seqOp(state: State, seq: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    val Array(call) = seq
    val hom = state.fb.newField[Boolean]
    val lastAllele = state.fb.newField[Int]
    val i = state.fb.newField[Int]

    def checkSize(a: Code[Int]): Code[Unit] = (a > state.nAlleles).orEmpty(
      Code._fatal(const("found allele outside of expected range [0, ")
        .concat(state.nAlleles.toS).concat("]: ").concat(a.toS)))

    Code(
      call.setup,
      call.m.mux(
        Code._empty,
        Code(
          i := 0,
          hom := true,
          lastAllele := -1,
          t.forEachAllele(state.fb, coerce[Int](call.v), { allele: Code[Int] =>
            Code(
              checkSize(allele),
              state.updateAlleleCountAtIndex(allele, state.nAlleles, _ + 1),
              (i > 0).orEmpty(hom := hom && allele.ceq(lastAllele)),
              lastAllele := allele,
              i := i + 1
            )
          }),
          (i > 1) && hom).orEmpty(
          state.updateHomCountAtIndex(lastAllele, state.nAlleles, _ + 1))
      )
    )
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    val i = state.fb.newField[Int]
    Code(
      other.nAlleles.cne(state.nAlleles).orEmpty(Code._fatal("length mismatch")),
      i := 0,
      Code.whileLoop(i < state.nAlleles,
        state.updateAlleleCountAtIndex(i, state.nAlleles, _ + other.alleleCountAtIndex(i, state.nAlleles)),
        state.updateHomCountAtIndex(i, state.nAlleles, _ + other.homCountAtIndex(i, state.nAlleles)),
        i := i + 1
      )
    )
  }


  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] = {
    val alleleNumber = state.fb.newField[Int]
    val i = state.fb.newField[Int]
    val x = state.fb.newField[Int]
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
