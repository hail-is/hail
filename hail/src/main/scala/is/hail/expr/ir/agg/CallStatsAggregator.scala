package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitMethodBuilder, EmitTriplet}
import is.hail.expr.types.physical._
import is.hail.io.{CodecSpec, InputBuffer, OutputBuffer}
import is.hail.stats.CallStats
import is.hail.utils._


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

case class CallStatsState(mb: EmitMethodBuilder) extends PointerBasedRVAState {
  val alleleCountsOffset: Code[Long] = CallStatsState.stateType.fieldOffset(off, 0)
  val homCountsOffset: Code[Long] = CallStatsState.stateType.fieldOffset(off, 1)
  val alleleCounts: Code[Long] = CallStatsState.stateType.loadField(off, 0)
  val homCounts: Code[Long] = CallStatsState.stateType.loadField(off, 1)
  val nAlleles: Code[Int] = CallStatsState.callStatsInternalArrayType.loadLength(alleleCounts)
  private val addr = mb.newField[Long]

  // unused but extremely useful for debugging if something goes wrong
  def dump(tag: String): Code[Unit] = {
    val i = mb.newField[Int]
    Code(
      Code._println(s"at tag $tag"),
      i := 0,
      Code.whileLoop(i < nAlleles,
        Code._println(const("at i=").concat(i.toS).concat(", AC=").concat(alleleCountAtIndex(i, nAlleles).toS).concat(", HOM=").concat(homCountAtIndex(i, nAlleles).toS)),
        i := i + 1)
    )
  }

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

  def serialize(codec: CodecSpec): Code[OutputBuffer] => Code[Unit] = {
    codec.buildEmitEncoderF[Long](CallStatsState.stateType, CallStatsState.stateType, mb.fb)(region, off, _)
  }

  def deserialize(codec: CodecSpec): Code[InputBuffer] => Code[Unit] = {
    off := codec.buildEmitDecoderF(CallStatsState.stateType, CallStatsState.stateType, mb.fb)(region, _)
  }

  def copyFromAddress(src: Code[Long]): Code[Unit] = {
    StagedRegionValueBuilder.deepCopy(er, CallStatsState.stateType, region.loadAddress(src), off)
  }
}

class CallStatsAggregator(t: PCall) extends StagedAggregator {

  type State = CallStatsState

  def resultType: PStruct = CallStatsState.resultType

  def createState(mb: EmitMethodBuilder): State = CallStatsState(mb)

  def initOp(state: State, init: Array[EmitTriplet], dummy: Boolean): Code[Unit] = {
    val Array(nAlleles) = init
    val addr = state.mb.newLocal[Long]
    val n = state.mb.newLocal[Int]
    val i = state.mb.newField[Int]
    Code(
      nAlleles.setup,
      nAlleles.m.mux(
        Code._fatal("call_stats: n_alleles may not be missing"),
        Code(
          n := coerce[Int](nAlleles.v),
          state.off := state.region.allocate(CallStatsState.stateType.alignment, CallStatsState.stateType.byteSize),
          addr := CallStatsState.callStatsInternalArrayType.allocate(state.region, n),
          CallStatsState.callStatsInternalArrayType.stagedInitialize(addr, n, zero = true),
          Region.storeAddress(state.alleleCountsOffset, addr),
          addr := CallStatsState.callStatsInternalArrayType.allocate(state.region, n),
          CallStatsState.callStatsInternalArrayType.stagedInitialize(addr, n, zero = true),
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
    val len = state.mb.newField[Int]
    val hom = state.mb.newLocal[Boolean]
    val lastAllele = state.mb.newLocal[Int]
    val i = state.mb.newField[Int]

    def checkSize(a: Code[Int]): Code[Unit] = (a > len).orEmpty(
      Code._fatal(const("found allele outside of expected range [0, ")
        .concat(len.toS).concat("]: ").concat(a.toS)))

    Code(
      call.setup,
      call.m.mux(
        Code._empty,
        Code(
          len := state.nAlleles,
          i := 0,
          hom := true,
          lastAllele := -1,
          t.forEachAllele(state.mb, coerce[Int](call.v), { allele: Code[Int] =>
            Code(
              checkSize(allele),
              state.updateAlleleCountAtIndex(allele, len, _ + 1),
              (i > 0).orEmpty(hom := hom && allele.ceq(lastAllele)),
              lastAllele := allele,
              i := i + 1
            )
          }),
          (i > 1) && hom).orEmpty(
          state.updateHomCountAtIndex(lastAllele, len, _ + 1))
      )
    )
  }

  def combOp(state: State, other: State, dummy: Boolean): Code[Unit] = {
    val len = state.mb.newField[Int]
    val i = state.mb.newField[Int]
    Code(
      len := state.nAlleles,
      other.nAlleles.cne(len).orEmpty(Code._fatal("length mismatch")),
      i := 0,
      Code.whileLoop(i < len,
        state.updateAlleleCountAtIndex(i, len, _ + other.alleleCountAtIndex(i, len)),
        state.updateHomCountAtIndex(i, len, _ + other.homCountAtIndex(i, len)),
        i := i + 1
      )
    )
  }


  def result(state: State, srvb: StagedRegionValueBuilder, dummy: Boolean): Code[Unit] = {
    val alleleNumber = state.mb.newLocal[Int]
    val i = state.mb.newLocal[Int]
    val nAlleles = state.mb.newLocal[Int]
    val x = state.mb.newLocal[Int]
    srvb.addBaseStruct(CallStatsState.resultType, {
      srvb =>
        Code(
          srvb.start(),
          alleleNumber := 0,
          nAlleles := state.nAlleles,
          srvb.addArray(resultType.fieldType("AC").asInstanceOf[PArray], {
            srvb =>
              Code(
                srvb.start(nAlleles),
                i := 0,
                Code.whileLoop(i < nAlleles,
                  x := state.alleleCountAtIndex(i, nAlleles),
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
                  srvb.start(nAlleles),
                  i := 0,
                  Code.whileLoop(i < nAlleles,
                    x := state.alleleCountAtIndex(i, nAlleles),
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
                srvb.start(nAlleles),
                i := 0,
                Code.whileLoop(i < nAlleles,
                  x := state.homCountAtIndex(i, nAlleles),
                  srvb.addInt(x),
                  srvb.advance(),
                  i := i + 1))
          })).asInstanceOf[Code[Unit]]
    })
  }
}
