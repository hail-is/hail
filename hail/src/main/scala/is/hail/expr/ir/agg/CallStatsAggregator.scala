package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitContext, IEmitCode}
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.types.physical._
import is.hail.types.physical.stypes.EmitType
import is.hail.types.physical.stypes.concrete.SBaseStructPointer
import is.hail.types.virtual.{TCall, TInt32, Type}
import is.hail.types.physical.stypes.interfaces._
import is.hail.utils._

import scala.language.existentials


object CallStatsState {
  val callStatsInternalArrayType = PCanonicalArray(PInt32Required, required = true)
  val stateType: PCanonicalTuple = PCanonicalTuple(true, callStatsInternalArrayType, callStatsInternalArrayType)

  val resultPType = PCanonicalStruct(required = false,
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

  def loadNAlleles(cb: EmitCodeBuilder): Unit = {
    cb.assign(nAlleles, CallStatsState.callStatsInternalArrayType.loadLength(alleleCounts))
  }

  // unused but extremely useful for debugging if something goes wrong
  def dump(cb: CodeBuilderLike, tag: String): Unit = {
    val i = cb.newLocal[Int]("i")
    cb += Code._println(s"at tag $tag")
    cb.for_(
      setup = cb.assign(i, 0),
      cond = i < nAlleles,
      incr = cb.assign(i, i + 1),
      body = {
        cb += Code._println(
          const("at i=") + i.toS +
            ", AC=" + alleleCountAtIndex(i, nAlleles).toS +
            ", HOM=" + homCountAtIndex(i, nAlleles).toS
        )
      }
    )
  }

  override def load(cb: EmitCodeBuilder, regionLoader: (EmitCodeBuilder, Value[Region]) => Unit, src: Value[Long]): Unit = {
    super.load(cb, regionLoader, src)
    loadNAlleles(cb)
  }

  def alleleCountAtIndex(idx: Code[Int], length: Code[Int]): Code[Int] =
    Region.loadInt(CallStatsState.callStatsInternalArrayType.loadElement(alleleCounts, length, idx))

  def updateAlleleCountAtIndex(cb: EmitCodeBuilder, idx: Code[Int], length: Code[Int], updater: Code[Int] => Code[Int]): Unit = {
    cb.assign(addr, CallStatsState.callStatsInternalArrayType.loadElement(alleleCounts, length, idx))
    cb += Region.storeInt(addr, updater(Region.loadInt(addr)))
  }

  def homCountAtIndex(idx: Code[Int], length: Code[Int]): Code[Int] =
    Region.loadInt(CallStatsState.callStatsInternalArrayType.loadElement(homCounts, length, idx))


  def updateHomCountAtIndex(cb: EmitCodeBuilder, idx: Code[Int], length: Code[Int], updater: Code[Int] => Code[Int]): Unit = {
    cb.assign(addr, CallStatsState.callStatsInternalArrayType.loadElement(homCounts, length, idx))
    cb += Region.storeInt(addr, updater(Region.loadInt(addr)))
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = { (cb, ob) =>
    val codecSpec = TypedCodecSpec(CallStatsState.stateType, codec)
    codecSpec.encodedType.buildEncoder(CallStatsState.stateType.sType, kb)
      .apply(cb, CallStatsState.stateType.loadCheapSCode(cb, off), ob)
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    { (cb: EmitCodeBuilder, ib: Value[InputBuffer]) =>
      val codecSpec = TypedCodecSpec(CallStatsState.stateType, codec)
      val decValue = codecSpec.encodedType.buildDecoder(CallStatsState.stateType.virtualType, kb)
        .apply(cb, region, ib)

      cb.assign(off, CallStatsState.stateType.store(cb, region, decValue, deepCopy = false))
      loadNAlleles(cb)
    }
  }

  def copyFromAddress(cb: EmitCodeBuilder, src: Value[Long]): Unit = {
    cb.assign(off, CallStatsState.stateType.store(cb, region, CallStatsState.stateType.loadCheapSCode(cb, src), deepCopy = true))
    loadNAlleles(cb)
  }
}

class CallStatsAggregator extends StagedAggregator {

  type State = CallStatsState

  val resultEmitType: EmitType = EmitType(SBaseStructPointer(CallStatsState.resultPType), true)
  val resultStorageType = resultEmitType.storageType.asInstanceOf[PBaseStruct]

  val initOpTypes: Seq[Type] = FastSeq(TInt32)
  val seqOpTypes: Seq[Type] = FastSeq(TCall)

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    val Array(nAlleles) = init
    val addr = state.kb.genFieldThisRef[Long]()
    val n = state.kb.genFieldThisRef[Int]()
    val i = state.kb.genFieldThisRef[Int]()

    nAlleles.toI(cb)
      .consume(cb,
        cb += Code._fatal[Unit]("hl.agg.call_stats: n_alleles may not be missing"),
        { sc =>
          cb.assign(n, sc.asInt.value)
          cb.assign(state.nAlleles, n)
          cb.assign(state.off, state.region.allocate(CallStatsState.stateType.alignment, CallStatsState.stateType.byteSize))
          cb.assign(addr, CallStatsState.callStatsInternalArrayType.allocate(state.region, n))
          CallStatsState.callStatsInternalArrayType.stagedInitialize(cb, addr, n)
          cb += Region.storeAddress(state.alleleCountsOffset, addr)
          cb.assign(addr, CallStatsState.callStatsInternalArrayType.allocate(state.region, n))
          CallStatsState.callStatsInternalArrayType.stagedInitialize(cb, addr, n)
          cb += Region.storeAddress(state.homCountsOffset, addr)
          cb.assign(i, 0)
          cb.while_(i < n,
            {
              state.updateAlleleCountAtIndex(cb, i, n, _ => 0)
              state.updateHomCountAtIndex(cb, i, n, _ => 0)
              cb.assign(i, i + 1)
            })
        })
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(call) = seq

    call.toI(cb).consume(cb, {
      /* do nothing if missing */
    }, { case call: SCallValue =>
      val hom = cb.newLocal[Boolean]("hom", true)
      val lastAllele = cb.newLocal[Int]("lastAllele", -1)
      val i = cb.newLocal[Int]("i", 0)
      call.forEachAllele(cb) { allele: Value[Int] =>
        cb.ifx(allele > state.nAlleles,
          cb._fatal(const("hl.agg.call_stats: found allele outside of expected range [0, ")
            .concat(state.nAlleles.toS).concat("]: ").concat(allele.toS)))
        state.updateAlleleCountAtIndex(cb, allele, state.nAlleles, _ + 1)
        cb.ifx(i > 0, cb.assign(hom, hom && allele.ceq(lastAllele)))
        cb.assign(lastAllele, allele)
        cb.assign(i, i + 1)
      }

      cb.ifx((i > 1) && hom, {
        state.updateHomCountAtIndex(cb, lastAllele, state.nAlleles, _ + 1)
      })
    })
  }

  protected def _combOp(ctx: ExecuteContext, cb: EmitCodeBuilder, state: CallStatsState, other: CallStatsState): Unit = {
    val i = state.kb.genFieldThisRef[Int]()
    cb.ifx(other.nAlleles.cne(state.nAlleles),
      cb += Code._fatal[Unit]("hl.agg.call_stats: length mismatch"),
      {
        cb.assign(i, 0)
        cb.while_(i < state.nAlleles,
          {
            state.updateAlleleCountAtIndex(cb, i, state.nAlleles, _ + other.alleleCountAtIndex(i, state.nAlleles))
            state.updateHomCountAtIndex(cb, i, state.nAlleles, _ + other.homCountAtIndex(i, state.nAlleles))
            cb.assign(i, i + 1)
          })
      })
  }


  protected def _result(cb: EmitCodeBuilder, state: State, region: Value[Region]): IEmitCode = {
    val rt = CallStatsState.resultPType
    val addr = cb.memoize(rt.allocate(region), "call_stats_aggregator_result_addr")
    rt.stagedInitialize(cb, addr, setMissing = false)
    val alleleNumber = cb.newLocal[Int]("callstats_result_alleleNumber", 0)

    val acType = resultStorageType.fieldType("AC").asInstanceOf[PCanonicalArray]

    // this is a little weird - computing AC has the side effect of updating AN
    val ac = acType.constructFromElements(cb, region, state.nAlleles, deepCopy = true) { (cb, i) =>
      val acAtIndex = cb.newLocal[Int]("callstats_result_acAtIndex", state.alleleCountAtIndex(i, state.nAlleles))
      cb.assign(alleleNumber, alleleNumber + acAtIndex)
      IEmitCode.present(cb, primitive(acAtIndex))
    }

    acType.storeAtAddress(cb, rt.fieldOffset(addr, "AC"), region, ac, deepCopy = false)

    cb.ifx(alleleNumber.ceq(0),
      rt.setFieldMissing(cb, addr, "AF"),
      {
        val afType = resultStorageType.fieldType("AF").asInstanceOf[PCanonicalArray]
        val af = afType.constructFromElements(cb, region, state.nAlleles, deepCopy = true) { (cb, i) =>
          val acAtIndex = cb.newLocal[Int]("callstats_result_acAtIndex", state.alleleCountAtIndex(i, state.nAlleles))
          IEmitCode.present(cb, primitive(cb.memoize(acAtIndex.toD / alleleNumber.toD)))
        }
        afType.storeAtAddress(cb, rt.fieldOffset(addr, "AF"), region, af, deepCopy = false)
      })

    val anType = resultStorageType.fieldType("AN")
    val an = primitive(alleleNumber)
    anType.storeAtAddress(cb, rt.fieldOffset(addr, "AN"), region, an, deepCopy = false)


    val homCountType = resultStorageType.fieldType("homozygote_count").asInstanceOf[PCanonicalArray]
    val homCount = homCountType.constructFromElements(cb, region, state.nAlleles, deepCopy = true) { (cb, i) =>
      val homCountAtIndex = cb.newLocal[Int]("callstats_result_homCountAtIndex", state.homCountAtIndex(i, state.nAlleles))
      IEmitCode.present(cb, primitive(homCountAtIndex))
    }

    homCountType.storeAtAddress(cb, rt.fieldOffset(addr, "homozygote_count"), region, homCount, deepCopy = false)
    IEmitCode.present(cb, rt.loadCheapSCode(cb, addr))
  }
}
