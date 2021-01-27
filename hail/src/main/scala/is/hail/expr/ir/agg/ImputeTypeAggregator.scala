package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder}
import is.hail.types.{RPrimitive, VirtualTypeWithReq}
import is.hail.types.physical._
import is.hail.types.virtual.{TInt32, TString, Type}
import is.hail.utils._

import scala.language.existentials

object ImputeTypeState {
  val resultType = PCanonicalStruct(required = true,
    "anyNonMissing" -> PBooleanRequired,
    "allDefined" -> PBooleanRequired,
    "supportsBool" -> PBooleanRequired,
    "supportsInt32" -> PBooleanRequired,
    "supportsInt64" -> PBooleanRequired,
    "supportsFloat64" -> PBooleanRequired)

  def matchBoolean(x: String): Boolean = try {
    x.toBoolean
    true
  } catch {
    case e: IllegalArgumentException => false
  }

  def matchInt32(x: String): Boolean = try {
    Integer.parseInt(x)
    true
  } catch {
    case e: IllegalArgumentException => false
  }


  def matchInt64(x: String): Boolean = try {
    java.lang.Long.parseLong(x)
    true
  } catch {
    case e: IllegalArgumentException => false
  }

  def matchFloat64(x: String): Boolean = try {
    java.lang.Double.parseDouble(x)
    true
  } catch {
    case e: IllegalArgumentException => false
  }

}

class ImputeTypeState(kb: EmitClassBuilder[_]) extends PrimitiveRVAState(Array(VirtualTypeWithReq(TInt32,RPrimitive()).setRequired(true)), kb) {

  private val repr = fields(0)._2.asInstanceOf[Settable[Int]]

  def getAnyNonMissing: Code[Boolean] = (repr.load() & 1).cne(0)

  def getAllDefined: Code[Boolean] = (repr.load() & 1 << 1).cne(0)

  def getSupportsBool: Code[Boolean] = (repr.load() & 1 << 2).cne(0)

  def getSupportsI32: Code[Boolean] = (repr.load() & 1 << 3).cne(0)

  def getSupportsI64: Code[Boolean] = (repr.load() & 1 << 4).cne(0)

  def getSupportsF64: Code[Boolean] = (repr.load() & 1 << 5).cne(0)

  private def setRepr(cb: EmitCodeBuilder,
    anyNonMissing: Code[Boolean],
    allDefined: Code[Boolean],
    supportsBool: Code[Boolean],
    supportsI32: Code[Boolean],
    supportsI64: Code[Boolean],
    supportsF64: Code[Boolean]
  ): Unit = {

    cb += repr.store(anyNonMissing.toI
      | (allDefined.toI << 1)
      | (supportsBool.toI << 2)
      | (supportsI32.toI << 3)
      | (supportsI64.toI << 4)
      | (supportsF64.toI << 5)
    )
  }

  def initialize(cb: EmitCodeBuilder): Unit = {
    setRepr(cb, false, true, true, true, true, true)
  }

  def seqOp(cb: EmitCodeBuilder, ec: EmitCode): Unit = {
    ec.toI(cb)
      .consume(cb,
        cb += repr.store(repr & (~(1 << 1))),
        { case (pc: PStringCode) =>
          val s = cb.newLocal[String]("impute_type_agg_seq_str")
          cb.assign(s, pc.loadString())

          setRepr(cb,
            true,
            getAllDefined,
            getSupportsBool && Code.invokeScalaObject1[String, Boolean](ImputeTypeState.getClass, "matchBoolean", s),
            getSupportsI32 && Code.invokeScalaObject1[String, Boolean](ImputeTypeState.getClass, "matchInt32", s),
            getSupportsI64 && Code.invokeScalaObject1[String, Boolean](ImputeTypeState.getClass, "matchInt64", s),
            getSupportsF64 && Code.invokeScalaObject1[String, Boolean](ImputeTypeState.getClass, "matchFloat64", s)
          )
        }
      )
  }

  def combOp(cb: EmitCodeBuilder, other: ImputeTypeState): Unit = {
    setRepr(cb,
      getAnyNonMissing || other.getAnyNonMissing,
      getAllDefined && other.getAllDefined,
      getSupportsBool && other.getSupportsBool,
      getSupportsI32 && other.getSupportsI32,
      getSupportsI64 && other.getSupportsI64,
      getSupportsF64 && other.getSupportsF64
    )
  }
}

class ImputeTypeAggregator() extends StagedAggregator {

  val initOpTypes: Seq[Type] = FastSeq()
  val seqOpTypes: Seq[Type] = FastSeq(TString)

  type State = ImputeTypeState

  def resultType: PStruct = ImputeTypeState.resultType

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 0)
    state.initialize(cb)
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(s) = seq

    state.seqOp(cb, s)
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    state.combOp(cb, other)
  }

  protected def _storeResult(cb: EmitCodeBuilder, state: State, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit = {
    val rt = ImputeTypeState.resultType
    assert(pt == rt)
    cb += rt.stagedInitialize(addr, setMissing = false)
    rt.types(0).storeAtAddress(cb, rt.fieldOffset(addr, 0), region, PCode(PBooleanRequired, state.getAnyNonMissing), deepCopy = true)
    rt.types(1).storeAtAddress(cb, rt.fieldOffset(addr, 0), region, PCode(PBooleanRequired, state.getAllDefined), deepCopy = true)
    rt.types(2).storeAtAddress(cb, rt.fieldOffset(addr, 0), region, PCode(PBooleanRequired, state.getSupportsBool), deepCopy = true)
    rt.types(3).storeAtAddress(cb, rt.fieldOffset(addr, 0), region, PCode(PBooleanRequired, state.getSupportsI32), deepCopy = true)
    rt.types(4).storeAtAddress(cb, rt.fieldOffset(addr, 0), region, PCode(PBooleanRequired, state.getSupportsI64), deepCopy = true)
    rt.types(5).storeAtAddress(cb, rt.fieldOffset(addr, 0), region, PCode(PBooleanRequired, state.getSupportsF64), deepCopy = true)
  }
}
