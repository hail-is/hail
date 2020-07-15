package is.hail.expr.ir.agg

import is.hail.annotations.StagedRegionValueBuilder
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder}
import is.hail.types.physical._
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

class ImputeTypeState(kb: EmitClassBuilder[_]) extends PrimitiveRVAState(Array(PInt32Required), kb) {

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
      getAnyNonMissing && other.getAnyNonMissing,
      getAllDefined && other.getAllDefined,
      getSupportsBool && other.getSupportsBool,
      getSupportsI32 && other.getSupportsI32,
      getSupportsI64 && other.getSupportsI64,
      getSupportsF64 && other.getSupportsF64
    )
  }
}

class ImputeTypeAggregator(st: PType) extends StagedAggregator {

  val initOpTypes: Seq[PType] = FastSeq()
  val seqOpTypes: Seq[PType] = FastSeq(st)

  type State = ImputeTypeState

  def resultType: PStruct = ImputeTypeState.resultType

  protected def _initOp(cb: EmitCodeBuilder, state: State, init: Array[EmitCode]): Unit = {
    assert(init.length == 0)
    state.initialize(cb)
  }

  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
    val Array(s) = seq
    assert(s.pt == st)

    state.seqOp(cb, s)
  }

  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit = {
    state.combOp(cb, other)
  }

  protected def _result(cb: EmitCodeBuilder, state: State, srvb: StagedRegionValueBuilder): Unit = {
    cb += srvb.addBaseStruct(ImputeTypeState.resultType, { srvb =>
      Code(FastSeq(
        srvb.start(),
        srvb.addBoolean(state.getAnyNonMissing),
        srvb.advance(),
        srvb.addBoolean(state.getAllDefined),
        srvb.advance(),
        srvb.addBoolean(state.getSupportsBool),
        srvb.advance(),
        srvb.addBoolean(state.getSupportsI32),
        srvb.advance(),
        srvb.addBoolean(state.getSupportsI64),
        srvb.advance(),
        srvb.addBoolean(state.getSupportsF64)))
    })
  }
}
