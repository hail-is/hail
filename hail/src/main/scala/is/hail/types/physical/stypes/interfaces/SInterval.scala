package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.{RInterval, TypeWithRequiredness}
import is.hail.types.physical.PInterval
import is.hail.types.physical.stypes.primitives.SInt64Value
import is.hail.types.physical.stypes.{EmitType, SCode, SType, SValue}

trait SInterval extends SType {
  def pointType: SType
  def pointEmitType: EmitType
  override def _typeWithRequiredness: TypeWithRequiredness = {
    val pt = pointEmitType.typeWithRequiredness.r
    RInterval(pt, pt)
  }
}

trait SIntervalValue extends SValue {
  def st: SInterval

  def includesStart(): Value[Boolean]

  def includesEnd(): Value[Boolean]

  def loadStart(cb: EmitCodeBuilder): IEmitCode

  def startDefined(cb: EmitCodeBuilder): Value[Boolean]

  def loadEnd(cb: EmitCodeBuilder): IEmitCode

  def endDefined(cb: EmitCodeBuilder): Value[Boolean]

  def isEmpty(cb: EmitCodeBuilder): Value[Boolean]

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value = {
    val storageType = st.storageType().asInstanceOf[PInterval]

    val pIntervalSize = this.st.storageType().byteSize
    val sizeSoFar = cb.newLocal[Long]("sstackstruct_size_in_bytes", pIntervalSize)

    loadStart(cb).consume(cb, {}, {sv =>
      sv.sizeToStoreInBytes(cb)
    })

    loadEnd(cb).consume(cb, {}, {sv =>
      sv.sizeToStoreInBytes(cb)
    })

    new SInt64Value(???)
  }
}
