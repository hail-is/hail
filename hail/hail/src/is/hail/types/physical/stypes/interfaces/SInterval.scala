package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s._
import is.hail.asm4s.implicits._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.io.PrefixCoder
import is.hail.types.{RInterval, TypeWithRequiredness}
import is.hail.types.physical.stypes.{EmitType, SType, SValue}
import is.hail.types.physical.stypes.primitives.SInt64Value

trait SInterval extends SType {
  def pointType: SType
  def pointEmitType: EmitType

  override def _typeWithRequiredness: TypeWithRequiredness = {
    val pt = pointEmitType.typeWithRequiredness.r
    RInterval(pt, pt)
  }
}

trait SIntervalValue extends SValue {
  override def st: SInterval

  override def prefixCode(cb: EmitCodeBuilder, pc: Value[PrefixCoder]): Unit = {
    def encodeEndpoint(iec: IEmitCode) =
      iec.consume(
        cb,
        if (!st.pointEmitType.required) pc.encodeMissing(cb),
        { (sv) =>
          if (!st.pointEmitType.required) pc.encodePresent(cb)
          sv.prefixCode(cb, pc)
        },
      )

    encodeEndpoint(loadStart(cb))
    // we need to take !includesStart since a closed left endpoint sorts before an open one
    pc.encodeBool(cb, !includesStart)
    encodeEndpoint(loadEnd(cb))
    // equivalent right endpoints sort naturally according to includeEnd
    pc.encodeBool(cb, includesEnd)
  }

  def includesStart: Value[Boolean]

  def includesEnd: Value[Boolean]

  def loadStart(cb: EmitCodeBuilder): IEmitCode

  def startDefined(cb: EmitCodeBuilder): Value[Boolean]

  def loadEnd(cb: EmitCodeBuilder): IEmitCode

  def endDefined(cb: EmitCodeBuilder): Value[Boolean]

  override def sizeToStoreInBytes(cb: EmitCodeBuilder): SInt64Value = {
    val pIntervalSize = this.st.storageType().byteSize
    val sizeSoFar = cb.newLocal[Long]("sstackstruct_size_in_bytes", pIntervalSize)

    loadStart(cb).consume(
      cb,
      {},
      sv => cb.assign(sizeSoFar, sizeSoFar + sv.sizeToStoreInBytes(cb).value),
    )

    loadEnd(cb).consume(
      cb,
      {},
      sv => cb.assign(sizeSoFar, sizeSoFar + sv.sizeToStoreInBytes(cb).value),
    )

    new SInt64Value(sizeSoFar)
  }

  def isEmpty(cb: EmitCodeBuilder): Value[Boolean] = {
    val gt = cb.emb.ecb.getOrderingFunction(st.pointType, CodeOrdering.Gt())
    val gteq = cb.emb.ecb.getOrderingFunction(st.pointType, CodeOrdering.Gteq())

    val start = cb.memoize(loadStart(cb), "start")
    val end = cb.memoize(loadEnd(cb), "end")
    val empty = cb.newLocal[Boolean]("is_empty")
    cb.if_(
      includesStart && includesEnd,
      cb.assign(empty, gt(cb, start, end)),
      cb.assign(empty, gteq(cb, start, end)),
    )
    empty
  }
}
