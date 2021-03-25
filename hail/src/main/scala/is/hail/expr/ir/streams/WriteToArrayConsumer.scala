package is.hail.expr.ir.streams

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitMethodBuilder, IEmitCode, StagedArrayBuilder}
import is.hail.types.physical.stypes.SType
import is.hail.types.physical.stypes.concrete.SIndexablePointerCode
import is.hail.types.physical.{PCanonicalArray, SingleCodeType}
import is.hail.utils._


trait ToArrayConsumer {
  def pushElement(cb: EmitCodeBuilder, element: EmitCode): Unit

  def finish(cb: EmitCodeBuilder): IEmitCode
}

class KnownSizeToArrayConsumer(mb: EmitMethodBuilder[_], destRegion: Value[Region], elemType: SType,
  separateRegions: Boolean, eltRegion: Settable[Region]) extends ToArrayConsumer {

  private[this] val pcArray = PCanonicalArray(elemType.canonicalPType(), true)
  private[this] var pushElemF: (EmitCodeBuilder, IEmitCode) => Unit = _
  private[this] var finishF: (EmitCodeBuilder => SIndexablePointerCode) = _

  def init(cb: EmitCodeBuilder, len: Code[Int]): Unit = {
    val (_pushElem, _finish) = pcArray.constructFromFunctions(cb, destRegion, cb.newLocal("toarray_len", len), separateRegions)
    pushElemF = _pushElem
    finishF = _finish
  }

  def pushElement(cb: EmitCodeBuilder, element: EmitCode): Unit = {
    pushElemF.apply(cb, element.toI(cb))
    if (separateRegions)
      cb += eltRegion.clearRegion()

  }

  def finish(cb: EmitCodeBuilder): IEmitCode = {
    if (separateRegions)
      cb += eltRegion.freeRegion()
    IEmitCode.present(cb, finishF(cb))
  }

}

class UnknownSizeToArrayConsumer(mb: EmitMethodBuilder[_], destRegion: Value[Region], elemType: SType,
  separateRegions: Boolean, eltRegion: Settable[Region]) extends ToArrayConsumer {

  private[this] val ab: StagedArrayBuilder = new StagedArrayBuilder(elemType.canonicalPType(), mb, 0)
  private[this] val sct = SingleCodeType.fromSType(elemType)

  def init(cb: EmitCodeBuilder): Unit = {
    cb += ab.clear
    cb += ab.ensureCapacity(const(16))
  }

  def pushElement(cb: EmitCodeBuilder, element: EmitCode): Unit = {
    element.toI(cb).consume(cb,
      ab.addMissing(),
      { sc =>
        ab.add(sct.coercePCode(cb, sc, destRegion, deepCopy = separateRegions).code)
      })
    if (separateRegions)
      cb += eltRegion.clearRegion()
  }

  override def finish(cb: EmitCodeBuilder): IEmitCode = {
    if (separateRegions)
      cb += eltRegion.freeRegion()
    val aTyp = PCanonicalArray(elemType.canonicalPType(), true)
    IEmitCode.present(cb, aTyp.constructFromElements(cb, destRegion, cb.newLocal[Int]("toarray_len", ab.size), deepCopy = false) { (cb, i) =>
      IEmitCode(cb, ab.isMissing(i), sct.loadToPCode(cb, destRegion, ab.apply(i)))
    })
  }
}


class WriteToArrayConsumer(
  mb: EmitMethodBuilder[_],
  outerRegion: Value[Region]
) extends StreamConsumer {

  private[this] var impl: ToArrayConsumer = _

  def init(cb: EmitCodeBuilder, eltType: SType, length: Option[Code[Int]], eltRegion: Settable[Region], separateRegions: Boolean): Unit = {
    length match {
      case Some(knownLen) =>
        val ksc = new KnownSizeToArrayConsumer(mb, outerRegion, eltType, separateRegions, eltRegion)
        ksc.init(cb, knownLen)
        impl = ksc
      case None =>
        val usc = new UnknownSizeToArrayConsumer(mb, outerRegion, eltType, separateRegions, eltRegion)
        usc.init(cb)
        impl = usc
    }

    if (separateRegions)
      cb.assign(eltRegion, Region.stagedCreate(Region.REGULAR, outerRegion.getPool()))
    else
      cb.assign(eltRegion, outerRegion)

  }

  def consumeElement(cb: EmitCodeBuilder, elt: EmitCode): Unit = {
    impl.pushElement(cb, elt)
  }

  val done: CodeLabel = CodeLabel()

  def finish(cb: EmitCodeBuilder): IEmitCode = {
    impl.finish(cb)
  }
}
