package is.hail.types.physical

import is.hail.annotations.{Annotation, Region}
import is.hail.backend.HailStateManager
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SIndexablePointerValue}
import is.hail.types.physical.stypes.interfaces.SIndexableValue
import is.hail.types.virtual.{TSet, Type}
import is.hail.utils._

object PCanonicalSet {
  def coerceArrayCode(contents: SIndexableValue): SIndexableValue = {
    contents.st match {
      case SIndexablePointer(PCanonicalArray(elt, r)) =>
        PCanonicalSet(elt, r).construct(contents)
    }
  }
}

final case class PCanonicalSet(elementType: PType,  required: Boolean = false) extends PSet with PArrayBackedContainer {
  val arrayRep = PCanonicalArray(elementType, required)

  def setRequired(required: Boolean) = if (required == this.required) this else PCanonicalSet(elementType, required)

  def _asIdent = s"set_of_${elementType.asIdent}"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("PCSet[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

  override def deepRename(t: Type) = deepRenameSet(t.asInstanceOf[TSet])

  private def deepRenameSet(t: TSet) =
    PCanonicalSet(this.elementType.deepRename(t.elementType), this.required)

  override def unstagedStoreJavaObject(sm: HailStateManager, annotation: Annotation, region: Region): Long = {
    val s: IndexedSeq[Annotation] = annotation.asInstanceOf[Set[Annotation]]
      .toFastIndexedSeq
      .sorted(elementType.virtualType.ordering(sm).toOrdering)
    arrayRep.unstagedStoreJavaObject(sm, s, region)
  }

  def construct(_contents: SIndexableValue): SIndexableValue = {
    val contents = _contents.asInstanceOf[SIndexablePointerValue]
    assert(contents.pt.equalModuloRequired(arrayRep), s"\n  contents:  ${ contents.pt }\n  arrayrep: ${ arrayRep }")
    val cont = contents.asInstanceOf[SIndexablePointerValue]
    new SIndexablePointerValue(SIndexablePointer(this), cont.a, cont.length, cont.elementsAddress)
  }

  override def copiedType: PType = {
    val copiedElement = elementType.copiedType
    if (copiedElement.eq(elementType))
      this
    else
      PCanonicalSet(copiedElement, required)
  }
}
