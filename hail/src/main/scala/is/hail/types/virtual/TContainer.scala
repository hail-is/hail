package is.hail.types.virtual

import is.hail.annotations.Annotation
import is.hail.types.physical.PContainer

abstract class TContainer extends TIterable {
  override def valuesSimilar(a1: Annotation, a2: Annotation, tolerance: Double, absolute: Boolean)
    : Boolean =
    a1 == a2 || (a1 != null && a2 != null
      && (a1.asInstanceOf[Iterable[_]].size == a2.asInstanceOf[Iterable[_]].size)
      && a1.asInstanceOf[Iterable[_]].zip(a2.asInstanceOf[Iterable[_]])
        .forall { case (e1, e2) => elementType.valuesSimilar(e1, e2, tolerance, absolute) })

  def arrayElementsRepr: TArray
}
