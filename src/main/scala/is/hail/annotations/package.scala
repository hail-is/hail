package is.hail

package object annotations {

  class AnnotationPathException(msg: String = "") extends Exception(msg)

  type  Annotation = Any

  type Deleter = (Annotation) => Annotation

  type Querier = (Annotation) => Any

  type Inserter = (Annotation, Any) => Annotation

  type Assigner = (Annotation, Any) => Annotation

  type Merger = (Annotation, Annotation) => Annotation

  type Filterer = (Annotation) => Annotation

  type UnsafeInserter = (Region, Long, RegionValueBuilder, () => Unit) => Unit

  case class Muple[T, U](var _1: T, var _2: U) {
    def set(newLeft: T, newRight: U) {
      _1 = newLeft
      _2 = newRight
    }
    def left = _1
    def right = _2
  }
  type JoinedRegionValue = Muple[RegionValue, RegionValue]
}
