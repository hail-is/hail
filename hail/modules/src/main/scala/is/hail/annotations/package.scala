package is.hail

import is.hail.utils.Muple

package annotations {
  class AnnotationPathException(msg: String = "") extends Exception(msg)
}

package object annotations {

  type Annotation = Any

  type Deleter = (Annotation) => Annotation

  type Querier = (Annotation) => Any

  type Inserter = (Annotation, Any) => Annotation

  type Assigner = (Annotation, Any) => Annotation

  type Merger = (Annotation, Annotation) => Annotation

  type Filterer = (Annotation) => Annotation

  type UnsafeInserter = (Region, Long, RegionValueBuilder, () => Unit) => Unit

  type JoinedRegionValue = Muple[RegionValue, RegionValue]
}
