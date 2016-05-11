package org.broadinstitute.hail

import org.broadinstitute.hail.expr._

package object annotations {

  class AnnotationPathException(msg: String = "") extends Exception(msg)

  type Annotation = Any

  type Deleter = (Annotation) => Annotation

  type Querier = (Annotation) => Option[Any]

  type Inserter = (Annotation, Option[Any]) => Annotation

  type Assigner = (Annotation, Option[Any]) => Annotation
}
