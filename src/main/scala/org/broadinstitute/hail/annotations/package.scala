package org.broadinstitute.hail

import org.apache.spark.sql.Row

package object annotations {

  class AnnotationPathException(msg: String = "") extends Exception(msg)

  class NoSuchAnnotationException(msg: String = "") extends Exception(msg)

  type Annotation = Any

  type Deleter = (Annotation) => Annotation

  type Querier = (Annotation) => Option[Any]

  type Inserter = (Annotation, Option[Any]) => Annotation
}
