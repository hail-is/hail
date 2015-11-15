package org.broadinstitute.hail

package object annotations {
  type AnnotationSignatures = Annotations[AnnotationSignature]
  type AnnotationData = Annotations[String]
}
