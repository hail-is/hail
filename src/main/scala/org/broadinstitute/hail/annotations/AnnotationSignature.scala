package org.broadinstitute.hail.annotations

abstract class AnnotationSignature {
  def emitUtilities: String
  def emitConversionIdentifier: String
  def emitType: String

}
