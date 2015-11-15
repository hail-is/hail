package org.broadinstitute.hail.annotations

abstract class AnnotationSignature {
  def buildCaseClasses: String
  def conversion: String
  def getType: String
}
