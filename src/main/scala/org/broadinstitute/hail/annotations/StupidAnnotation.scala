package org.broadinstitute.hail.annotations

class StupidAnnotation() extends AnnotationSignature {
  def buildCaseClasses: String = throw new UnsupportedOperationException
  def conversion: String = throw new UnsupportedOperationException
  def getType: String = throw new UnsupportedOperationException
}
