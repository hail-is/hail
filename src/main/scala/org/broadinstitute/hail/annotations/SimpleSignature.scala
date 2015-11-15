package org.broadinstitute.hail.annotations

class SimpleSignature(scalaType: String, conversionMethod: String, description: String)
  extends AnnotationSignature {

  def buildCaseClasses: String = ""

  def conversion: String = conversionMethod

  def getType: String = scalaType
}
