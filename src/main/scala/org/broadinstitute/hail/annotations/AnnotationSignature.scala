package org.broadinstitute.hail.annotations

abstract class AnnotationSignature {
  def emitUtilities: String
  def emitConversionIdentifier: String
  def emitType: String

}

case class SimpleSignature(emitType: String, emitConversionIdentifier: String, desc: String) extends AnnotationSignature {

  def this(emitType: String, emitConversionIdentifier: String) = this(emitType, emitConversionIdentifier, "")

  def emitUtilities = ""

}