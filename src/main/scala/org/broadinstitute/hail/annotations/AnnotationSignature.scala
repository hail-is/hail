package org.broadinstitute.hail.annotations

abstract class AnnotationSignature {
  def typeOf: Class
  def optional: Boolean
}

case class SimpleSignature(typeOf: Class, optional: Boolean = false) extends AnnotationSignature