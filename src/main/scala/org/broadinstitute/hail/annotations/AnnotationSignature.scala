package org.broadinstitute.hail.annotations


abstract class AnnotationSignature {
  def typeOf: String
}

case class SimpleSignature(typeOf: String) extends AnnotationSignature