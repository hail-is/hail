package org.broadinstitute.hail.annotations

import java.io.{DataInputStream, DataOutputStream}
import org.broadinstitute.hail.Utils._


abstract class AnnotationSignature {
  def typeOf: String
}

case class SimpleSignature(typeOf: String) extends AnnotationSignature