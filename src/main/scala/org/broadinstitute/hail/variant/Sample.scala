package org.broadinstitute.hail.variant

case class Sample(id: String) {
  override def toString: String = id
}
