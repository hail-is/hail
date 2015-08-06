package org.broadinstitute.k3.variant

case class Variant(contig: String,
  // FIXME 0- or 1-based?
  start: Int,
  ref: String,
  alt: String)
