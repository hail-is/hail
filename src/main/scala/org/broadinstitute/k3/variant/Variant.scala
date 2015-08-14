package org.broadinstitute.k3.variant

case class Variant(contig: String,
                   // FIXME: 0- or 1-based?
                   start: Int,
                   ref: String,
                   alt: String) {
  require(ref.length == 1 || alt.length == 1)
  // FIXME do we need ref(0) == alt(0) for insertions and deletions?a

  def isSNP: Boolean = ref.length == alt.length

  def isIndel: Boolean = ref.length != alt.length

  def isInsertion: Boolean = ref.length < alt.length

  def isDeletion: Boolean = ref.length > alt.length

  def isComplex: Boolean = false

  def isTransition: Boolean = isSNP &&
    ((ref == "A" && alt == "G") || (ref == "G" && alt == "A") ||
      (ref == "C" && alt == "T") || (ref == "T" && alt == "C"))

  def isTransversion: Boolean = isSNP && !isTransition
}
