package is.hail.variant

import is.hail.types.physical.{PCanonicalLocus, PPackedLocus}
import is.hail.types.virtual.TLocus

/** Locus types on the builtin reference genomes, by name.
  *
  * Only tests build locus types from a name. Production code passes the reference genome itself.
  */
object BuiltinLoci {
  def builtin(name: String): ReferenceGenome = ReferenceGenome.builtinReferences()(name)

  def tlocus(name: String): TLocus = TLocus(builtin(name))

  def pcanonicalLocus(name: String, required: Boolean = false): PCanonicalLocus =
    PCanonicalLocus(builtin(name), required)

  def ppackedLocus(name: String, required: Boolean = false): PPackedLocus =
    PPackedLocus(builtin(name), required)
}
