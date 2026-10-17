package is.hail.variant

import is.hail.utils.fatal

/** The reference genomes in scope when parsing a type, by name.
  *
  * Builtin reference genomes are always in scope: a name not given here resolves to the cached
  * builtin of that name, if there is one.
  */
final class ReferenceGenomeEnv(byName: Map[String, ReferenceGenome]) {

  def resolve(name: String): ReferenceGenome =
    byName.get(name)
      .orElse(ReferenceGenome.builtins.get(name))
      .getOrElse(fatal(s"Reference genome '$name' is not in scope."))

  override def toString: String = byName.keys.mkString("ReferenceGenomeEnv(", ", ", ")")
}

object ReferenceGenomeEnv {

  /** Only the builtin reference genomes. */
  val empty: ReferenceGenomeEnv = new ReferenceGenomeEnv(Map.empty)

  def apply(byName: Map[String, ReferenceGenome]): ReferenceGenomeEnv =
    new ReferenceGenomeEnv(byName)
}
