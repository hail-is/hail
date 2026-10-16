package is.hail.variant

import is.hail.utils.fatal

import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString

/** The reference genomes in scope, by name.
  *
  * Builtin reference genomes are always in scope: a name not given here resolves to the cached
  * builtin of that name, if there is one.
  */
final class ReferenceGenomeEnv(byName: Map[String, ReferenceGenome]) {

  def resolve(name: String): ReferenceGenome =
    byName.get(name)
      .orElse(ReferenceGenome.builtins.get(name))
      .getOrElse(fatal(s"Reference genome '$name' is not in scope."))

  def apply(name: String): ReferenceGenome = resolve(name)

  /** These reference genomes, plus `rgs`. Fails if any of `rgs` differs from a reference genome of
    * the same name already here.
    */
  def withReferences(rgs: Iterable[ReferenceGenome]): ReferenceGenomeEnv =
    new ReferenceGenomeEnv(ReferenceGenome.addFatalOnCollision(byName, rgs))

  override def toString: String = byName.keys.mkString("ReferenceGenomeEnv(", ", ", ")")
}

object ReferenceGenomeEnv {

  /** Only the builtin reference genomes. */
  val empty: ReferenceGenomeEnv = new ReferenceGenomeEnv(Map.empty)

  def apply(byName: Map[String, ReferenceGenome]): ReferenceGenomeEnv =
    new ReferenceGenomeEnv(byName)
}

/** A json4s serializer that writes a reference genome as its name, and reads it back by resolving
  * that name against the reference genomes in scope. Without them, it only writes.
  */
class ReferenceGenomeNameSerializer(rgs: Option[ReferenceGenomeEnv])
    extends CustomSerializer[ReferenceGenome](_ =>
      (
        {
          case JString(name) =>
            rgs.getOrElse(throw new IllegalStateException(
              s"Cannot read reference genome '$name': these json4s Formats have no reference " +
                "genomes in scope."
            )).resolve(name)
        },
        { case rg: ReferenceGenome => JString(rg.name) },
      )
    )
