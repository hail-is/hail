package is.hail.variant

import is.hail.utils.fatal

import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString

/** The reference genomes in scope, by name.
  *
  * Only user-defined reference genomes are held here. Builtin reference genomes are always in
  * scope: a name not held here resolves to the cached builtin of that name, if there is one. A
  * builtin can be neither removed nor shadowed by a different reference genome of the same name.
  */
final class ReferenceGenomeEnv private (private val byName: Map[String, ReferenceGenome]) {

  def resolve(name: String): ReferenceGenome =
    byName.get(name)
      .orElse(ReferenceGenome.builtins.get(name))
      .getOrElse(fatal(s"Reference genome '$name' is not in scope."))

  def apply(name: String): ReferenceGenome = resolve(name)

  /** These reference genomes, plus `rgs`. Fails if any of `rgs` differs from a reference genome of
    * the same name already in scope, builtin or not. A reference genome equal to one already in
    * scope is not added, so the one already in scope is kept, and a builtin is never stored here.
    */
  def withReferences(rgs: Iterable[ReferenceGenome]): ReferenceGenomeEnv =
    new ReferenceGenomeEnv(rgs.foldLeft(byName) { (acc, rg) =>
      ReferenceGenome.builtins.get(rg.name).orElse(acc.get(rg.name)) match {
        case Some(existing) if existing == rg => acc
        case Some(_) =>
          fatal(
            s"Cannot add reference genome '${rg.name}', a different reference with that name already exists. " ++
              "Choose a reference name NOT in the following list:" ++
              (acc.keySet ++ ReferenceGenome.builtins.keySet).toSeq.sorted
                .mkString(start = "\n  ", sep = "\n  ", end = "")
          )
        case None => acc + (rg.name -> rg)
      }
    })

  /** These reference genomes, less the one named `name`, if there is one. Fails if `name` is a
    * builtin's: builtins are always in scope.
    */
  def without(name: String): ReferenceGenomeEnv = {
    if (ReferenceGenome.builtins.contains(name))
      fatal(s"Cannot remove builtin reference genome '$name'.")
    new ReferenceGenomeEnv(byName - name)
  }

  override def equals(that: Any): Boolean =
    that match {
      case that: ReferenceGenomeEnv => byName == that.byName
      case _ => false
    }

  override def hashCode: Int = byName.hashCode

  override def toString: String = byName.keys.mkString("ReferenceGenomeEnv(", ", ", ")")
}

object ReferenceGenomeEnv {

  /** Only the builtin reference genomes. */
  val empty: ReferenceGenomeEnv = new ReferenceGenomeEnv(Map.empty)

  /** The builtin reference genomes, plus `rgs`, by the rules of
    * [[ReferenceGenomeEnv.withReferences]].
    */
  def apply(rgs: Iterable[ReferenceGenome]): ReferenceGenomeEnv =
    empty.withReferences(rgs)
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
