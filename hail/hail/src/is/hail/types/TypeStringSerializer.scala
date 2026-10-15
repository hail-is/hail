package is.hail.types

import is.hail.variant.ReferenceGenomeEnv

import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString

/** A json4s serializer that writes a type as a string and reads it back by parsing that string.
  *
  * Reading a type needs the reference genomes in scope. Without them, the serializer only writes.
  */
abstract class TypeStringSerializer[T: Manifest](
  rgs: Option[ReferenceGenomeEnv],
  parse: (String, ReferenceGenomeEnv) => T,
  render: T => String,
) extends CustomSerializer[T](_ =>
      (
        { case JString(s) => parse(s, rgs.getOrElse(TypeStringSerializer.cannotRead[T])) },
        { case t: T => JString(render(t)) },
      )
    )

object TypeStringSerializer {
  def cannotRead[T: Manifest]: Nothing =
    throw new IllegalStateException(
      s"Cannot read a ${manifest[T].runtimeClass.getSimpleName}: these json4s Formats only " +
        "write types. Reading a type needs the reference genomes in scope."
    )
}
