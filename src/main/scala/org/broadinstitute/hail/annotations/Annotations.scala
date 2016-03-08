package org.broadinstitute.hail.annotations

case class Annotations(attrs: Map[String, Any]) extends Serializable {

  def contains(elem: String): Boolean = attrs.contains(elem)

  def get[T](key: String): T = attrs(key).asInstanceOf[T]

  def getOption[T](key: String): Option[T] = attrs.get(key).map(_.asInstanceOf[T])

  def +(key: String, value: Any): Annotations = Annotations(attrs + (key -> value))

  def ++(other: Annotations): Annotations = {
    Annotations(attrs ++ other.attrs.map {
      case (key, value) =>
        attrs.get(key) match {
          case Some(a1: Annotations) =>
            value match {
              case a2: Annotations => (key, a1 ++ a2)
              case _ => (key, value)
            }
          case _ => (key, value)
        }
    })
  }

  def -(key: String): Annotations = Annotations(attrs - key)
}

object Annotations {
  def empty(): Annotations = Annotations(Map.empty[String, Any])

  def emptyIndexedSeq(n: Int): IndexedSeq[Annotations] = Array.fill(n)(Annotations.empty())

  def validSignatures(a: Any): Boolean = {
    a match {
      case anno1: Annotations => anno1.attrs.forall { case (k, v) => validSignatures(v) }
      case sig: AnnotationSignature => true
      case _ => false
    }
  }
}

object AnnotationClassBuilder {

  def makeDeclarationsRecursive(sigs: Annotations, depth: Int = 1, nSpace: Int = 0): String = {
    val spaces = (0 until (depth * 2 + nSpace)).map(i => " ").foldLeft("")(_ + _)
    val param = s"__a$depth"
    sigs.attrs.map { case (key, attr) =>
      attr match {
        case a2: Annotations =>
          s"""${spaces}case class `__$key`(__a${depth + 1}: org.broadinstitute.hail.annotations.Annotations) {
             |${makeDeclarationsRecursive(a2, depth = depth + 1, nSpace = nSpace)}$spaces}
             |${spaces}lazy val `$key`: `__$key` = new `__$key`(${
            s"""$param.get[org.broadinstitute.hail.annotations.Annotations]("$key")"""
          })
             |""".stripMargin
        case sig: AnnotationSignature =>
          s"${spaces}lazy val `$key`: FilterOption[${sig.typeOf}] = new FilterOption(${
            s"""$param.getOption[${sig.typeOf}]("$key"))
               |""".stripMargin
          }"
        case _ => s"$key -> $attr \n"
      }
    }
      .foldLeft("")(_ + _)
  }

  def makeDeclarations(sigs: Annotations, className: String, nSpace: Int = 0): String = {
    val spaces = (0 until nSpace).map(i => " ").foldRight("")(_ + _)
    s"""case class $className(__a1: org.broadinstitute.hail.annotations.Annotations) {
        |${makeDeclarationsRecursive(sigs, nSpace = nSpace)}$spaces}
        |""".stripMargin
  }

  def instantiate(exposedName: String, className: String, rawName: String): String = {
    s"lazy val $exposedName = new $className($rawName)"
  }

  def instantiateIndexedSeq(exposedName: String, className: String, rawArrayName: String): String =
    s"""lazy val $exposedName: IndexedSeq[$className] = $rawArrayName.map(new $className(_))""".stripMargin
}
