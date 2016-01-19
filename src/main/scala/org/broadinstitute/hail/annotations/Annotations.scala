package org.broadinstitute.hail.annotations

case class Annotations(attrs: Map[String, Any]) extends Serializable {

  def contains(elem: String): Boolean = attrs.contains(elem)

  def isAnnotations(elem: String): Boolean = attrs(elem).isInstanceOf[Annotations]

  def isValue(elem: String): Boolean = !attrs(elem).isInstanceOf[Annotations]

  def getAnnotations(key: String): Annotations = attrs(key).asInstanceOf[Annotations]

  def +(key: String, value: Any): Annotations = Annotations(attrs + (key -> value))

  def +(map: Map[String, Any]): Annotations = Annotations(attrs ++ map)

  def ++(other: Annotations): Annotations = this + other.attrs
}

object Annotations {
  def empty(): Annotations = Annotations(Map.empty[String, Any])

  def emptyIndexedSeq(n: Int): IndexedSeq[Annotations] = IndexedSeq.fill(n)(Annotations.empty())
}

object AnnotationClassBuilder {

  def makeDeclarationsRecursive(sigs: Annotations, depth: Int = 1, nSpace: Int = 0): String = {
    val spaces = (0 until (depth * 2 + nSpace)).map(i => " ").foldLeft("")(_ + _)
    val param = s"a$depth"
    sigs.attrs.map { case (key, attr) =>
      attr match {
        case a2: Annotations =>
          s"""${spaces}class __$key(a${depth + 1}: org.broadinstitute.hail.annotations.Annotations) {
             |${makeDeclarationsRecursive(a2, depth = depth + 1, nSpace = nSpace)}${spaces}}
             |${spaces}def $key: __$key = new __$key(${s"""$param.attrs("$key").asInstanceOf[org.broadinstitute.hail.annotations.Annotations]"""})
             |""".stripMargin
        case sig: AnnotationSignature =>
          s"${spaces}def $key: FilterOption[${sig.typeOf}] = new FilterOption(${
            s"""$param.attrs.get("$key").asInstanceOf[Option[${sig.typeOf}]])
               |""".stripMargin
          }"
        case _ => s"$key -> $attr \n"
      }
    }
      .foldLeft("")(_ + _)
  }

  def makeDeclarations(sigs: Annotations, className: String, nSpace: Int = 0): String = {
    val spaces = (0 until nSpace).map(i => " ").foldRight("")(_ + _)
    s"""class $className(a1: org.broadinstitute.hail.annotations.Annotations) {
        |${makeDeclarationsRecursive(sigs, nSpace = nSpace)}}
        |""".stripMargin
  }

  def instantiate(exposedName: String, className: String, rawName: String): String = {
    s"val $exposedName = new $className($rawName)"
  }

  def instantiateIndexedSeq(exposedName: String, className: String, rawArrayName: String): String =
    s"""val $exposedName: IndexedSeq[$className] = $rawArrayName.map(new $className(_))""".stripMargin
}
