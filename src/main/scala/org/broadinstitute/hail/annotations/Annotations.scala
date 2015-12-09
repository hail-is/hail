package org.broadinstitute.hail.annotations

case class Annotations[T](maps: Map[String, Map[String, T]], vals: Map[String, T]) extends Serializable {

  def nAttrs: Int = {
    var i = 0
    maps.foreach {
      case (id, m) =>
        i += m.size
    }
    i += vals.size
    i
  }

  def hasMap(str: String): Boolean = maps.contains(str)

  def contains(str: String): Boolean = vals.contains(str.toLowerCase)

  def contains(parent: String, str: String): Boolean = hasMap(parent) && maps(parent).contains(str)

  def get(str: String): Option[T] = vals.get(str)

  def get(parent: String, str: String): Option[T] = {
    if (!hasMap(parent))
      None
    else
      maps(parent).get(str)
  }

  def getOrElse(parent: String, str: String, default: T): T = {
    if (!hasMap(parent) || !contains(parent, str))
      default
    else
      maps(parent)(str)
  }

  def getOrElse(str: String, default: T): T = {
    if (!contains(str))
      default
    else
      vals(str)
  }

  def addMap(name: String, m: Map[String, T]): Annotations[T] = {
    Annotations(maps
      .-(name)
      .+((name, m)), vals)
  }

  def addMaps(newMaps: Map[String, Map[String, T]]): Annotations[T] = {
    Annotations(maps
      .--(newMaps.keys)
      .++(newMaps), vals)
  }

  def addVal(name: String, mapping: T): Annotations[T] = {
    Annotations(maps, vals
      .-(name)
      .+((name, mapping)))
  }

  def addVals(newVals: Map[String, T]): Annotations[T] = {
    Annotations(maps, vals
      .--(newVals.keys)
      .++(newVals))
  }
}

object EmptyAnnotationSignatures {
  def apply(): AnnotationSignatures = {
    Annotations(Map.empty[String, Map[String, AnnotationSignature]], Map.empty[String, AnnotationSignature])
  }
}

object EmptyAnnotations {
  def apply(): AnnotationData = {
    Annotations(Map.empty[String, Map[String, String]], Map.empty[String, String])
  }
}

object EmptySampleAnnotations {
  def apply(nSamples: Int): IndexedSeq[AnnotationData] = {
    (0 until nSamples)
      .map(i => Annotations(Map.empty[String, Map[String, String]], Map.empty[String, String]))
  }
}

object AnnotationUtils {

  def annotationToString(ar: AnyRef): String = {
    ar match {
      case iter: Iterable[_] => iter.map(_.toString).reduceRight(_ + ", " + _)
      case _ => ar.toString
    }
  }

  def parseAnnotationType(str: String): String = {
    str match {
      case "Flag" => "Boolean"
      case "Integer" => "Int"
      case "Float" => "Double"
      case "String" => "String"
      case _ => throw new UnsupportedOperationException("unexpected annotation type")
    }
  }
}

object AnnotationClassBuilder {

  def signatures(sigs: AnnotationSignatures, hiddenClassName: String): String = {
    val internalClasses = sigs.maps.map {
      case (subclass, subMap) =>
        s"class __${subclass}Annotations(subMap: Map[String, String]) extends Serializable {\n" +
          subMap.map { case (k, sig) =>
//            s"""  val $k: $kType = subMap.getOrElse("$k", \"false\").$kMethod\n"""
            val default = getDefault(sig.getType)
            s"""  val $k: ${sig.getType} = subMap.getOrElse("$k", "$default").${sig.conversion}\n"""
          }
            .foldRight[String]("")(_ + _) + "}\n"
    }
      .foldRight[String]("")(_ + _)

    val hiddenClass = s"class ${hiddenClassName}Annotations" +
      s"(annot: org.broadinstitute.hail.annotations.AnnotationData) extends Serializable {\n" +
      sigs.maps.map { case (subclass, subMap) =>
        s"""  val $subclass =  new __${subclass}Annotations(annot.maps(\"$subclass\"))\n""" }
        .foldRight[String]("")(_ + _) +
      sigs.vals.map { case (k, sig) =>
        val default = getDefault(sig.getType)
        s"""  val $k: ${sig.getType} = annot.vals.getOrElse("$k", "$default").${sig.conversion} \n"""
      }
        .foldRight[String]("")(_ + _) + "}\n"

    "\n" + internalClasses + hiddenClass
  }

  def instantiate(exposedName: String, hiddenClassName: String): String = {
    s"val $exposedName = new ${hiddenClassName}Annotations($hiddenClassName)\n"
  }

  def makeIndexedSeq(hiddenOutputName: String, hiddenClassName: String, hiddenAnnotationArrayName: String): String = {
    s"val $hiddenOutputName: IndexedSeq[${hiddenClassName}Annotations] = " +
      s"$hiddenAnnotationArrayName.map(new ${hiddenClassName}Annotations(_))\n"
  }

  val arrayRegex = """Array\[(\w+)\]""".r
  val optionRegex = """Option\[(\w+)\]""".r
  private def getDefault(typeStr: String): String = {
    if (typeStr == "Int" || typeStr == "Double")
      "0"
    else if (typeStr == "Boolean")
      "false"
    else
      typeStr match {
        case optionRegex(subType) => "None"
        case arrayRegex(subType) => getDefault(subType)
        case _ => ""
      }
  }
}