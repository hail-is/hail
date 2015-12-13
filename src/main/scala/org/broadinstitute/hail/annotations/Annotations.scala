package org.broadinstitute.hail.annotations

case class Annotations[T](maps: Map[String, Map[String, T]], vals: Map[String, T]) extends Serializable {

  def nAttrs: Int = maps.map(_._2.size).sum + vals.size

  def hasMap(str: String): Boolean = maps.contains(str)

  def contains(str: String): Boolean = vals.contains(str)

  def contains(parent: String, str: String): Boolean = hasMap(parent) && maps(parent).contains(str)

  def get(str: String): Option[T] = vals.get(str)

  def get(parent: String, str: String): Option[T] =
    maps.get(parent).flatMap(_.get(str))

  def getMap(parent: String): Option[Map[String, T]] = maps.get(parent)

  def addMap(name: String, m: Map[String, T]): Annotations[T] =
    Annotations(maps + ((name, m)), vals)

  def addMaps(newMaps: Map[String, Map[String, T]]): Annotations[T] =
    Annotations(maps ++ newMaps, vals)

  def addVal(name: String, mapping: T): Annotations[T] = Annotations(maps, vals + ((name, mapping)))

  def addVals(newVals: Map[String, T]): Annotations[T] = Annotations(maps, vals ++ newVals)

  def ++ (other: Annotations[T]): Annotations[T] = {
    new Annotations(maps ++ other.maps, vals ++ other.vals)
  }
}

object Annotations {
  def emptyOfSignature(): AnnotationSignatures =
    Annotations(Map.empty[String, Map[String, AnnotationSignature]], Map.empty[String, AnnotationSignature])

  def emptyOfString(): AnnotationData =
    Annotations(Map.empty[String, Map[String, String]], Map.empty[String, String])

  def emptyOfArrayString(nSamples: Int): IndexedSeq[AnnotationData] =
    (0 until nSamples)
      .map(i => Annotations(Map.empty[String, Map[String, String]], Map.empty[String, String]))
}

object AnnotationUtils {

  def annotationToString(ar: AnyRef): String = {
    ar match {
      case iter: Iterable[_] => if (iter.isEmpty) "" else iter.map(_.toString).mkString(", ")
      case _ => ar.toString
    }
  }
}

object AnnotationClassBuilder {

  def signatures(sigs: AnnotationSignatures, hiddenClassName: String,
    makeToString: Boolean = false, missing: String = ""): String = {
    val internalClasses = sigs.maps.map {
      case (subclass, subMap) =>
        s"class __${subclass}Annotations(subMap: Map[String, String]) extends Serializable {\n" +
          subMap.map { case (k, sig) =>
            //            s"""  val $k: $kType = subMap.getFromMapOrElse("$k", \"false\").$kMethod\n"""
            val default = getDefault(sig.getType)
            s"""  val $k: ${sig.getType} = subMap.getOrElse("$k", "$default").${sig.conversion}\n"""
          }
            .foldRight[String]("")(_ + _) + {
          if (makeToString) {
            val keys = subMap.keys.toArray.sorted
            "  def __fields: Array[String] = Array(" + {
              if (keys.isEmpty) ""
              else keys.map(s => s"""formatString($s, "$missing")""")
                .mkString(",")
            } + ")\n" +
            """  override def toString: String =
              |    if (__fields.length == 0) "" else __fields.mkString(";")
              |  def all: String = if (__fields.length == 0) "" else __fields.mkString("\t")
              |
            """.stripMargin
          }
          else ""
        } +
          "}\n"
    }
      .foldRight[String]("")(_ + _)

    val hiddenClass = s"class ${hiddenClassName}Annotations" +
      s"(annot: org.broadinstitute.hail.annotations.AnnotationData) extends Serializable {\n" +
      sigs.maps.map { case (subclass, subMap) =>
        s"""  val $subclass =  new __${subclass}Annotations(annot.maps(\"$subclass\"))\n"""
      }
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

  def makeIndexedSeq(hiddenOutputName: String, hiddenClassName: String, hiddenAnnotationArrayName: String): String =
    s"""val $hiddenOutputName: IndexedSeq[${hiddenClassName}Annotations] =
       |$hiddenAnnotationArrayName.map(new ${hiddenClassName}Annotations(_))
       |
     """.stripMargin

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
