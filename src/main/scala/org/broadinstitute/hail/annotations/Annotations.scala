package org.broadinstitute.hail.annotations

import org.broadinstitute.hail.Utils._

case class Annotations[T](maps: Map[String, Map[String, T]], vals: Map[String, T]) extends Serializable {

  def hasMap(str: String): Boolean = maps.contains(str)

  def containsVal(str: String): Boolean = vals.contains(str)

  def containsInMap(parent: String, str: String): Boolean = hasMap(parent) && maps(parent).contains(str)

  def getVal(str: String): Option[T] = vals.get(str)

  def getInMap(parent: String, str: String): Option[T] =
    maps.get(parent).flatMap(_.get(str))

  def getMap(parent: String): Option[Map[String, T]] = maps.get(parent)

  def addMap(name: String, m: Map[String, T]): Annotations[T] =
    Annotations(maps + ((name, m)), vals)

  def addMaps(newMaps: Map[String, Map[String, T]]): Annotations[T] =
    Annotations(maps ++ newMaps, vals)

  def addVal(name: String, mapping: T): Annotations[T] = Annotations(maps, vals + ((name, mapping)))

  def addVals(newVals: Map[String, T]): Annotations[T] = Annotations(maps, vals ++ newVals)

  def ++(other: Annotations[T]): Annotations[T] = {
    new Annotations(maps ++ other.maps, vals ++ other.vals)
  }

  def toArrays(): (Array[(String, String, T)], Array[(String, T)]) = {
    (maps.toArray.flatMap {
      case (mapName, map) => map
        .toArray
        .map{ case (s, t) => (mapName, s, t) }},
      vals.toArray)
  }
}

object Annotations {

  def fromIndexedSeqs[T](arr1: IndexedSeq[(String, String, T)], arr2: IndexedSeq[(String, T)]): Annotations[T] = {
    val maps = arr1
      .groupBy(_._1)
      .mapValues(l => l.map {
        case (name, fieldName, field) => (fieldName, field) }.toMap).force
    val vals = arr2.toMap.force
    Annotations(maps, vals)
  }

  def empty[T](): Annotations[T] =
    Annotations(Map.empty[String, Map[String, T]], Map.empty[String, T])

  def emptyOfSignature(): AnnotationSignatures = empty[AnnotationSignature]()

  def emptyOfData(): AnnotationData = empty[String]()

  def emptyOfArrayString(nSamples: Int): IndexedSeq[AnnotationData] =
    IndexedSeq.fill[Annotations[String]](nSamples)(empty[String]())
}

object AnnotationClassBuilder {

  def signatures(sigs: AnnotationSignatures, className: String,
    makeToString: Boolean = false): String = {
    def realConversion(s: String) = s match {
      case "toDouble" => "toRealDouble"
      case "toInt" => "toRealInt"
      case _ => s
    }
    val internalClasses = sigs.maps.map {
      case (subclass, subMap) =>
        val attrs = subMap
          .map { case (k, sig) =>
            s"""  val $k: FilterOption[${sig.emitType}] = new FilterOption(subMap.get("$k").map(_.${realConversion(sig.emitConversionIdentifier)}))"""
          }
          .mkString("\n")
        val methods: String = {
          if (makeToString) {
            s"""  def __fields: Array[String] = Array(
                |    ${subMap.keys.toArray.sorted.map(s => s"""toTSVString($s)""").mkString(",")}
                |  )
                |  override def toString: String = __fields.mkRealString(";")
                |  def all: String = __fields.mkRealString("\t")""".stripMargin
          } else ""
        }
        s"""class __$subclass(subMap: Map[String, String]) extends Serializable {
            |$attrs
            |$methods
            |}""".stripMargin
    }
      .mkString("\n")

    val hiddenClass = {
      val classes =
        sigs.maps.map { case (subclass, subMap) =>
          s"""  val $subclass = new __$subclass(annot.maps("$subclass"))"""
        }
          .mkString("\n")
      val vals = sigs.vals.map { case (k, sig) =>
        s"""  val $k: FilterOption[${sig.emitType}] = new FilterOption[${sig.emitType}](annot.getVal("$k").map(_.${realConversion(sig.emitConversionIdentifier)}))"""
      }
        .mkString("\n")
      s"""class $className(annot: org.broadinstitute.hail.annotations.AnnotationData)
          |  extends Serializable {
          |  ${if (internalClasses.nonEmpty) internalClasses else "// no internal class declarations"}
          |  ${if (classes.nonEmpty) classes else "// no class instantiations"}
          |  ${if (vals.nonEmpty) vals else "// no vals"}
          |}
          |""".stripMargin
    }

    s"""
       |$hiddenClass
    """.stripMargin
  }

  def instantiate(exposedName: String, className: String, rawName: String): String = {
    s"val $exposedName = new $className($rawName)\n"
  }

  def instantiateIndexedSeq(exposedName: String, classIdentifier: String, rawArrayName: String): String =
    s"""val $exposedName: IndexedSeq[$classIdentifier] =
        |  $rawArrayName.map(new $classIdentifier(_))
     """.stripMargin
}
