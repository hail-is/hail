package org.broadinstitute.hail.annotations

import java.io.{DataInputStream, DataOutputStream}
import org.broadinstitute.hail.Utils._
import com.esotericsoftware.kryo

case class Annotations(attrs: Map[String, Any]) extends Serializable {

  def contains(elem: String): Boolean = attrs.contains(elem)

  def isAnnotations(elem: String): Boolean = attrs(elem).isInstanceOf[Annotations]

  def isValue(elem: String): Boolean = !attrs(elem).isInstanceOf[Annotations]

  def getAnnotations(key: String): Annotations = attrs(key).asInstanceOf[Annotations]

  def + (key: String, value: Any): Annotations = Annotations(attrs + (key -> value))

  def + (map: Map[String, Any]): Annotations = Annotations(attrs ++ map)

  def ++ (other: Annotations): Annotations = this + other.attrs


}

object Annotations {
  def empty(): Annotations = Annotations(Map.empty[String, Any])
  def emptyIndexedSeq(n: Int): IndexedSeq[Annotations] = IndexedSeq.fill(n)(Annotations.empty())
//  def toArrays(): (Array[(String, String, T)], Array[(String, T)]) = {
//    (maps.toArray.flatMap {
//      case (mapName, map) => map
//        .toArray
//        .map { case (s, t) => (mapName, s, t) }
//    },
//      vals.toArray)
//  }

//  implicit def writableAnnotations[T](implicit writableT: DataWritable[T]): DataWritable[Annotations[T]] =
//    new DataWritable[Annotations[T]] {
//      def write(dos: DataOutputStream, t: Annotations[T]) {
//        writeData(dos, t.maps)
//        writeData(dos, t.vals)
//      }
//    }
//
//
//  implicit def readableAnnotations[T](implicit readableT: DataReadable[T]): DataReadable[Annotations[T]] =
//    new DataReadable[Annotations[T]] {
//      def read(dis: DataInputStream): Annotations[T] = {
//        Annotations(readData[Map[String, Map[String, T]]](dis),
//          readData[Map[String, T]](dis))
//      }
//    }

//  def fromIndexedSeqs[T](arr1: IndexedSeq[(String, String, T)], arr2: IndexedSeq[(String, T)]): Annotations[T] = {
//    val maps = arr1
//      .groupBy(_._1)
//      .mapValues(l => l.map {
//        case (name, fieldName, field) => (fieldName, field)
//      }.toMap).force
//    val vals = arr2.toMap.force
//    Annotations(maps, vals)
//  }

//  def empty[T](): Annotations[T] =
//    Annotations(Map.empty[String, Map[String, T]], Map.empty[String, T])
//
//  def emptyOfSignature(): AnnotationSignatures = empty[AnnotationSignature]()
//
//  def emptyOfData(): AnnotationData = empty[String]()
//
//  def emptyOfArrayString(nSamples: Int): IndexedSeq[AnnotationData] =
//    IndexedSeq.fill[Annotations[String]](nSamples)(empty[String]())
}

object AnnotationClassBuilder {

  def makeDeclarationsRecursive(sigs: Annotations, depth: Int = 1, nSpace: Int = 0): String = {
    val spaces = (0 until (depth*2 + nSpace)).map(i => " ").foldLeft("")(_ + _)
    val param = s"a$depth"
    sigs.attrs.map { case (key, attr) =>
      attr match {
        case a2: Annotations =>
          s"""${spaces}class __$key(a${depth+1}: Annotations) {
             |${makeDeclarationsRecursive(a2, depth = depth + 1)}${spaces}}
             |${spaces}val $key: __$key = new __$key(${s"""$param.attrs("$key").asInstanceOf[org.broadinstitute.hail.annotations.Annotations]"""})
             |""".stripMargin
        case sig: AnnotationSignature =>
            s"${spaces}val $key: FilterOption[${sig.typeOf}] = new FilterOption(${
                s"""$param.attrs.get("$key").asInstanceOf[Option[${sig.typeOf}]])
                    |""".stripMargin
            }"
        case _ => "somebody goofed\n"
      }
    }
    .foldLeft("")(_ + _)
  }

  def makeDeclarations(sigs: Annotations, exposedName: String, annotationsPath: String, nSpace: Int = 0): String = {
    val spaces = (0 until nSpace).map(" ").foldRight("")(_ + _)
    s"""${spaces}class ${annotationsPath}Class(a1: Annotations) {
       |${makeDeclarationsRecursive(sigs, nSpace = nSpace)}}
       |${spaces}val $exposedName = new ${annotationsPath}Class($annotationsPath)
     """.stripMargin
  }

  def signatures(sigs: Annotations, className: String,
    makeToString: Boolean = false): String = {
    throw new UnsupportedOperationException
  }
//    val internalClasses = sigs.attrs.map { attr =>
//      attr match {
//        case
//      }
//      case (subclass, subMap) =>
//        val attrs = subMap
//          .map { case (k, sig) =>
//            s"""  val $k: FilterOption[${sig.emitType}] = new FilterOption(subMap.get("$k").map(_.${realConversion(sig.emitConversionIdentifier)}))"""
//          }
//          .mkString("\n")
//        val methods: String = {
//          if (makeToString) {
//            s"""  def __fields: Array[String] = Array(
//                |    ${subMap.keys.toArray.sorted.map(s => s"""toTSVString($s)""").mkString(",")}
//                |  )
//                |  override def toString: String = __fields.mkRealString(";")
//                |  def all: String = __fields.mkRealString("\t")""".stripMargin
//          } else ""
//        }
//        s"""class __$subclass(subMap: Map[String, String]) extends Serializable {
//            |$attrs
//            |$methods
//            |}""".stripMargin
//    }
//      .mkString("\n")
//
//    val hiddenClass = {
//      val classes =
//        sigs.attrs.map { case (subclass, subMap) =>
//          s"""  val $subclass = new __$subclass(annot.attrs("$subclass"))"""
//        }
//          .mkString("\n")
//      val vals = sigs.vals.map { case (k, sig) =>
//        s"""  val $k: FilterOption[${sig.emitType}] = new FilterOption[${sig.emitType}](annot.getVal("$k").map(_.${realConversion(sig.emitConversionIdentifier)}))"""
//      }
//        .mkString("\n")
//      s"""class $className(annot: org.broadinstitute.hail.annotations.AnnotationData)
//          |  extends Serializable {
//          |  ${if (internalClasses.nonEmpty) internalClasses else "// no internal class declarations"}
//          |  ${if (classes.nonEmpty) classes else "// no class instantiations"}
//          |  ${if (vals.nonEmpty) vals else "// no vals"}
//          |}
//          |""".stripMargin
//    }
//
//    s"""
//       |$hiddenClass
//    """.stripMargin
//  }

  def instantiate(exposedName: String, className: String, rawName: String): String = {
    s"val $exposedName = new $className($rawName)\n"
  }

  def instantiateIndexedSeq(exposedName: String, classIdentifier: String, rawArrayName: String): String =
    s"""val $exposedName: IndexedSeq[$classIdentifier] =
        |  $rawArrayName.map(new $classIdentifier(_))
     """.stripMargin
}
