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
          s"""${spaces}class __$key(a${depth+1}: org.broadinstitute.hail.annotations.Annotations) {
             |${makeDeclarationsRecursive(a2, depth = depth + 1, nSpace = nSpace)}${spaces}}
             |${spaces}val $key: __$key = new __$key(${s"""$param.attrs("$key").asInstanceOf[org.broadinstitute.hail.annotations.Annotations]"""})
             |""".stripMargin
        case sig: AnnotationSignature =>
            s"${spaces}val $key: FilterOption[${sig.typeOf}] = new FilterOption(${
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
