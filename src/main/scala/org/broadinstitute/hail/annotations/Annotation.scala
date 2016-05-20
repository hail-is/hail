package org.broadinstitute.hail.annotations

import org.apache.spark.sql.Row
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr.{EvalContext, _}
import org.broadinstitute.hail.variant.{AltAllele, Genotype, Variant}
import org.json4s._
import org.json4s.jackson.Serialization

object Annotation {

  final val SAMPLE_HEAD = "sa"

  final val VARIANT_HEAD = "va"

  final val GLOBAL_HEAD = "global"

  def empty: Annotation = null

  def emptyIndexedSeq(n: Int): IndexedSeq[Annotation] = IndexedSeq.fill[Annotation](n)(Annotation.empty)

  def printAnnotation(a: Any, nSpace: Int = 0): String = {
    val spaces = " " * nSpace
    a match {
      case null => "Null"
      case r: Row =>
        "Row:\n" +
          r.toSeq.zipWithIndex.map { case (elem, index) =>
            s"""$spaces[$index] ${printAnnotation(elem, nSpace + 4)}"""
          }
            .mkString("\n")
      case a => a.toString + ": " + a.getClass.getSimpleName
    }
  }

  def apply(args: Any*): Annotation = Row.fromSeq(args)

  def fromSeq(values: Seq[Any]): Annotation = Row.fromSeq(values)

  def fromJson(jv: JValue, t: BaseType, parent: String): Annotation = {
    implicit val formats = Serialization.formats(NoTypeHints)

    (jv, t) match {
      case (JNull | JNothing, _) => null
      case (JInt(x), TInt) => x.toInt
      case (JInt(x), TLong) => x.toLong
      case (JInt(x), TDouble) => x.toDouble
      case (JInt(x), TString) => x.toString
      case (JDouble(x), TDouble) => x
      case (JDouble(x), TFloat) => x.toFloat
      case (JString(x), TString) => x
      case (JString(x), TChar) => x
      case (JString(x), TInt) =>
        x.toInt
      case (JString(x), TDouble) =>
        if (x.startsWith("-:"))
          x.drop(2).toDouble
        else
          x.toDouble
      case (JBool(x), TBoolean) => x

      case (JObject(jfields), t: TStruct) =>
        if (t.size == 0)
          Annotation.empty
        else {
          val a = Array.fill[Any](t.size)(null)

          for ((name, jv2) <- jfields) {
            t.selfField(name) match {
              case Some(f) =>
                a(f.index) = fromJson(jv2, f.`type`, parent + "." + name)

              case None =>
                warn(s"Signature for $parent has no field $name")
            }
          }

          Annotation(a: _*)
        }
      case (_, TAltAllele) =>
        jv.extract[AltAllele]
      case (_, TVariant) =>
        jv.extract[JSONExtractVariant].toVariant
      case (_, TGenotype) =>
        jv.extract[JSONExtractGenotype].toGenotype

      case (JArray(a), TArray(elementType)) =>
        a.iterator.map(jv2 => fromJson(jv2, elementType, parent + ".<array>")).toArray[Any]: IndexedSeq[Any]

      case (JArray(a), TSet(elementType)) =>
        a.iterator.map(jv2 => fromJson(jv2, elementType, parent + ".<array>")).toSet[Any]

      case _ =>
        warn(s"Can't convert json value $jv to signature $t for $parent.")
        null
    }
  }

  case class JSONExtractGenotype(
    gt: Option[Int],
    ad: Option[Array[Int]],
    dp: Option[Int],
    gq: Option[Int],
    pl: Option[Array[Int]],
    fakeRef: Boolean) {
    def toGenotype =
      Genotype(gt, ad, dp, gq, pl, fakeRef)
  }

  case class JSONExtractVariant(contig: String,
    start: Int,
    ref: String,
    altAlleles: List[AltAllele]) {
    def toVariant =
      Variant(contig, start, ref, altAlleles.toArray)
  }

  def jsonExtractVariant(t: Type, variantFields: String): Any => Option[Variant] = {
    val ec = EvalContext(Map(
      "root" ->(0, t)))

    val fs: Array[(BaseType, () => Option[Any])] = Parser.parseExprs(variantFields, ec)

    if (fs.length != 4)
      fatal(s"wrong number of variant field expressions: expected 4, got ${fs.length}")

    if (fs(0)._1 != TString)
      fatal(s"wrong type for chromosome field: expected String, got ${fs(0)._1}")
    if (fs(1)._1 != TInt)
      fatal(s"wrong type for pos field: expected Int, got ${fs(1)._1}")
    if (fs(2)._1 != TString)
      fatal(s"wrong type for ref field: expected String, got ${fs(2)._1}")
    if (fs(3)._1 != TArray(TString))
      fatal(s"wrong type for alt field: expected Array[String], got ${fs(3)._1}")

    (root: Annotation) => {
      ec.setAll(root)

      val vfs = fs.map(_._2())

      vfs(0).flatMap { chr =>
        vfs(1).flatMap { pos =>
          vfs(2).flatMap { ref =>
            vfs(3).map { alt =>
              Variant(chr.asInstanceOf[String],
                pos.asInstanceOf[Int],
                ref.asInstanceOf[String],
                alt.asInstanceOf[IndexedSeq[String]].toArray)
            }
          }
        }
      }
    }
  }

  def jsonExtractSample(t: Type, sampleExpr: String): Any => Option[String] = {
    val ec = EvalContext(Map(
      "root" ->(0, t)))

    val f: () => Option[Any] = Parser.parse(sampleExpr, ec, TString)

    (root: Annotation) => {
      ec.setAll(root)
      f().map(_.asInstanceOf[String])
    }
  }
}
