package is.hail.annotations

import org.apache.spark.sql.Row
import is.hail.expr._
import is.hail.utils.Interval
import is.hail.variant._

object Annotation {

  final val SAMPLE_HEAD = "sa"

  final val VARIANT_HEAD = "va"

  final val GLOBAL_HEAD = "global"

  def empty: Annotation = null

  def emptyIndexedSeq(n: Int): IndexedSeq[Annotation] = Array.fill[Annotation](n)(Annotation.empty)

  def printAnnotation(a: Any, nSpace: Int = 0): String = {
    val spaces = " " * nSpace
    a match {
      case null => "Null"
      case r: Row =>
        "Struct:\n" +
          r.toSeq.zipWithIndex.map { case (elem, index) =>
            s"""$spaces[$index] ${printAnnotation(elem, nSpace + 4)}"""
          }
            .mkString("\n")
      case a => a.toString + ": " + a.getClass.getSimpleName
    }
  }

  def expandType(t: Type): Type = t match {
    case TChar => TString
    case TVariant => Variant.t
    case TSample => TString
    case TGenotype => Genotype.t
    case TLocus => Locus.t
    case TArray(elementType) =>
      TArray(expandType(elementType))
    case TStruct(fields) =>
      TStruct(fields.map { f => f.copy(typ = expandType(f.typ)) })
    case TSet(elementType) =>
      TArray(expandType(elementType))
    case TDict(elementType) =>
      TArray(TStruct(
        "key" -> TString,
        "value" -> expandType(elementType)))
    case TAltAllele => AltAllele.t
    case TInterval =>
      TStruct(
        "start" -> Locus.t,
        "end" -> Locus.t)
    case _ => t
  }

  def expandAnnotation(a: Annotation, t: Type): Annotation =
    if (a == null)
      null
    else
      t match {
        case TVariant => a.asInstanceOf[Variant].toRow
        case TGenotype => a.asInstanceOf[Genotype].toRow
        case TLocus => a.asInstanceOf[Locus].toRow

        case TArray(elementType) =>
          a.asInstanceOf[IndexedSeq[_]].map(expandAnnotation(_, elementType))
        case TStruct(fields) =>
          Row.fromSeq((a.asInstanceOf[Row].toSeq, fields).zipped.map { case (ai, f) =>
            expandAnnotation(ai, f.typ)
          })

        case TSet(elementType) =>
          (a.asInstanceOf[Set[_]]
            .toArray[Any] : IndexedSeq[_])
            .map(expandAnnotation(_, elementType))

        case TDict(elementType) =>
          (a.asInstanceOf[Map[String, _]]
            .toArray[(String, Any)]: IndexedSeq[(String, Any)])
            .map { case (k, v) =>
              Annotation(k, expandAnnotation(v, elementType))
            }

        case TAltAllele => a.asInstanceOf[AltAllele].toRow

        case TInterval =>
          val i = a.asInstanceOf[Interval[Locus]]
          Annotation(i.start.toRow,
            i.end.toRow)

        // including TChar, TSample
        case _ => a
      }

  def flattenType(t: Type): Type = t match {
    case t: TStruct =>
      val flatFields = t.fields.flatMap { f =>
        flattenType(f.typ) match {
          case t2: TStruct =>
            t2.fields.map { f2 => (f.name + "." + f2.name, f2.typ) }

          case _ => Seq(f.name -> f.typ)
        }
      }

      TStruct(flatFields: _*)

    case _ => t
  }

  def flattenAnnotation(a: Annotation, t: Type): Annotation = t match {
    case t: TStruct =>
      val s =
        if (a == null)
          Seq.fill(t.fields.length)(null)
        else
          a.asInstanceOf[Row].toSeq

      val fs = (s, t.fields).zipped.flatMap { case (ai, f) =>
        f.typ match {
          case t: TStruct =>
            flattenAnnotation(ai, f.typ).asInstanceOf[Row].toSeq

          case _ =>
            Seq(ai)
        }
      }
      Row.fromSeq(fs)

    case _ => a
  }

  def apply(args: Any*): Annotation = Row.fromSeq(args)

  def fromSeq(values: Seq[Any]): Annotation = Row.fromSeq(values)
}

