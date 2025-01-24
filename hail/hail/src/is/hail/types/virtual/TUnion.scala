package is.hail.types.virtual

import is.hail.annotations.{Annotation, ExtendedOrdering}
import is.hail.backend.HailStateManager
import is.hail.check.Gen
import is.hail.expr.ir.IRParser
import is.hail.utils._

import scala.reflect.ClassTag

import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString

final case class Case(name: String, typ: Type, index: Int) {

  def unify(cf: Case): Boolean =
    name == cf.name &&
      typ.unify(cf.typ) &&
      index == cf.index

  def pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = {
    if (compact) {
      sb.append(prettyIdentifier(name))
      sb.append(":")
    } else {
      sb.append(" " * indent)
      sb.append(prettyIdentifier(name))
      sb.append(": ")
    }
    typ.pretty(sb, indent, compact)
  }
}

class TUnionSerializer extends CustomSerializer[TUnion](format =>
      (
        { case JString(s) => IRParser.parseUnionType(s) },
        { case t: TUnion => JString(t.parsableString()) },
      )
    )

object TUnion {
  val empty: TUnion = TUnion(FastSeq())

  def apply(args: (String, Type)*): TUnion =
    TUnion(args
      .iterator
      .zipWithIndex
      .map { case ((n, t), i) => Case(n, t, i) }
      .toArray)
}

final case class TUnion(cases: IndexedSeq[Case]) extends Type {
  lazy val types: Array[Type] = cases.map(_.typ).toArray

  lazy val caseIdx: collection.Map[String, Int] = toMapFast(cases)(_.name, _.index)

  lazy val caseNames: Array[String] = cases.map(_.name).toArray

  def size: Int = cases.length

  override def unify(concrete: Type): Boolean = concrete match {
    case TUnion(cfields) =>
      cases.length == cfields.length &&
      (cases, cfields).zipped.forall { case (f, cf) =>
        f.unify(cf)
      }
    case _ => false
  }

  override def subst(): TUnion = TUnion(cases.map(f => f.copy(typ = f.typ.subst())))

  def index(str: String): Option[Int] = caseIdx.get(str)

  def selfCase(name: String): Option[Case] = caseIdx.get(name).map(i => cases(i))

  def hasCase(name: String): Boolean = caseIdx.contains(name)

  def getCase(name: String): Case = cases(caseIdx(name))

  def fieldType(name: String): Type = types(caseIdx(name))

  def rename(m: Map[String, String]): TUnion = {
    val newFieldsBuilder = new BoxedArrayBuilder[(String, Type)]()
    cases.foreach { fd =>
      val n = fd.name
      newFieldsBuilder += (m.getOrElse(n, n) -> fd.typ)
    }
    TUnion(newFieldsBuilder.result(): _*)
  }

  def _toPretty: String = {
    val sb = new StringBuilder
    _pretty(sb, 0, compact = true)
    sb.result()
  }

  override def _typeCheck(a: Any): Boolean = ???

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("union{")
    cases.foreachBetween({ field =>
      sb.append(prettyIdentifier(field.name))
      sb.append(": ")
      field.typ.pyString(sb)
    })(sb.append(", "))
    sb.append('}')
  }

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = {
    if (compact) {
      sb.append("Union{")
      cases.foreachBetween(_.pretty(sb, indent, compact))(sb += ',')
      sb += '}'
    } else {
      if (size == 0)
        sb.append("Union { }")
      else {
        sb.append("Union {")
        sb += '\n'
        cases.foreachBetween(_.pretty(sb, indent + 4, compact))(sb.append(",\n"))
        sb += '\n'
        sb.append(" " * indent)
        sb += '}'
      }
    }
  }

  override def genNonmissingValue(sm: HailStateManager): Gen[Annotation] = ???

  override def scalaClassTag: ClassTag[AnyRef] = ???

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering = ???

  override def isIsomorphicTo(t: Type): Boolean =
    t match {
      case u: TUnion =>
        size == u.size &&
        (cases, u.cases).zipped.forall(_.typ isIsomorphicTo _.typ)
      case _ =>
        false
    }
}
