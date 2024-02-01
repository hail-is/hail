package is.hail.types.virtual

import is.hail.annotations._
import is.hail.backend.HailStateManager
import is.hail.expr.ir.{Env, IRParser, IntArrayBuilder}
import is.hail.utils._

import scala.collection.JavaConverters._
import org.apache.spark.sql.Row
import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString

import scala.collection.mutable

class TStructSerializer extends CustomSerializer[TStruct](format =>
      (
        { case JString(s) => IRParser.parseStructType(s) },
        { case t: TStruct => JString(t.parsableString()) },
      )
    )

object TStruct {
  val empty: TStruct = TStruct()

  def apply(args: (String, Type)*): TStruct =
    TStruct(args
      .iterator
      .zipWithIndex
      .map { case ((n, t), i) => Field(n, t, i) }
      .toArray)

  def apply(names: java.util.List[String], types: java.util.List[Type]): TStruct = {
    val sNames = names.asScala.toArray
    val sTypes = types.asScala.toArray
    if (sNames.length != sTypes.length)
      fatal(
        s"number of names does not match number of types: found ${sNames.length} names and ${sTypes.length} types"
      )

    TStruct(sNames.zip(sTypes): _*)
  }

  def concat(struct1: TStruct, struct2: TStruct): TStruct = {
    struct2.fieldNames.foreach(field => assert(!struct1.hasField(field)))
    TStruct(struct1.fields ++ struct2.fields.map(field =>
      field.copy(index = field.index + struct1.size)
    ))
  }
}

final case class TStruct(fields: IndexedSeq[Field]) extends TBaseStruct {
  assert(fields.zipWithIndex.forall { case (f, i) => f.index == i })

  lazy val types: Array[Type] = fields.map(_.typ).toArray

  val fieldNames: Array[String] = {
    val seen = mutable.Set.empty[String]
    fields.toArray.map { f =>
      val name = f.name
      assert(seen.add(name), f"duplicate name '$name' found in '${_toPretty}'.")
      name
    }
  }

  def size: Int = fields.length

  override def truncate(newSize: Int): TStruct = TStruct(fields.take(newSize))

  override def mkOrdering(sm: HailStateManager, missingEqual: Boolean): ExtendedOrdering =
    TBaseStruct.getOrdering(sm, types, missingEqual)

  override def canCompare(other: Type): Boolean = other match {
    case t: TStruct => size == t.size && fields.zip(t.fields).forall { case (f1, f2) =>
          f1.name == f2.name && f1.typ.canCompare(f2.typ)
        }
    case _ => false
  }

  override def unify(concrete: Type): Boolean = concrete match {
    case TStruct(cfields) =>
      fields.length == cfields.length &&
      (fields, cfields).zipped.forall { case (f, cf) =>
        f.unify(cf)
      }
    case _ => false
  }

  override def subst() = TStruct(fields.map(f => f.copy(typ = f.typ.subst().asInstanceOf[Type])))

  def index(str: String): Option[Int] = fieldIdx.get(str)

  def selfField(name: String): Option[Field] = fieldIdx.get(name).map(i => fields(i))

  def hasField(name: String): Boolean = fieldIdx.contains(name)

  def field(name: String): Field = fields(fieldIdx(name))

  def fieldType(name: String): Type = types(fieldIdx(name))

  def fieldOption(path: IndexedSeq[String]): Option[Field] =
    if (path.isEmpty) None
    else (1 until path.length).foldLeft(selfField(path.head)) {
      case (Some(f), i) => f.typ match {
          case s: TStruct => s.selfField(path(i))
          case _ => return None
        }
      case _ => return None
    }

  override def queryTyped(p: List[String]): (Type, Querier) = {
    if (p.isEmpty)
      (this, identity[Annotation])
    else {
      selfField(p.head) match {
        case Some(f) =>
          val (t, q) = f.typ.queryTyped(p.tail)
          val localIndex = f.index
          (
            t,
            (a: Any) =>
              if (a == null)
                null
              else
                q(a.asInstanceOf[Row].get(localIndex)),
          )
        case None => throw new AnnotationPathException(s"struct has no field ${p.head}")
      }
    }
  }

  def insert(signature: Type, path: IndexedSeq[String]): (TStruct, Inserter) = {
    if (path.isEmpty)
      throw new IllegalArgumentException(s"Empty path to new field of type '$signature'.")

    val missing: Annotation =
      null.asInstanceOf[Annotation]

    def updateField(typ: TStruct, idx: Int)(f: Inserter)(a: Annotation, v: Any): Annotation =
      a match {
        case r: Row =>
          r.update(idx, f(r.get(idx), v))
        case _ =>
          val arr = new Array[Any](typ.size)
          arr.update(idx, f(missing, v))
          Row.fromSeq(arr)
      }

    def addField(typ: TStruct)(f: Inserter)(a: Annotation, v: Any): Annotation = {
      val arr = new Array[Any](typ.size + 1)
      a match {
        case r: Row =>
          for (i <- 0 until typ.size)
            arr.update(i, r.get(i))
        case _ =>
      }
      arr(typ.size) = f(missing, v)
      Row.fromSeq(arr)
    }

    val (newType, inserter) =
      path
        .view
        .scanLeft((this, identity[Type] _, identity[Inserter] _)) {
          case ((parent, _, _), name) =>
            parent.selfField(name) match {
              case Some(Field(name, t, idx)) =>
                (
                  t match {
                    case s: TStruct => s
                    case _ => TStruct.empty
                  },
                  typ => parent.updateKey(name, idx, typ),
                  updateField(parent, idx),
                )
              case None =>
                (
                  TStruct.empty,
                  typ => parent.appendKey(name, typ),
                  addField(parent),
                )
            }
        }
        .foldRight((signature, ((_, toIns) => toIns): Inserter)) {
          case ((_, insertField, transform), (newType, inserter)) =>
            (insertField(newType), transform(inserter))
        }

    (newType.asInstanceOf[TStruct], inserter)
  }

  def structInsert(signature: Type, p: IndexedSeq[String]): TStruct = {
    require(
      p.nonEmpty || signature.isInstanceOf[TStruct],
      s"tried to remap top-level struct to non-struct $signature",
    )
    val (t, _) = insert(signature, p)
    t
  }

  def updateKey(key: String, i: Int, sig: Type): TStruct = {
    assert(fieldIdx.contains(key))

    val newFields = Array.fill[Field](fields.length)(null)
    for (i <- fields.indices)
      newFields(i) = fields(i)
    newFields(i) = Field(key, sig, i)
    TStruct(newFields)
  }

  def deleteKey(key: String): TStruct = deleteKey(key, fieldIdx(key))

  def deleteKey(key: String, index: Int): TStruct = {
    assert(fieldIdx.contains(key))
    if (fields.length == 1)
      TStruct.empty
    else {
      val newFields = Array.fill[Field](fields.length - 1)(null)
      for (i <- 0 until index)
        newFields(i) = fields(i)
      for (i <- index + 1 until fields.length)
        newFields(i - 1) = fields(i).copy(index = i - 1)
      TStruct(newFields)
    }
  }

  def appendKey(key: String, sig: Type): TStruct = {
    assert(!fieldIdx.contains(key))
    val newFields = Array.fill[Field](fields.length + 1)(null)
    for (i <- fields.indices)
      newFields(i) = fields(i)
    newFields(fields.length) = Field(key, sig, fields.length)
    TStruct(newFields)
  }

  def annotate(other: TStruct): (TStruct, Merger) = {
    val newFieldsBuilder = new BoxedArrayBuilder[(String, Type)]()
    val fieldIdxBuilder = new IntArrayBuilder()
    // In fieldIdxBuilder, positive integers are field indices from the left.
    // Negative integers are the complement of field indices from the right.

    val rightFieldIdx = other.fields.map(f => f.name -> (f.index -> f.typ)).toMap
    val leftFields = fieldNames.toSet

    fields.foreach { f =>
      rightFieldIdx.get(f.name) match {
        case Some((rightIdx, typ)) =>
          fieldIdxBuilder += ~rightIdx
          newFieldsBuilder += f.name -> typ
        case None =>
          fieldIdxBuilder += f.index
          newFieldsBuilder += f.name -> f.typ
      }
    }
    other.fields.foreach { f =>
      if (!leftFields.contains(f.name)) {
        fieldIdxBuilder += ~f.index
        newFieldsBuilder += f.name -> f.typ
      }
    }

    val newStruct = TStruct(newFieldsBuilder.result(): _*)
    val fieldIdx = fieldIdxBuilder.result()
    val leftNulls = Row.fromSeq(Array.fill[Any](size)(null))
    val rightNulls = Row.fromSeq(Array.fill[Any](other.size)(null))

    val annotator = (a1: Annotation, a2: Annotation) => {
      if (a1 == null && a2 == null)
        null
      else {
        val leftValues = if (a1 == null) leftNulls else a1.asInstanceOf[Row]
        val rightValues = if (a2 == null) rightNulls else a2.asInstanceOf[Row]
        val resultValues = new Array[Any](fieldIdx.length)
        var i = 0
        while (i < resultValues.length) {
          val idx = fieldIdx(i)
          if (idx < 0)
            resultValues(i) = rightValues(~idx)
          else
            resultValues(i) = leftValues(idx)
          i += 1
        }
        Row.fromSeq(resultValues)
      }
    }
    newStruct -> annotator
  }

  def insertFields(fieldsToInsert: TraversableOnce[(String, Type)]): TStruct = {
    val ab = new BoxedArrayBuilder[Field](fields.length)
    var i = 0
    while (i < fields.length) {
      ab += fields(i)
      i += 1
    }
    val it = fieldsToInsert.toIterator
    while (it.hasNext) {
      val (name, typ) = it.next
      if (fieldIdx.contains(name)) {
        val j = fieldIdx(name)
        ab(j) = Field(name, typ, j)
      } else
        ab += Field(name, typ, ab.length)
    }
    TStruct(ab.result())
  }

  def rename(m: Map[String, String]): TStruct = {
    val newFieldsBuilder = new BoxedArrayBuilder[(String, Type)]()
    fields.foreach { fd =>
      val n = fd.name
      newFieldsBuilder += (m.getOrElse(n, n) -> fd.typ)
    }
    TStruct(newFieldsBuilder.result(): _*)
  }

  def filterSet(set: Set[String], include: Boolean = true): (TStruct, Deleter) = {
    val notFound = set.filter(name => selfField(name).isEmpty).map(prettyIdentifier)
    if (notFound.nonEmpty)
      fatal(
        s"""invalid struct filter operation: ${plural(notFound.size, s"field ${notFound.head}", s"fields [ ${notFound.mkString(", ")} ]")} not found
           |  Existing struct fields: [ ${fields.map(f => prettyIdentifier(f.name)).mkString(", ")} ]""".stripMargin
      )

    val fn = (f: Field) =>
      if (include)
        set.contains(f.name)
      else
        !set.contains(f.name)
    filter(fn)
  }

  def ++(that: TStruct): TStruct = {
    val overlapping = fields.map(_.name).toSet.intersect(
      that.fields.map(_.name).toSet
    )
    if (overlapping.nonEmpty)
      fatal(s"overlapping fields in struct concatenation: ${overlapping.mkString(", ")}")

    TStruct(fields.map(f => (f.name, f.typ)) ++ that.fields.map(f => (f.name, f.typ)): _*)
  }

  def filter(f: (Field) => Boolean): (TStruct, (Annotation) => Annotation) = {
    val included = fields.map(f)

    val newFields = fields.zip(included)
      .flatMap { case (field, incl) =>
        if (incl)
          Some(field)
        else
          None
      }

    val newSize = newFields.size

    val filterer = (a: Annotation) =>
      if (a == null)
        a
      else if (newSize == 0)
        Annotation.empty
      else {
        val r = a.asInstanceOf[Row]
        val newValues = included.zipWithIndex
          .flatMap {
            case (incl, i) =>
              if (incl)
                Some(r.get(i))
              else None
          }
        assert(newValues.length == newSize)
        Annotation.fromSeq(newValues)
      }

    (TStruct(newFields.zipWithIndex.map { case (f, i) => f.copy(index = i) }), filterer)
  }

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("struct{")
    fields.foreachBetween({ field =>
      sb.append(prettyIdentifier(field.name))
      sb.append(": ")
      field.typ.pyString(sb)
    })(sb.append(", "))
    sb.append('}')
  }

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = {
    if (compact) {
      sb.append("Struct{")
      fields.foreachBetween(_.pretty(sb, indent, compact))(sb += ',')
      sb += '}'
    } else {
      if (size == 0)
        sb.append("Struct { }")
      else {
        sb.append("Struct {")
        sb += '\n'
        fields.foreachBetween(_.pretty(sb, indent + 4, compact))(sb.append(",\n"))
        sb += '\n'
        sb.append(" " * indent)
        sb += '}'
      }
    }
  }

  def select(keep: IndexedSeq[String]): (TStruct, (Row) => Row) = {
    val t = TStruct(keep.map(n => n -> field(n).typ): _*)

    val keepIdx = keep.map(fieldIdx)
    val selectF: Row => Row = { r => Row.fromSeq(keepIdx.map(r.get)) }
    (t, selectF)
  }

  def typeAfterSelectNames(keep: IndexedSeq[String]): TStruct =
    TStruct(keep.map(n => n -> fieldType(n)): _*)

  def typeAfterSelect(keep: IndexedSeq[Int]): TStruct =
    TStruct(keep.map(i => fieldNames(i) -> types(i)): _*)

  def toEnv: Env[Type] = Env(fields.map(f => (f.name, f.typ)): _*)

  override def valueSubsetter(subtype: Type): Any => Any = {
    if (this == subtype)
      return identity

    val subStruct = subtype.asInstanceOf[TStruct]
    val subsetFields =
      subStruct.fields.map(f => (fieldIdx(f.name), fieldType(f.name).valueSubsetter(f.typ)))

    { (a: Any) =>
      val r = a.asInstanceOf[Row]
      Row.fromSeq(subsetFields.map { case (i, subset) => subset(r.get(i)) })
    }
  }

  def isSubsetOf(other: TStruct): Boolean =
    fields.forall(f => other.fieldIdx.get(f.name).exists(other.fields(_).typ == f.typ))

}
