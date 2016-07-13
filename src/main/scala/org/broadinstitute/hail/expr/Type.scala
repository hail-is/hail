package org.broadinstitute.hail.expr

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.DataType
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.{Annotation, AnnotationPathException, _}
import org.broadinstitute.hail.check.{Arbitrary, Gen}
import org.broadinstitute.hail.variant.{AltAllele, Genotype, Variant}
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.reflect.ClassTag


trait Parsable {
  def parse(s: String): Annotation
}

sealed abstract class BaseType extends Serializable {
  def typeCheck(a: Any): Boolean
}

object Type {
  val genScalar = Gen.oneOf[Type](TBoolean, TChar, TInt, TLong, TFloat, TDouble, TString,
    TVariant, TAltAllele, TGenotype)

  def genSized(size: Int): Gen[Type] = {
    if (size < 1)
      Gen.const(TStruct.empty)
    else if (size < 2)
      genScalar
    else
      Gen.oneOfGen(genScalar,
        genArb.resize(size - 1).map(TArray),
        genArb.resize(size - 1).map(TSet),
        genArb.resize(size - 1).map(TDict),
        Gen.buildableOf[Array[(String, Type)], (String, Type)](
          Gen.zip(Gen.identifier,
            genArb))
          .filter(fields => fields.map(_._1).areDistinct())
          .map(fields => TStruct(fields: _*)))
  }

  def genArb: Gen[Type] = Gen.sized(genSized)

  implicit def arbType = Arbitrary(genArb)
}

abstract class Type extends BaseType {
  def getAsOption[T](fields: String*)(implicit ct: ClassTag[T]): Option[T] = {
    getOption(fields: _*)
      .flatMap { t =>
        if (ct.runtimeClass.isInstance(t))
          Some(t.asInstanceOf[T])
        else
          None
      }
  }

  def getOption(fields: String*): Option[Type] = getOption(fields.toList)

  def getOption(path: List[String]): Option[Type] = {
    if (path.isEmpty)
      Some(this)
    else
      None
  }

  def delete(fields: String*): (Type, Deleter) = delete(fields.toList)

  def delete(path: List[String]): (Type, Deleter) = {
    if (path.nonEmpty)
      throw new AnnotationPathException()
    else
      (TStruct.empty, a => Annotation.empty)
  }

  def insert(signature: Type, fields: String*): (Type, Inserter) = insert(signature, fields.toList)

  def insert(signature: Type, path: List[String]): (Type, Inserter) = {
    if (path.nonEmpty)
      TStruct.empty.insert(signature, path)
    else
      (signature, (a, toIns) => toIns.orNull)
  }

  def query(fields: String*): Querier = query(fields.toList)

  def query(path: List[String]): Querier = {
    if (path.nonEmpty)
      throw new AnnotationPathException()
    else
      a => Option(a)
  }

  def pretty(sb: StringBuilder, indent: Int, printAttrs: Boolean = false) {
    sb.append(toString)
  }

  def fieldOption(fields: String*): Option[Field] = fieldOption(fields.toList)

  def fieldOption(path: List[String]): Option[Field] =
    None

  def schema: DataType = SparkAnnotationImpex.exportType(this)

  def str(a: Annotation): String = if (a == null) "NA" else a.toString

  def toJSON(a: Annotation): JValue = JSONAnnotationImpex.exportAnnotation(a, this)

  def genValue: Gen[Annotation] = Gen.const(Annotation.empty)
}

case object TBoolean extends Type with Parsable {
  override def toString = "Boolean"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Boolean]

  def parse(s: String): Annotation = s.toBoolean

  override def genValue: Gen[Annotation] = Gen.arbBoolean
}

case object TChar extends Type {
  override def toString = "Char"

  def typeCheck(a: Any): Boolean = a == null || (a.isInstanceOf[String]
    && a.asInstanceOf[String].length == 1)

  override def genValue: Gen[Annotation] = Gen.arbString
    .filter(_.nonEmpty)
    .map(s => s.substring(0, 1))
}

object TNumeric {
  def promoteNumeric(types: Set[TNumeric]): Type = {
    if (types.size == 1)
      types.head
    else if (types(TDouble))
      TDouble
    else if (types(TFloat))
      TFloat
    else {
      assert(types == Set(TLong))
      TLong
    }
  }
}

abstract class TNumeric extends Type {
  def makeDouble[U](a: Any): Double
}

abstract class TIntegral extends TNumeric {
  def makeLong[U](a: Any): Long
}

case object TInt extends TIntegral with Parsable {
  override def toString = "Int"

  def makeDouble[U](a: Any): Double = a.asInstanceOf[Int].toDouble

  def makeLong[U](a: Any): Long = a.asInstanceOf[Int].toLong

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Int]

  def parse(s: String): Annotation = s.toInt

  override def genValue: Gen[Annotation] = Gen.arbInt
}

case object TLong extends TIntegral with Parsable {
  override def toString = "Long"

  def makeDouble[U](a: Any): Double = a.asInstanceOf[Long].toDouble

  def makeLong[U](a: Any): Long = a.asInstanceOf[Long]

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Long]

  def parse(s: String): Annotation = s.toLong

  override def genValue: Gen[Annotation] = Gen.arbLong
}

case object TFloat extends TNumeric with Parsable {
  override def toString = "Float"

  def makeDouble[U](a: Any): Double = a.asInstanceOf[Float].toDouble

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Float]

  def parse(s: String): Annotation = s.toFloat

  override def genValue: Gen[Annotation] = Gen.arbDouble.map(_.toFloat)
}

case object TDouble extends TNumeric with Parsable {
  override def toString = "Double"

  def makeDouble[U](a: Any): Double = a.asInstanceOf[Double]

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Double]

  def parse(s: String): Annotation = s.toDouble

  override def genValue: Gen[Annotation] = Gen.arbDouble
}

case object TString extends Type with Parsable {
  override def toString = "String"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[String]

  def parse(s: String): Annotation = s

  override def genValue: Gen[Annotation] = Gen.arbString
}


case class TAggregable(ec: EvalContext) extends BaseType {
  override def toString = "Aggregable"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Iterable[_]]
}

abstract class TIterable extends Type {
  def elementType: Type
}

case class TArray(elementType: Type) extends TIterable {
  override def toString = s"Array[$elementType]"

  override def pretty(sb: StringBuilder, indent: Int, printAttrs: Boolean) {
    sb.append("Array[")
    elementType.pretty(sb, indent, printAttrs)
    sb.append("]")
  }

  def typeCheck(a: Any): Boolean = a == null || (a.isInstanceOf[IndexedSeq[_]] &&
    a.asInstanceOf[IndexedSeq[_]].forall(elementType.typeCheck))

  override def str(a: Annotation): String = compact(toJSON(a))

  override def genValue: Gen[Annotation] = Gen.buildableOf[IndexedSeq[Annotation], Annotation](
    elementType.genValue)
}

case class TSet(elementType: Type) extends TIterable {
  override def toString = s"Set[$elementType]"

  def typeCheck(a: Any): Boolean =
    a == null || (a.isInstanceOf[Set[_]] && a.asInstanceOf[Set[_]].forall(elementType.typeCheck))

  override def pretty(sb: StringBuilder, indent: Int, printAttrs: Boolean) {
    sb.append("Set[")
    elementType.pretty(sb, indent, printAttrs)
    sb.append("]")
  }

  override def str(a: Annotation): String = compact(toJSON(a))

  override def genValue: Gen[Annotation] = Gen.buildableOf[Set[Annotation], Annotation](
    elementType.genValue)
}

case class TDict(elementType: Type) extends Type {
  override def toString = s"Dict[$elementType]"

  override def pretty(sb: StringBuilder, indent: Int, printAttrs: Boolean) {
    sb.append("Dict[")
    elementType.pretty(sb, indent, printAttrs)
    sb.append("]")
  }

  def typeCheck(a: Any): Boolean = a == null || (a.isInstanceOf[Map[_, _]] &&
    a.asInstanceOf[Map[_, _]].forall { case (k, v) => k.isInstanceOf[String] && elementType.typeCheck(v) })

  override def str(a: Annotation): String = compact(toJSON(a))

  override def genValue: Gen[Annotation] = Gen.buildableOf[Map[String, Annotation], (String, Annotation)](
    Gen.zip(Gen.arbString, elementType.genValue))
}

case object TSample extends Type {
  override def toString = "Sample"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[String]

  override def genValue: Gen[Annotation] = Gen.identifier
}

case object TGenotype extends Type {
  override def toString = "Genotype"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Genotype]

  override def genValue: Gen[Annotation] = Genotype.genArb
}

case object TAltAllele extends Type {
  override def toString = "AltAllele"

  def typeCheck(a: Any): Boolean = a == null || a == null || a.isInstanceOf[AltAllele]

  override def genValue: Gen[Annotation] = AltAllele.gen
}

case object TVariant extends Type {
  override def toString = "Variant"

  def typeCheck(a: Any): Boolean = a == null || a.isInstanceOf[Variant]

  override def genValue: Gen[Annotation] = Variant.gen
}

case class Field(name: String, `type`: Type,
  index: Int,
  attrs: Map[String, String] = Map.empty) {
  def attr(s: String): Option[String] = attrs.get(s)

  def pretty(sb: StringBuilder, indent: Int, printAttrs: Boolean) {
    sb.append(" " * indent)
    sb.append(prettyIdentifier(name))
    sb.append(": ")
    `type`.pretty(sb, indent, printAttrs)
    if (printAttrs) {
      if (attrs.nonEmpty)
        sb += '\n'
      attrs.foreachBetween { attr =>
        sb.append(" " * (indent + 2))
        sb += '@'
        sb.append(prettyIdentifier(attr._1))
        sb.append("=\"")
        sb.append(escapeString(attr._2))
        sb += '"'
      }(() => sb += '\n')
    }
  }
}

object TStruct {
  def empty: TStruct = TStruct(Array.empty[Field])

  def apply(args: (String, Type)*): TStruct =
    TStruct(args
      .iterator
      .zipWithIndex
      .map { case ((n, t), i) => Field(n, t, i) }
      .toArray)
}

case class TStruct(fields: IndexedSeq[Field]) extends Type {
  val fieldIdx: Map[String, Int] =
    fields.map(f => (f.name, f.index)).toMap

  def selfField(name: String): Option[Field] = fieldIdx.get(name).map(i => fields(i))

  def size: Int = fields.length

  override def getOption(path: List[String]): Option[Type] =
    if (path.isEmpty)
      Some(this)
    else
      selfField(path.head).map(_.`type`).flatMap(t => t.getOption(path.tail))

  override def fieldOption(path: List[String]): Option[Field] =
    if (path.isEmpty)
      None
    else {
      val f = selfField(path.head)
      if (path.length == 1)
        f
      else
        f.flatMap(_.`type`.fieldOption(path.tail))
    }

  override def query(p: List[String]): Querier = {
    if (p.isEmpty)
      a => Option(a)
    else {
      selfField(p.head) match {
        case Some(f) =>
          val q = f.`type`.query(p.tail)
          val localIndex = f.index
          a =>
            if (a == Annotation.empty)
              None
            else
              q(a.asInstanceOf[Row].get(localIndex))
        case None => throw new AnnotationPathException()
      }
    }
  }

  override def delete(p: List[String]): (Type, Deleter) = {
    if (p.isEmpty)
      (TStruct.empty, a => Annotation.empty)
    else {
      val key = p.head
      val f = selfField(key) match {
        case Some(f) => f
        case None => throw new AnnotationPathException(s"$key not found")
      }
      val index = f.index
      val (newFieldType, d) = f.`type`.delete(p.tail)
      val newType: Type =
        if (newFieldType == TStruct.empty)
          deleteKey(key, f.index)
        else
          updateKey(key, f.index, newFieldType)

      val localDeleteFromRow = newFieldType == TStruct.empty

      val deleter: Deleter = { a =>
        if (a == Annotation.empty)
          Annotation.empty
        else {
          val r = a.asInstanceOf[Row]

          if (localDeleteFromRow)
            r.delete(index)
          else
            r.update(index, d(r.get(index)))
        }
      }
      (newType, deleter)
    }
  }

  override def insert(signature: Type, p: List[String]): (Type, Inserter) = {
    if (p.isEmpty)
      (signature, (a, toIns) => toIns.orNull)
    else {
      val key = p.head
      val f = selfField(key)
      val keyIndex = f.map(_.index)
      val (newKeyType, keyF) = f
        .map(_.`type`)
        .getOrElse(TStruct.empty)
        .insert(signature, p.tail)

      val newSignature = keyIndex match {
        case Some(i) => updateKey(key, i, newKeyType)
        case None => appendKey(key, newKeyType)
      }

      val localSize = fields.size

      val inserter: Inserter = (a, toIns) => {
        val r = if (a == null || localSize == 0) // localsize == 0 catches cases where we overwrite a path
          Row.fromSeq(Array.fill[Any](localSize)(null))
        else
          a.asInstanceOf[Row]
        keyIndex match {
          case Some(i) => r.update(i, keyF(r.get(i), toIns))
          case None => r.append(keyF(Annotation.empty, toIns))
        }
      }
      (newSignature, inserter)
    }
  }

  def updateKey(key: String, i: Int, sig: Type): Type = {
    assert(fieldIdx.contains(key))

    val newFields = Array.fill[Field](fields.length)(null)
    for (i <- fields.indices)
      newFields(i) = fields(i)
    newFields(i) = Field(key, sig, i)
    TStruct(newFields)
  }

  def deleteKey(key: String, index: Int): Type = {
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

  override def toString = if (size == 0) "Empty" else "Struct"

  override def pretty(sb: StringBuilder, indent: Int, printAttrs: Boolean) {
    if (size == 0)
      sb.append("Empty")
    else {
      sb.append("Struct {")
      sb += '\n'
      fields.foreachBetween(f => {
        f.pretty(sb, indent + 4, printAttrs)
      })(() => {
        sb += ','
        sb += '\n'
      })
      sb += '\n'
      sb.append(" " * indent)
      sb += '}'
    }
  }

  override def typeCheck(a: Any): Boolean =
    if (fields.isEmpty)
      a == null
    else a == null ||
      a.isInstanceOf[Row] &&
        a.asInstanceOf[Row].toSeq.zip(fields).forall { case (v, f) => f.`type`.typeCheck(v) }

  override def str(a: Annotation): String = compact(toJSON(a))

  override def genValue: Gen[Annotation] = {
    if (size == 0)
      Gen.const[Annotation](Annotation.empty)
    else
      Gen.sequence[IndexedSeq[Annotation], Annotation](
        fields.map(f => f.`type`.genValue))
        .map(a => Annotation(a: _*))
  }
}
