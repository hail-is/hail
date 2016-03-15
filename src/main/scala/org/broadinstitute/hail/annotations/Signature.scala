package org.broadinstitute.hail.annotations

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.Utils._

object Signature {
  def empty: Signature = EmptySignature()
}

abstract class Signature extends Serializable {
  def dType: expr.Type

  def dType(path: List[String]): expr.Type = {
    if (path.isEmpty)
      dType
    else
      throw new AnnotationPathException()
  }

  def typeCheck(a: Annotation): Boolean = {
    dType.typeCheck(a)
  }

  def parser(missing: Set[String], colName: String): String => Annotation = {
      dType match {
        case expr.TDouble =>
          (v: String) =>
            try {
              if (missing(v)) null else v.toDouble
            } catch {
              case e: java.lang.NumberFormatException =>
                fatal( s"""java.lang.NumberFormatException: tried to convert "$v" to Double in column "$colName" """)
            }
        case expr.TInt =>
          (v: String) =>
            try {
              if (missing(v)) null else v.toInt
            } catch {
              case e: java.lang.NumberFormatException =>
                fatal( s"""java.lang.NumberFormatException: tried to convert "$v" to Int in column "$colName" """)
            }
        case expr.TBoolean =>
          (v: String) =>
            try {
              if (missing(v)) null else v.toBoolean
            } catch {
              case e: java.lang.IllegalArgumentException =>
                fatal( s"""java.lang.IllegalArgumentException: tried to convert "$v" to Boolean in column "$colName" """)
            }
        case expr.TString =>
          (v: String) =>
            if (missing(v)) null else v
        case _ => throw new UnsupportedOperationException(s"Cannot generage a parser for $dType")
      }
  }

  def getOption(fields: String*): Option[Signature] = getOption(fields.toList)

  def getOption(path: List[String]): Option[Signature] = {
    if (path.isEmpty)
      Some(this)
    else
      None
  }

  def delete(fields: String*): (Signature, Deleter) = delete(fields.toList)

  def delete(path: List[String]): (Signature, Deleter) = {
    if (path.nonEmpty)
      throw new AnnotationPathException()
    else
      (EmptySignature(), a => null)
  }

  def insert(signature: Signature, fields: String*): (Signature, Inserter) = insert(signature, fields.toList)

  def insert(signature: Signature, path: List[String]): (Signature, Inserter) = {
    if (path.nonEmpty)
      StructSignature(Map.empty).insert(signature, path)
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

  def schema: DataType = dType.schema

  def printSchema(key: String, nSpace: Int, path: String): String = {
    s"""${" " * nSpace}$key: $dType"""
  }

  def isEmpty: Boolean = false
}

case class StructSignature(m: Map[String, (Int, Signature)]) extends Signature {
  override def dType: expr.Type = expr.TStruct(m.map { case (k, (i, v)) => (k, (i, v.dType)) })

  def size: Int = m.size

  override def getOption(path: List[String]): Option[Signature] = {
    if (path.isEmpty)
      Some(this)
    else
      m.get(path.head)
        .flatMap(_._2.getOption(path.tail))
  }

  override def query(p: List[String]): Querier = {
    if (p.isEmpty)
      a => Option(a)
    else {
      m.get(p.head) match {
        case Some((i, sig)) =>
          val q = sig.query(p.tail)
          a =>
            if (a == null)
              None
            else
              q(a.asInstanceOf[Row].get(i))
        case None => throw new AnnotationPathException()
      }
    }
  }

  override def delete(p: List[String]): (Signature, Deleter) = {
    if (p.isEmpty)
      (EmptySignature(), a => null)
    else {
      val key = p.head
      val (keyIndex, keyS) = m.get(key) match {
        case Some((i, s)) => (i, s)
        case None => throw new AnnotationPathException(s"$key not found")
      }
      val (newKeyS, d) = keyS.delete(p.tail)
      val newSignature: Signature =
        if (newKeyS.isEmpty)
          deleteKey(key, keyIndex)
        else
          updateKey(key, keyIndex, newKeyS)

      val localDeleteFromRow = newKeyS.isEmpty

      val f: Deleter = { a =>
        if (a == null)
          null
        else {
          val r = a.asInstanceOf[Row]

          if (localDeleteFromRow)
            r.delete(keyIndex)
          else
            r.update(keyIndex, d(r.get(keyIndex)))
        }
      }
      (newSignature, f)
    }
  }

  override def insert(signature: Signature, p: List[String]): (Signature, Inserter) = {
    if (p.isEmpty)
      (signature, (a, toIns) => toIns.orNull)
    else {
      val key = p.head
      val keyIndexS = m.get(key)
      val keyIndex = keyIndexS.map(_._1)
      val (newKeyS, keyF) = keyIndexS
        .map(_._2)
        .getOrElse(StructSignature.empty)
        .insert(signature, p.tail)

      val newSignature = keyIndex match {
        case Some(i) => updateKey(key, i, newKeyS)
        case None => appendKey(key, newKeyS)
      }

      val localSize = m.size

      val f: Inserter = (a, toIns) => {
        val r = if (a == null)
          Row.fromSeq(Array.fill[Any](localSize)(null))
        else
          a.asInstanceOf[Row]
        keyIndex match {
          case Some(i) => r.update(i, keyF(r.get(i), toIns))
          case None => r.append(keyF(null, toIns))
        }
      }
      (newSignature, f)
    }
  }

  override def printSchema(key: String, nSpace: Int, path: String): String = {
    val spaces = " " * nSpace
    s"""$spaces$key: $path.<identifier>\n""" +
      m.toArray
        .sortBy {
          case (k, (i, v)) => i
        }
        .map {
          case (k, (i, v)) => v.printSchema(s"""$k""", nSpace + 2, path + "." + k)
          //          keep for future debugging:
          //          case (k, (i, v)) => v.printSchema(s"""[$i] $k""", nSpace + 2, path + "." + k)
        }
        .mkString("\n")
  }

  def updateKey(key: String, i: Int, sig: Signature): Signature = {
    assert(m.contains(key))
    StructSignature(m + ((key, (i, sig))))
  }

  def deleteKey(key: String, index: Int): Signature = {
    assert(m.contains(key))
    if (m.size == 1)
      EmptySignature()
    else
      StructSignature((m - key).mapValues { case (i, s) =>
        if (i > index)
          (i - 1, s)
        else
          (i, s)
      }.force)
  }

  def appendKey(key: String, sig: Signature): StructSignature = {
    assert(!m.contains(key))
    StructSignature(m + ((key, (m.size, sig))))
  }
}

object StructSignature {
  def empty: StructSignature = StructSignature(Map.empty)
}

case class EmptySignature(dType: expr.Type = expr.TBoolean) extends Signature {
  override def schema: DataType = BooleanType

  override def printSchema(key: String, nSpace: Int, path: String): String = s"""${" " * nSpace}$key: EMPTY"""

  override def query(path: List[String]): Querier = { a =>
    assert(a == null)
    None
  }

  override def getOption(path: List[String]): Option[Signature] = None

  override def delete(path: List[String]): (Signature, Deleter) = (EmptySignature(), { a =>
    assert(a == null)
    null
  })

  override def isEmpty: Boolean = true
}

case class SimpleSignature(dType: expr.Type) extends Signature

object SimpleSignature {
  def apply(s: String): SimpleSignature = {
    s match {
      case "Double" => SimpleSignature(expr.TDouble)
      case "Int" => SimpleSignature(expr.TInt)
      case "Boolean" => SimpleSignature(expr.TBoolean)
      case "String" => SimpleSignature(expr.TString)
      case _ => fatal(
        s"""Unrecognized type "$s".  Hail supports parsing the following types in annotations:
            |  - Double (floating point number)
            |  - Int  (integer)
            |  - Boolean
            |  - String
            |
            |  Note that the above types are case sensitive.""".stripMargin)
    }
  }
}