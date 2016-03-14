package org.broadinstitute.hail.annotations

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.Utils._

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

  def getOption(fields: String*): Option[Signature] = getOption(fields.toList)

  def getOption(path: List[String]): Option[Signature] = {
    if (path.isEmpty)
      Some(this)
    else
      None
  }

  def delete(fields: String*): (Signature, Deleter) = delete(fields.toList)

  def delete(path: List[String]): (Signature, Deleter) = {
    if (path.nonEmpty) {
      throw new AnnotationPathException()
    }
    else
      (EmptySignature(), a => null)
  }

  def insert(signature: Signature, fields: String*): (Signature, Inserter) = insert(signature, fields.toList)

  def insert(signature: Signature, path: List[String]): (Signature, Inserter) = {
    if (path.nonEmpty) {
      StructSignature(Map.empty).insert(signature, path)
    } else
      (signature, (a, toIns) => toIns.orNull)
  }

  def query(fields: String*): Querier = query(fields.toList)

  def query(path: List[String]): Querier = {
    if (path.nonEmpty)
      throw new AnnotationPathException()
    else
      a => Option(a)
  }

  def getSchema: DataType = {
    dType match {
      case expr.TArray(expr.TInt) => ArrayType(IntegerType)
      case expr.TArray(expr.TDouble) => ArrayType(DoubleType)
      case expr.TArray(expr.TString) => ArrayType(StringType)
      case expr.TString => StringType
      case expr.TInt => IntegerType
      case expr.TLong => LongType
      case expr.TDouble => DoubleType
      case expr.TFloat => FloatType
      case expr.TSet(expr.TInt) => ArrayType(IntegerType)
      case expr.TSet(expr.TString) => ArrayType(StringType)
      case expr.TBoolean => BooleanType
      case expr.TChar => StringType
      case _ => throw new UnsupportedOperationException()
    }
  }

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
      a => Some(a)
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
      val ret = m.get(key)
      val (index, sigToDelete) = ret.getOrElse(throw new AnnotationPathException(s"$key not found"))
      val (newS, d) = sigToDelete.delete(p.tail)
      val newSignature: Signature = if (m.size == 1)
        EmptySignature()
      else
        StructSignature((m - key).mapValues { case (i, s) =>
          if (i > index)
            (i - 1, s)
          else
            (i, s)
        }.force)

      val f: Deleter = a => {
        val r = if (a == null)
          null
        else
          a.asInstanceOf[Row]

        if (newSignature.isEmpty)
          null
        else if (p.length > 1)
          r.update(index, d(r.get(index)))
        else
          r.delete(index)
      }
      (newSignature, f)
    }
  }

  override def insert(signature: Signature, p: List[String]): (Signature, Inserter) = {
    if (p.isEmpty)
      (signature, (a, toIns) => toIns.orNull)
    else {
      val key = p.head

      val ret = m.get(key)
      val sigIndex = ret.map(_._1)
      val (ss, ff) = ret
        .map(_._2)
        .getOrElse(StructSignature(Map.empty[String, (Int, Signature)]))
        .insert(signature, p.tail)

      val newStruct = StructSignature(m + ((key, (sigIndex.getOrElse(m.size), ss))))

      val f: Inserter = (a, toIns) => {
        val r = if (a == null)
          Row.fromSeq(Array.fill[Any](m.size)(null))
        else
          a.asInstanceOf[Row]
        sigIndex match {
          case Some(i) => r.update(i, ff(r.get(i), toIns))
          case None => r.append(ff(null, toIns))
        }
      }
      (newStruct, f)
    }
  }

  override def getSchema: DataType = {
    val s =
      StructType(m
        .toArray
        .sortBy {
          case (key, (index, sig)) => index
        }
        .map {
          case (key, (index, sig)) =>
            StructField(key, sig.getSchema, nullable = true)
        }
      )
    assert(s.length > 0)
    s
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
}

case class EmptySignature(dType: expr.Type = expr.TBoolean) extends Signature {
  override def getSchema: DataType = BooleanType

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