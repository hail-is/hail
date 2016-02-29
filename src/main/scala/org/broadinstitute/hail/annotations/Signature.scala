package org.broadinstitute.hail.annotations

import org.apache.spark.sql.types._
import org.broadinstitute.hail.expr
//import scala.collection.mutable

import scala.collection.mutable

case class Path(p: List[String]) {
  def empty: Boolean = p.isEmpty
  def tail: Path = Path(p.tail)
  def head: String = p.head
}

abstract class Signature {
  def dType: expr.Type

  def dType(path: Path): expr.Type = {
    if (path.empty)
      dType
    else
      throw new AnnotationPathException(s"invalid path: ${path.p.mkString(",")}")
  }

  def delete(path: Path): (Signature, Deleter)

  def query(p: Path, index: Int = -1): Querier = {
    if (p.empty)
      a =>
      a.getAs("hello")
    else
      throw new AnnotationPathException()
  }

  def query(p: Path, list: List[Int]): List[Int] = {
    if (p.empty)
      list
    else
      throw new AnnotationPathException()


  }
}


case class StructSignature(m: Map[String, (Int, Signature)]) extends Signature {
  def dType: expr.Type = null

  def query(query: Iterable[String]): Array[Int] = {
    val (path, sigs) = query.take(query.size - 1).foldLeft((mutable.ArrayBuilder.make[Int], this)) { case ((arr, sig), key) =>
      val next = sig.get[StructSignature](key)
      arr += next.index
      (arr, next)
    }
    path += sigs
      .get[Signature](query.last)
      .index

    path.result()
  }

  override def query(p: Path): Querier = {
    if (p.empty)
      a => Some(a)
    else {
      val f = p.head
      m.get(f) match {
        case Some((i, s)) =>
          val q = s.query(p.tail)
          a => q(a.getAs[Annotation](i))
        case None =>
          // f not a member of m
          a => None
      }
    }
  }

  def contains(elem: String): Boolean = m.contains(elem)

  def contains(path: Iterable[String]): Boolean = {
    path
      .take(path.size - 1)
      .foldLeft((m, true)) { case ((map, continue), key) =>
        if (!continue)
          (map, continue)
        else
          map.get(key) match {
            case Some(sigs: StructSignature) => (sigs.m, true)
            case _ => (m, false)
          }
      }
      ._1
      .contains(path.last)
  }

  def get[T](key: String): T = m(key).asInstanceOf[T]

  def getOption[T](key: String): Option[T] = m.get(key).map(_.asInstanceOf[T])

  def nextIndex(): Int = m.size

  def getSchema: StructType = {
    //    println(m)
    //    m.foreach(println)
    StructType(m
      .toArray
      .sortBy { case (key, (index, sig)) => index }
      .map { case (key, (index, sig)) => sig match {
        case sigs: StructSignature => StructField(key, sigs.getSchema, true)
        case sig => StructSignature.toSchemaType(key, sig)
      }}
    )
  }
}

object StructSignature {
  def empty(): StructSignature = StructSignature(Map.empty[String, Signature])

  def toSchemaType(key: String, signature: Signature): StructField = {
    signature match {
      case sigs: StructSignature => StructField(key, sigs.getSchema, true)
      case _ => signature.dType match {
        case expr.TArray(expr.TInt) => StructField(key, ArrayType(IntegerType), true)
        case expr.TArray(expr.TDouble) => StructField(key, ArrayType(DoubleType), true)
        case expr.TArray(expr.TString) => StructField(key, ArrayType(StringType), true)
        case expr.TString => StructField(key, StringType, true)
        case expr.TInt => StructField(key, IntegerType, true)
        case expr.TLong => StructField(key, LongType, true)
        case expr.TDouble => StructField(key, DoubleType, true)
        case expr.TFloat => StructField(key, FloatType, true)
        case expr.TSet(expr.TInt) => StructField(key, ArrayType(IntegerType), true)
        case expr.TSet(expr.TString) => StructField(key, ArrayType(StringType), true)
        case expr.TBoolean => StructField(key, BooleanType, true)
        case expr.TChar => StructField(key, StringType, true)
      }
    }
  }
}

case class SimpleSignature(dType: expr.Type) extends Signature