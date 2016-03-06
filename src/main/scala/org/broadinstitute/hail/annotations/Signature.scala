package org.broadinstitute.hail.annotations

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.RichRow._

import scala.collection.mutable

//import scala.collection.mutable

//
//case class Path(p: List[String]) {
//  def empty: Boolean = p.isEmpty
//  def tail: Path = Path(p.tail)
//  def head: String = p.head
//  def length: Int = p.length
//  def prepend(i: String): Path = Path(p.::(i)
//}
//
//case class IndexPath(p: List[Int]) {
//  def empty: Boolean = p.isEmpty
//  def tail: IndexPath = IndexPath(p.tail)
//  def head: Int = p.head
//  def length: Int = p.length
//  def prepend(i: Int): IndexPath = IndexPath(p.::(i))
//}

abstract class Signature {
  def dType: expr.Type

  def dType(path: List[String]): expr.Type = {
    if (path.isEmpty)
      dType
    else
      throw new AnnotationPathException()
  }

  def getOption(path: List[String]): Option[Signature] = {
    if (path.isEmpty)
      Some(this)
    else
      None
  }

  def delete(path: List[String]): (Signature, Deleter) = {
    if (path.nonEmpty) {
      throw new AnnotationPathException()
    }
    else
      (null, null)
  }

  def insert(path: List[String], signature: Signature): (Signature, Inserter) = {
    if (path.nonEmpty) {
      StructSignature(Map.empty[String, (Int, Signature)]).insert(path.tail, signature)
    }
    else
      (this, (a, toIns) => {
        toIns.getOrElse(null)
      })
  }

  def insertBefore(path: List[String], signature: Signature): (Signature, Inserter) = {
    throw new AnnotationPathException()
  }

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

  def printSchema(path: String): String = {
    s"""$path: $dType"""
  }
}

case class StructSignature(m: Map[String, (Int, Signature)]) extends Signature {
  override def dType: expr.Type = null

  def size: Int = m.size

  override def getOption(path: List[String]): Option[Signature] = {
    if (path.isEmpty)
      Some(this)
    else
      m.get(path.head)
        .flatMap(_._2.getOption(path.tail))
  }

  override def query(p: List[String]): Querier = {
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

  override def delete(p: List[String]): (StructSignature, Deleter) = {
    if (p.isEmpty)
      (null, null)
    else {
      val key = p.head
      m.get(key) match {
        case Some((i, s)) =>
          s.delete(p.tail) match {
            case (null, null) =>
              // remove this path
              val newStruct = StructSignature((m - key).mapValues { case (index, sig) =>
                if (index > i)
                  (index - 1, sig)
                else
                  (index, sig)
              })
              if (newStruct.size == 0)
                (null, null)
              else {
                val f: Deleter = a =>
                  if (a == null)
                    a
                  else
                    a.asInstanceOf[Row].delete(i)
                (newStruct, f)
              }
            case (sig, deleter) =>
              val newStruct = StructSignature(m + ((key, (i, sig))))
              val f: Deleter = a =>
                if (a == null)
                  a
                else {
                  val r = a.asInstanceOf[Row]
                  r.update(i, deleter(r.get(i)))
                }
              (newStruct, f)
          }
        case None => throw new AnnotationPathException()
      }
    }
  }

  override def insert(p: List[String], signature: Signature): (StructSignature, Inserter) = {
    val key = p.head
    if (p.length == 1) {
      m.get(key) match {
        case Some((i, s)) =>
          val f: Inserter = (a, toIns) => a.asInstanceOf[Row].update(i, toIns.orNull)
          val newStruct = StructSignature(m + ((key, (i, signature))))
          (newStruct, f)
        case None =>
          // append, not overwrite
          val f: Inserter = (a, toIns) => a.asInstanceOf[Row].append(toIns.orNull)
          val newStruct = StructSignature(m + ((key, (m.size, signature))))
          (newStruct, f)
      }
    }
    else {
      m.get(key) match {
        case Some((i, s)) =>
          val (sig, ins) = s.insert(p.tail, signature)
          val f: Inserter = (a, toIns) =>
            if (a == null)
              Row.fromSeq(Array.fill[Any](m.size)(null))
                .update(i, ins(null: Any, toIns))
            else {
              val r = a.asInstanceOf[Row]
              r.update(i, ins(r.get(i), toIns))
            }
          val newStruct = StructSignature(m + ((key, (i, sig))))
          (newStruct, f)
        case None => // gotta put it in
          val (sig, ins) = {
            if (p.length > 1)
              StructSignature(Map.empty[String, (Int, Signature)])
                .insert(p.tail, signature)
            else
              signature.insert(p.tail, signature)
          }
          val f: Inserter = (a, toIns) => ins(null: Any, toIns)
          (StructSignature(m + ((key, (m.size, sig)))), f)
      }
    }
  }

  def getStruct(id: String): Option[StructSignature] = m.get(id).map(_.asInstanceOf[StructSignature])

  override def getSchema: DataType = {
    if (m.isEmpty)
    //FIXME placeholder?
      StringType
    else {
      val s =
        StructType(m
          .toArray
          .sortBy {
            case (key, (index, sig)) => index
          }
          .map {
            case (key, (index, sig)) =>
              StructField(key, sig.getSchema, true)
          }
        )
      assert(s.length > 0)
      s
    }
  }
  override def printSchema(path: String): String = {
    s"""$path: group of ${m.size} annotations\n""" +
    m.toArray
      .sortBy { case (k, (i, v)) => i}
      .map { case (k, (i, v)) => v.printSchema(path + s".$k")}
        .mkString("\n")
  }
}

object StructSignature {

  def empty(): StructSignature = StructSignature(Map.empty[String, (Int, Signature)])

  def emptyRowOfNull(i: Int): Row = Row.fromSeq(Array.fill[Any](i)(null))
}

case class SimpleSignature(dType: expr.Type) extends Signature