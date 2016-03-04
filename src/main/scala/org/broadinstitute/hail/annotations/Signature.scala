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

  def delete(path: List[String]): (Signature, Deleter) = {
    throw new AnnotationPathException()
    //    if (path.nonEmpty) {
    //    }
    //    else
    //      (this, )
  }

  def insert(path: List[String], signature: Signature): (Signature, Inserter) = {
    if (path.nonEmpty) {
      StructSignature(Map.empty[String, (Int, Signature)]).insert(path.tail, signature)
    }
    else
      (this, (a, toIns) => {
        assert(a == null)
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
}

case class StructSignature(m: Map[String, (Int, Signature)]) extends Signature {
  def dType: expr.Type = null

  def size: Int = m.size

  def contains(key: String): Boolean = m.contains(key)

  def contains(keys: List[String]): Boolean = {
    try {
      val q = query(keys)
      false
    }
    catch {
      case e: AnnotationPathException => true
    }
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
    val key = p.head
    if (p.length == 1) {
      m.get(key) match {
        case Some((i, s)) =>
          val f: Deleter = a => a.asInstanceOf[Row].delete(i)
          val newStruct = StructSignature((m - key).mapValues {
            case (index, sig) =>
              if (index > i)
                (index - 1, sig)
              else
                (index, sig)
          })
          (newStruct, f)
        case None => throw new AnnotationPathException()
      }
    }
    else {
      m.get(key) match {
        case Some((i, s)) =>
          val (sig, d) = s.delete(p.tail)
          val f: Deleter = a =>
            if (a == null)
              a
            else {
              val r = a.asInstanceOf[Row]
              r.update(i, d(r.get(i)))
            }
          val newStruct = StructSignature(m + ((key, (i, sig))))
          (newStruct, f)
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

//  def contains(elem: String): Boolean = m.contains(elem)
//
//  def contains(path: Iterable[String]): Boolean = {
//    path
//      .take(path.size - 1)
//      .foldLeft((m, true)) { case ((map, continue), key) =>
//        if (!continue)
//          (map, continue)
//        else
//          map.get(key) match {
//            case Some((i, sigs: StructSignature)) => (sigs.m, true)
//            case _ => (m, false)
//          }
//      }
//      ._1
//      .contains(path.last)
//  }
//
//  def get[T](key: String): T = m(key).asInstanceOf[T]
//
//  def getOption[T](key: String): Option[T] = m.get(key).map(_.asInstanceOf[T])
//
//  def nextIndex(): Int = m.size

  def getSchema: StructType = {
    //    println(m)
    //    m.foreach(println)
    StructType(m
      .toArray
      .sortBy { case (key, (index, sig)) => index }
      .map { case (key, (index, sig)) => sig match {
        case sigs: StructSignature => StructField(key, sigs.getSchema, true)
        case sig => StructSignature.toSchemaType(key, sig)
      }
      }
    )
  }
}

object StructSignature {

  def empty(): StructSignature = StructSignature(Map.empty[String, (Int, Signature)])

  def emptyRowOfNull(i: Int): Row = Row.fromSeq(Array.fill[Any](i)(null))

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
        case _ => throw new UnsupportedOperationException()
      }
    }
  }
}

case class SimpleSignature(dType: expr.Type) extends Signature