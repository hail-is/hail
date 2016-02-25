package org.broadinstitute.hail.annotations

import org.apache.spark.sql.Row
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.Utils._
import scala.collection.mutable

class CollisionException(s: String) extends Exception(s)

case class MutableSignatures(sigs: mutable.Map[String, AnnotationSignature], index: Int) extends AnnotationSignature {
  def typeOf = "Mutable Signatures"

  def remapIndex(newIndex: Int) = copy(index = newIndex)

  def insertSig(toInsert: AnnotationSignature, path: Iterable[String]): Array[Int] = {
    val (map, indexPath) = MutableSignatures.descendNestedMaps(sigs, path.take(path.size - 1))
    map.get(path.last) match {
      case Some(remap) => map += ((path.last, toInsert.remapIndex(remap.index)))
      case None => map += ((path.last, toInsert.remapIndex(map.size)))
    }
    indexPath
  }

  def removeSig(path: Iterable[String]): Array[Int] = {
    val (map, indexPath) = try {
      MutableSignatures.descendNestedMaps(sigs, path.take(path.size - 1))
    } catch {
      case e: CollisionException => fatal(e.getMessage)
    }

    val lastIndex = map.get(path.last) match {
      case Some(sig) =>
        val ind = sig.index
        map.remove(path.last)
        ind
      case None => fatal("tried to remove signature that did not exist")
    }
    map.foreach { case (key, sig) =>
      if (sig.index > lastIndex) {
        map(key) = sig.remapIndex(sig.index - 1)
      }
      else
        sig
    }

    indexPath ++ Array(lastIndex)
  }

  def result(): AnnotationSignatures = {
    AnnotationSignatures(
      sigs.mapValues {
        case ms: MutableSignatures => ms.result()
        case x => x
      }
        .toMap, index)
  }
}

object MutableSignatures {
  def apply(sigs: AnnotationSignatures): MutableSignatures = {
    val m = mutable.Map.empty[String, AnnotationSignature]
    sigs.attrs.foreach {
      pair => m += pair
    }

    MutableSignatures(m, sigs.index)
  }


  def descendNestedMaps(m: mutable.Map[String, AnnotationSignature],
    path: Iterable[String]): (mutable.Map[String, AnnotationSignature], Array[Int]) = {

    val indexPath = mutable.ArrayBuilder.make[Int]
    val map = path.foldLeft(m) { (mMap, key) =>
      if (!mMap.contains(key)) {
        val index = mMap.size
        mMap += ((key, MutableSignatures(mutable.Map.empty[String, AnnotationSignature], index)))
      }

      mMap(key) match {
        case as: AnnotationSignatures =>
          val convertedSigs = MutableSignatures(as)
          mMap += ((key, convertedSigs))
          indexPath += convertedSigs.index
          convertedSigs.sigs
        case ms: MutableSignatures =>
          indexPath += ms.index
          ms.sigs
        case error =>
          throw new CollisionException(s"expected signatures or map in '$key', got ${error.getClass.getName}")
      }
    }
    (map, indexPath.result())
  }
}

case class MutableRow(arr: mutable.ArrayBuffer[Any]) {
  def applyOp(other: Row, op: Op) {
    val toAdd = op.path2.foldLeft(other) { (r, index) => r.getAs[Row](index) }
      .get(op.path2.last)
    op match {
      case a: AddOp =>
        val array = MutableRow.descendNestedRows(arr, op.path1)
        array += toAdd
      case r: RemapOp =>
        val array = MutableRow.descendNestedRows(arr, op.path1)
        array(r.path2.last) = toAdd
    }
  }

  def remove(path: Iterable[Int]) {
    val array = MutableRow.descendNestedRows(arr, path.take(path.size - 1))
    array.remove(path.last)
  }

  def insert(path: Iterable[Int], toAdd: Any) {
    val array = MutableRow.descendNestedRows(arr, path.take(path.size - 1))
    arr(path.last) = toAdd
  }

  def result(): Row = {
    println("doing result")
    println(this)


    Row.fromSeq(arr.toSeq.map {
      case mr: MutableRow => mr.result()
      case x => x
    })
  }
}

object MutableRow {
  def apply(row: Row): MutableRow = {
    val buffer = mutable.ArrayBuffer.empty[Any]
    buffer ++= row.toSeq
    MutableRow(buffer)
  }

  def descendNestedRows(arr: mutable.ArrayBuffer[Any], path: Iterable[Int]): mutable.ArrayBuffer[Any] = {
    path.foldLeft(arr) { (a, index) =>
      if (a.length < index) {
        a += MutableRow(mutable.ArrayBuffer.empty[Any])
      }
      a(index) match {
        case row: Row =>
          val convertedRow = MutableRow(row)
          a(index) = convertedRow
          convertedRow.arr
        case mr: MutableRow => mr.arr
        case error => throw new UnsupportedOperationException(s"expected row or arr, got ${error.getClass.getName}")
      }
    }
  }
}

sealed trait Op {
  def path1: Iterable[Int]

  def path2: Iterable[Int]
}

case class RemapOp(path1: Iterable[Int], path2: Iterable[Int]) extends Op

case class AddOp(path1: Iterable[Int], path2: Iterable[Int]) extends Op

case class AnnotationData(row: Row) extends Serializable {
  def get[T](i: Iterable[Int]): T = {
    i.take(i.size - 1).foldLeft(row)((row, index) => row.getAs[Row](index))
      .getAs[T](i.last)
  }

  def getOption[T](i: Iterable[Int]): Option[T] = {
    val parent = i.take(i.size - 1).foldLeft(row) { (row, index) =>
      if (row != null) {
        row.get(index) match {
          case r: Row => r
          case _ => return None
        }
      }
      else
        row
      row.getAs[Row](index)
    }
    if (parent == null)
      None
    else
      parent.isNullAt(i.last) match {
        case true => None
        case false => Some(parent.getAs[T](i.last))
      }
  }

  def same(other: AnnotationData): Boolean = AnnotationData.rowSame(row, other.row)
}

object AnnotationData {
  def apply(data: Seq[Any]): AnnotationData = AnnotationData(makeRow(data))

  def apply(data: Seq[Any], depth: Int): AnnotationData =
    AnnotationData(makeRow(data, depth))

  def rowSame(row1: Row, row2: Row): Boolean = {
    row1.size == row2.size &&
      row1.toSeq.iterator.zip(row2.toSeq.iterator).forall {
        case (r1: Row, r2: Row) => rowSame(r1, r2)
        case (a1: Array[_], a2: Array[_]) => a1.sameElements(a2)
        case (elem1, elem2) => elem1 == elem2
      }
  }

  def empty(): AnnotationData = new AnnotationData(Row.empty)

  def emptyIndexedSeq(n: Int): IndexedSeq[AnnotationData] = IndexedSeq.fill[AnnotationData](n)(AnnotationData.empty())

  def printNestedIterable(r: Iterable[Any], nSpace: Int = 0) {
    val spaces = (0 until nSpace).map(i => " ").foldRight("")(_ + _)
    r.zipWithIndex.foreach { case (elem, index) =>
      elem match {
        case row: Row =>
          println(s"""$spaces[$index] ROW:""")
          printRow(row, nSpace + 2)
        case iter: Iterable[Any] =>
          println(s"""$spaces[$index] ITER:""")
          printNestedIterable(iter, nSpace + 2)
        case _ =>
          println(s"""$spaces[$index] $elem""")
      }
    }
  }

  def printRow(r: Row, nSpace: Int = 0) {
    val spaces = (0 until nSpace).map(i => " ").foldRight("")(_ + _)
    r.toSeq.zipWithIndex.foreach { case (elem, index) =>
      elem match {
        case row: Row =>
          println(s"""$spaces[$index] ROW:""")
          printRow(row, nSpace + 2)
        case _ =>
          println(s"""$spaces[$index] $elem""")
      }
    }
  }

  def printData(ad: AnnotationData) {
    printRow(ad.row)
  }

  def makeRow(values: Seq[Any]): Row = Row.fromSeq(values.toSeq)

  def makeRow(values: Seq[Any], depth: Int): Row =
    (0 until depth).foldLeft(Row.fromSeq(values)) { (row, i) => Row.fromSeq(Seq(row)) }

  def removeSignature(base: AnnotationSignatures,
    path: Array[String]): (AnnotationSignatures, (AnnotationData => AnnotationData)) = {
    val ms = MutableSignatures(base)
    val indexPath = ms.removeSig(path)

    val f: AnnotationData => AnnotationData =
      (ad) => {
        val mr = MutableRow(ad.row)
        mr.remove(indexPath)
        AnnotationData(mr.result())
      }

    (ms.result(), f)
  }

  def insertSignature(base: AnnotationSignatures, toAdd: AnnotationSignature,
    path: Array[String]): (AnnotationSignatures, (AnnotationData, Any) => AnnotationData) = {

    val ms = MutableSignatures(base)
    val insertPath = ms.insertSig(toAdd, path)
    val newSigs = ms.result()

    val f: (AnnotationData, Any) => AnnotationData = {
      (ad, t) =>
        val mr = MutableRow(ad.row)
        mr.insert(insertPath, t)
        AnnotationData(mr.result())
    }

    (newSigs, f)
  }

  def splitMulti(base: AnnotationSignatures): (AnnotationData, Int) => AnnotationData = {
    base.getOption[AnnotationSignature]("info") match {
      case Some(infoSigs: AnnotationSignatures) =>
        val functions: Array[((Any, Int) => Any)] = infoSigs.attrs
          .toArray
          .sortBy { case (key, value) => value.index }
          .map { case (key, value) => value match {
            case vcfSig: VCFSignature if vcfSig.number == "A" =>
              vcfSig.typeOf match {
                case "Array[Int]" => (a: Any, ind: Int) =>
                  Array(a.asInstanceOf[Array[Int]]
                    .apply(ind))
                case _ => (a: Any, ind: Int) =>
                  Array(a.asInstanceOf[Array[Double]]
                    .apply(ind))
              }
            case vcfSig: VCFSignature if vcfSig.number == "R" =>
              vcfSig.typeOf match {
                case "Array[Int]" => (a: Any, ind: Int) =>
                  val arr = a.asInstanceOf[Array[Int]]
                  Array(arr(0), arr(ind))
                case _ => (a: Any, ind: Int) =>
                  val arr = a.asInstanceOf[Array[Double]]
                  Array(arr(0), arr(ind))
              }
            case vcfSig: VCFSignature if vcfSig.number == "G" =>
              vcfSig.typeOf match {
                case "Array[Int]" => (a: Any, ind: Int) =>
                  val arr = a.asInstanceOf[Array[Int]]
                  Array(arr(0),
                    triangle(ind + 1) + 1,
                    triangle(ind + 2) - 1)
                case _ => (a: Any, ind: Int) =>
                  val arr = a.asInstanceOf[Array[Double]]
                  Array(arr(0),
                    triangle(ind + 1) + 1,
                    triangle(ind + 2) - 1)
              }
            case _ => (a: Any, ind: Int) => a
          }
          }
        (ad: AnnotationData, index: Int) =>
          val adArr = ad.row.toSeq.toArray
          val infoR = adArr(infoSigs.index).asInstanceOf[Row]
          adArr(infoSigs.index) = Row.fromSeq(
            functions.zipWithIndex
              .map { case (f, i) =>
                f(infoR.get(i), i)
              })
          AnnotationData(Row.fromSeq(adArr))
      case _ => (ad, index) => ad
    }
  }
}

case class AnnotationSignatures(attrs: Map[String, AnnotationSignature],
  index: Int = -1) extends AnnotationSignature {
  def typeOf: String = "Signatures"

  def query(query: Iterable[String]): Array[Int] = {
    val (path, sigs) = query.take(query.size - 1).foldLeft((mutable.ArrayBuilder.make[Int], this)) { case ((arr, sig), key) =>
      val next = sig.get[AnnotationSignatures](key)
      arr += next.index
      (arr, next)
    }
    path += sigs
      .get[AnnotationSignature](query.last)
      .index

    path.result()
  }

  def query(query: String): Array[Int] = {
    Array(attrs(query).index)
  }

  def contains(elem: String): Boolean = attrs.contains(elem)

  def contains(path: Iterable[String]): Boolean = {
    path
      .take(path.size - 1)
      .foldLeft((attrs, true)) { case ((map, continue), key) =>
        if (!continue)
          (map, continue)
        else
          map.get(key) match {
            case Some(sigs: AnnotationSignatures) => (sigs.attrs, true)
            case _ => (attrs, false)
          }
      }
      ._1
      .contains(path.last)
  }

  def get[T](key: String): T = attrs(key).asInstanceOf[T]

  def getOption[T](key: String): Option[T] = attrs.get(key).map(_.asInstanceOf[T])

  def remapIndex(newIndex: Int): AnnotationSignature = this.copy(index = newIndex)

  def nextIndex(): Int = attrs.size
}

object AnnotationSignatures {
  def empty(): AnnotationSignatures = AnnotationSignatures(Map.empty[String, AnnotationSignature])
}
