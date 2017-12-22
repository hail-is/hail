package is.hail.rvd

import is.hail.annotations._
import is.hail.expr.Parser
import is.hail.expr.typ._
import is.hail.utils._
import org.apache.commons.lang3.builder.HashCodeBuilder
import org.json4s.JsonAST.{JArray, JObject, JString, JValue}

class OrderedRVType(
  val partitionKey: Array[String],
  val key: Array[String], // full key
  val rowType: TStruct) extends Serializable {
  assert(key.startsWith(partitionKey))

  val (pkType, _) = rowType.select(partitionKey)
  val (kType, _) = rowType.select(key)

  val keySet: Set[String] = key.toSet
  val (valueType, _) = rowType.filter(f => !keySet.contains(f.name))

  val valueFieldIdx: Array[Int] = (0 until rowType.size)
    .filter(i => !keySet.contains(rowType.fields(i).name))
    .toArray

  val kRowFieldIdx: Array[Int] = key.map(n => rowType.fieldIdx(n))
  val pkRowFieldIdx: Array[Int] = partitionKey.map(n => rowType.fieldIdx(n))
  val pkKFieldIdx: Array[Int] = partitionKey.map(n => kType.fieldIdx(n))
  assert(pkKFieldIdx sameElements (0 until pkType.size))

  val pkOrd: UnsafeOrdering = pkType.unsafeOrdering(missingGreatest = true)
  val kOrd: UnsafeOrdering = kType.unsafeOrdering(missingGreatest = true)

  val pkRowOrd: UnsafeOrdering = OrderedRVType.selectUnsafeOrdering(pkType, (0 until pkType.size).toArray, rowType, pkRowFieldIdx)
  val pkKOrd: UnsafeOrdering = OrderedRVType.selectUnsafeOrdering(pkType, (0 until pkType.size).toArray, kType, pkKFieldIdx)
  val pkInRowOrd: UnsafeOrdering = OrderedRVType.selectUnsafeOrdering(rowType, pkRowFieldIdx, rowType, pkRowFieldIdx)
  val kInRowOrd: UnsafeOrdering = OrderedRVType.selectUnsafeOrdering(rowType, kRowFieldIdx, rowType, kRowFieldIdx)
  val pkInKOrd: UnsafeOrdering = OrderedRVType.selectUnsafeOrdering(kType, pkKFieldIdx, kType, pkKFieldIdx)
  val kRowOrd: UnsafeOrdering = OrderedRVType.selectUnsafeOrdering(kType, (0 until kType.size).toArray, rowType, kRowFieldIdx)

  def insert(typeToInsert: Type, path: List[String]): (OrderedRVType, UnsafeInserter) = {
    assert(path.nonEmpty)
    assert(!key.contains(path.head))

    val (newRowType, inserter) = rowType.unsafeInsert(typeToInsert, path)

    (new OrderedRVType(partitionKey, key, newRowType.asInstanceOf[TStruct]), inserter)
  }

  def toJSON: JValue =
    JObject(List(
      "partitionKey" -> JArray(partitionKey.map(JString).toList),
      "key" -> JArray(key.map(JString).toList),
      "rowType" -> JString(rowType.toString)))

  override def equals(that: Any): Boolean = that match {
    case that: OrderedRVType =>
      (partitionKey sameElements that.partitionKey) &&
        (key sameElements that.key) &&
        rowType == that.rowType
    case _ => false
  }

  override def hashCode: Int = {
    val b = new HashCodeBuilder(43, 19)
    b.append(partitionKey.length)
    partitionKey.foreach(b.append)

    b.append(key.length)
    key.foreach(b.append)

    b.append(rowType)
    b.toHashCode
  }
}

object OrderedRVType {
  def selectUnsafeOrdering(t1: TStruct, fields1: Array[Int],
    t2: TStruct, fields2: Array[Int]): UnsafeOrdering = {
    require(fields1.length == fields2.length)
    require((fields1, fields2).zipped.forall { case (f1, f2) =>
      t1.fieldType(f1) == t2.fieldType(f2)
    })

    val nFields = fields1.length
    val fieldOrderings = fields1.map(f1 => t1.fieldType(f1).unsafeOrdering(missingGreatest = true))

    new UnsafeOrdering {
      def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
        var i = 0
        while (i < nFields) {
          val f1 = fields1(i)
          val f2 = fields2(i)
          val leftDefined = t1.isFieldDefined(r1, o1, f1)
          val rightDefined = t2.isFieldDefined(r2, o2, f2)

          if (leftDefined && rightDefined) {
            val c = fieldOrderings(i).compare(r1, t1.loadField(r1, o1, f1), r2, t2.loadField(r2, o2, f2))
            if (c != 0)
              return c
          } else if (leftDefined != rightDefined) {
            val c = if (leftDefined) -1 else 1
            return c
          }

          i += 1
        }

        0
      }
    }
  }

  def apply(jv: JValue): OrderedRVType = {
    case class Extract(partitionKey: Array[String],
      key: Array[String],
      rowType: String)
    val ex = jv.extract[Extract]
    new OrderedRVType(ex.partitionKey, ex.key, Parser.parseType(ex.rowType).asInstanceOf[TStruct])
  }
}
