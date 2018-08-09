package is.hail.rvd

import is.hail.annotations._
import is.hail.expr.Parser
import is.hail.expr.types._
import is.hail.utils._
import org.apache.commons.lang3.builder.HashCodeBuilder
import org.json4s.CustomSerializer
import org.json4s.JsonAST.{JArray, JObject, JString, JValue}

class OrderedRVDTypeSerializer extends CustomSerializer[OrderedRVDType](format => ( {
  case JString(s) => Parser.parseOrderedRVDType(s)
}, {
  case orvdType: OrderedRVDType => JString(orvdType.toString)
}))

final case class OrderedRVDType(key: IndexedSeq[String], rowType: TStruct)
  extends Serializable {

  val keySet: Set[String] = key.toSet

  val (kType, _) = rowType.select(key)
  val (valueType, _) = rowType.filter(f => !keySet.contains(f.name))

  val kFieldIdx: Array[Int] = key.map(n => rowType.fieldIdx(n)).toArray
  val valueFieldIdx: Array[Int] = (0 until rowType.size)
    .filter(i => !keySet.contains(rowType.fields(i).name))
    .toArray

  val kOrd: UnsafeOrdering = kType.unsafeOrdering(missingGreatest = true)
  val kInRowOrd: UnsafeOrdering =
    OrderedRVDType.selectUnsafeOrdering(rowType, kFieldIdx, rowType, kFieldIdx)
  val kRowOrd: UnsafeOrdering =
    OrderedRVDType.selectUnsafeOrdering(kType, Array.range(0, kType.size), rowType, kFieldIdx)

  def kComp(other: OrderedRVDType): UnsafeOrdering =
    OrderedRVDType.selectUnsafeOrdering(
      this.rowType,
      this.kFieldIdx,
      other.rowType,
      other.kFieldIdx)

  def kRowOrdView(region: Region) = new OrderingView[RegionValue] {
    val wrv = WritableRegionValue(kType, region)
    def setFiniteValue(representative: RegionValue) {
      wrv.setSelect(rowType, kFieldIdx, representative)
    }
    def compareFinite(rv: RegionValue): Int =
      kRowOrd.compare(wrv.value, rv)
  }

  def insert(typeToInsert: Type, path: List[String]): (OrderedRVDType, UnsafeInserter) = {
    assert(path.nonEmpty)
    assert(!key.contains(path.head))

    val (newRowType, inserter) = rowType.unsafeInsert(typeToInsert, path)

    (OrderedRVDType(key, newRowType.asInstanceOf[TStruct]), inserter)
  }

  def toJSON: JValue =
    JObject(List(
      "key" -> JArray(key.map(JString).toList),
      "rowType" -> JString(rowType.parsableString())))

  override def toString: String = {
    val sb = new StringBuilder()
    sb.append("OrderedRVDType{key:[[")
    if (key.nonEmpty) {
      key.foreachBetween(k => sb.append(prettyIdentifier(k)))(sb += ',')
    }
    sb.append("]],row:")
    sb.append(rowType.parsableString())
    sb += '}'
    sb.result()
  }
}

object OrderedRVDType {

  def selectUnsafeOrdering(t1: TStruct, fields1: Array[Int],
    t2: TStruct, fields2: Array[Int]): UnsafeOrdering = {
    require(fields1.length == fields2.length)
    require((fields1, fields2).zipped.forall { case (f1, f2) =>
      t1.types(f1) isOfType t2.types(f2)
    })

    val nFields = fields1.length
    val fieldOrderings = Range(0, nFields).map { i =>
      t1.types(fields1(i)).unsafeOrdering(t2.types(fields2(i)), missingGreatest = true)
    }.toArray

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
}
