package is.hail.rvd

import is.hail.annotations._
import is.hail.expr.Parser
import is.hail.expr.types._
import is.hail.utils._
import org.apache.commons.lang3.builder.HashCodeBuilder
import org.json4s.CustomSerializer
import org.json4s.JsonAST.{JArray, JObject, JString, JValue}

class RVDTypeSerializer extends CustomSerializer[RVDType](format => ( {
  case JString(s) => Parser.parseRVDType(s)
}, {
  case rvdType: RVDType => JString(rvdType.toString)
}))

final case class RVDType(rowType: TStruct, key: IndexedSeq[String] = FastIndexedSeq())
  extends Serializable {

  val keySet: Set[String] = key.toSet

  val (kType, _) = rowType.select(key)
  val (valueType, _) = rowType.filter(f => !keySet.contains(f.name))

  val kFieldIdx: Array[Int] = key.map(n => rowType.fieldIdx(n)).toArray
  val valueFieldIdx: Array[Int] = (0 until rowType.size)
    .filter(i => !keySet.contains(rowType.fields(i).name))
    .toArray

  val kOrd: UnsafeOrdering = kType.physicalType.unsafeOrdering(missingGreatest = true)
  val kInRowOrd: UnsafeOrdering =
    RVDType.selectUnsafeOrdering(rowType, kFieldIdx, rowType, kFieldIdx)
  val kRowOrd: UnsafeOrdering =
    RVDType.selectUnsafeOrdering(kType, Array.range(0, kType.size), rowType, kFieldIdx)

  def kComp(other: RVDType): UnsafeOrdering =
    RVDType.selectUnsafeOrdering(
      this.rowType,
      this.kFieldIdx,
      other.rowType,
      other.kFieldIdx,
      true)

  def joinComp(other: RVDType): UnsafeOrdering =
    RVDType.selectUnsafeOrdering(
      this.rowType,
      this.kFieldIdx,
      other.rowType,
      other.kFieldIdx,
      false)

  /** Comparison of a point with an interval, for use in joins where one side
    * is keyed by intervals.
    */
  def intervalJoinComp(other: RVDType): UnsafeOrdering = {
    require(other.key.length == 1)
    require(other.rowType.field(other.key(0)).typ.asInstanceOf[TInterval].pointType == rowType.field(key(0)).typ)

    new UnsafeOrdering {
      val t1 = rowType
      val t2 = other.rowType
      val t1p = t1.physicalType
      val t2p = t2.physicalType
      val f1 = kFieldIdx(0)
      val f2 = other.kFieldIdx(0)
      val intervalType = t2.types(f2).asInstanceOf[TInterval].physicalType
      val pord = t1p.types(f1).unsafeOrdering(intervalType.pointType)

      // Left is a point, right is an interval.
      // Returns -1 if point is below interval, 0 if it is inside, and 1 if it
      // is above, always considering missing greatest.
      def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {

        val leftDefined = rowType.isFieldDefined(r1, o1, f1)
        val rightDefined = other.rowType.isFieldDefined(r2, o2, f2)

        if (leftDefined && rightDefined) {
          val k1 = t1.loadField(r1, o1, f1)
          val k2 = t2.loadField(r2, o2, f2)
          if (intervalType.startDefined(r2, k2)) {
            val c = pord.compare(r1, k1, r2, intervalType.loadStart(r2, k2))
            if (c < 0 || (c == 0 && !intervalType.includesStart(r2, k2))) {
              -1
            } else {
              if (intervalType.endDefined(r2, k2)) {
                val c = pord.compare(r1, k1, r2, intervalType.loadEnd(r2, k2))
                if (c < 0 || (c == 0 && intervalType.includesEnd(r2, k2)))
                  0
                else 1
              } else 0
            }
          } else -1
        } else if (leftDefined != rightDefined) {
          if (leftDefined) -1 else 1
        } else 0
      }
    }
  }


  def kRowOrdView(region: Region) = new OrderingView[RegionValue] {
    val wrv = WritableRegionValue(kType.physicalType, region)
    def setFiniteValue(representative: RegionValue) {
      wrv.setSelect(rowType.physicalType, kFieldIdx, representative)
    }
    def compareFinite(rv: RegionValue): Int =
      kRowOrd.compare(wrv.value, rv)
  }

  def insert(typeToInsert: Type, path: List[String]): (RVDType, UnsafeInserter) = {
    assert(path.nonEmpty)
    assert(!key.contains(path.head))

    val (newRowPType, inserter) = rowType.physicalType.unsafeInsert(typeToInsert.physicalType, path)
    val newRowType = newRowPType.virtualType

    (RVDType(newRowType.asInstanceOf[TStruct], key), inserter)
  }

  def toJSON: JValue =
    JObject(List(
      "partitionKey" -> JArray(key.map(JString).toList),
      "key" -> JArray(key.map(JString).toList),
      "rowType" -> JString(rowType.parsableString())))

  override def toString: String = {
    val sb = new StringBuilder()
    sb.append("RVDType{key:[[")
    if (key.nonEmpty) {
      key.foreachBetween(k => sb.append(prettyIdentifier(k)))(sb += ',')
    }
    sb.append("]],row:")
    sb.append(rowType.parsableString())
    sb += '}'
    sb.result()
  }
}

object RVDType {
  def selectUnsafeOrdering(t1: TStruct, fields1: Array[Int],
    t2: TStruct, fields2: Array[Int], missingEqual: Boolean=true): UnsafeOrdering = {
    require(fields1.length == fields2.length)
    require((fields1, fields2).zipped.forall { case (f1, f2) =>
      t1.types(f1) isOfType t2.types(f2)
    })

    val t1p = t1.physicalType
    val t2p = t2.physicalType

    val nFields = fields1.length
    val fieldOrderings = Range(0, nFields).map { i =>
      t1p.types(fields1(i)).unsafeOrdering(t2p.types(fields2(i)), missingGreatest = true)
    }.toArray

    new UnsafeOrdering {
      def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
        var i = 0
        var hasMissing=false
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
          } else hasMissing = true

          i += 1
        }
        if (!missingEqual && hasMissing) -1 else 0
      }
    }
  }
}
