package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.expr.ir.defs.{
  GetField, GetTupleElement, If, MakeStruct, MakeTuple, StreamRange, ToArray, ToStream,
}
import is.hail.utils._

import org.apache.spark.sql.Row
import org.testng.annotations.Test

class ArrayDeforestationSuite extends HailSuite {
  implicit val execStrats: ExecStrategy.ValueSet = ExecStrategy.values

  def primitiveArrayNoRegion(len: IR): IR =
    ToArray(mapIR(StreamRange(0, len, 1))(_ + 5))

  def arrayWithRegion(len: IR): IR =
    ToArray(mapIR(StreamRange(0, len, 1)) { x2 =>
      MakeStruct(FastSeq("f1" -> (x2 + 1), "f2" -> 0))
    })

  def primitiveArrayWithRegion(len: IR): IR = {
    val array = arrayWithRegion(len)
    ToArray(mapIR(ToStream(array))(GetField(_, "f1")))
  }

  def arrayFoldWithStructWithPrimitiveValues(len: IR, max1: Int, max2: Int): IR = {
    val zero = MakeStruct(FastSeq[(String, IR)]("max1" -> max1, "max2" -> max2))
    foldIR(ToStream(primitiveArrayWithRegion(len)), zero) { (accum, value) =>
      If(
        value > GetField(accum, "max1"),
        MakeStruct(FastSeq("max1" -> value, "max2" -> GetField(accum, "max1"))),
        If(
          value > GetField(accum, "max2"),
          MakeStruct(FastSeq("max1" -> GetField(accum, "max1"), "max2" -> value)),
          accum,
        ),
      )
    }
  }

  def arrayFoldWithStruct(len: IR, v1: Int, v2: Int): IR = {
    val zero = MakeTuple.ordered(FastSeq(
      MakeStruct(FastSeq[(String, IR)]("f1" -> v1, "f2" -> v2)),
      MakeStruct(FastSeq[(String, IR)]("f1" -> v1, "f2" -> v2)),
    ))
    val array = arrayWithRegion(len)
    foldIR(ToStream(array), zero) { (accum, value) =>
      MakeTuple.ordered(FastSeq(GetTupleElement(accum, 1), value))
    }
  }

  @Test def testArrayFold(): Unit = {
    assertEvalsTo(arrayFoldWithStructWithPrimitiveValues(5, -5, -6), Row(5, 4))
    assertEvalsTo(arrayFoldWithStruct(5, -5, -6), Row(Row(4, 0), Row(5, 0)))
  }

}
