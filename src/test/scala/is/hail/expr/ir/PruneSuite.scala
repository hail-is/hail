package is.hail.expr.ir

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.expr.{TableFilter, TableImport, TableMapRows, TableParallelize, TableRange}
import is.hail.expr.types._
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class PruneSuite extends SparkSuite {
  @Test def testUnionType() {
    val base = TStruct(
      "a" -> TStruct(
        "aa" -> TInt32(),
        "ab" -> TStruct(
          "aaa" -> TString()
        )
      ),
      "b" -> TInt32(),
      "c" -> TArray(TStruct(
        "ca" -> TInt32()
      ))
    )

    assert(PruneDeadFields.unify(base, TStruct()) == TStruct())
    assert(PruneDeadFields.unify(base, TStruct("b" -> TInt32())) == TStruct("b" -> TInt32()))
    assert(PruneDeadFields.unify(base, TStruct("a" -> TStruct())) == TStruct("a" -> TStruct()))
    assert(PruneDeadFields.unify(base, TStruct("a" -> TStruct()),
      TStruct("b" -> TInt32())) == TStruct("a" -> TStruct(), "b" -> TInt32()))
    assert(PruneDeadFields.unify(base, TStruct("c" -> TArray(TStruct()))) == TStruct("c" -> TArray(TStruct())))
    assert(PruneDeadFields.unify(base, TStruct("a" -> TStruct("ab" -> TStruct())),
      TStruct("c" -> TArray(TStruct()))) == TStruct("a" -> TStruct("ab" -> TStruct()), "c" -> TArray(TStruct())))
  }

  @Test def testPipeline() {
    hc // initialize
    var ir = TableParallelize(TableType(TStruct(
      "a" -> TString(),
      "b" -> TString(),
      "c" -> TStruct("c1" -> TInt64(), "c2" -> TArray(TInt32())),
      "d" -> TArray(TStruct("d1" -> TInt32(), "d2" -> TFloat64()))),
      Some(Array("a")),
      TStruct()),
      IndexedSeq(
        Row("foo", "row1-b", Row(1L, IndexedSeq(1, 2, 3)), IndexedSeq(Row(1, 1.5), Row(2, 2.5))),
        Row("bar", "row2-b", Row(10L, IndexedSeq(-1, -2, -3)), IndexedSeq(Row(5, 5.5), Row(7, 7.5)))))

    val sig = AggSignature(Sum(), TInt64(), Seq(), None)
    var ir2 = TableFilter(ir, ApplyComparisonOp(LT(TInt32()), GetField(ArrayRef(GetField(Ref("row", ir.typ.rowType), "d"), I32(0)), "d1"), I32(3)))
    var aggir2 = TableAggregate(ir2, ApplyAggOp(SeqOp(GetField(GetField(Ref("row", ir2.typ.rowType), "c"), "c1"), I32(0), sig), FastIndexedSeq(), None, sig))

    val row = Ref("row", ir.typ.rowType)
    var ir3 = TableMapRows(ir, MakeStruct(Seq(
      "a" -> GetField(row, "a"),
      "b" -> GetField(row, "b"),
      "c" -> GetField(row, "c"),
      "d" -> GetField(row, "d"),
      "e" -> I32(10))), None)

    var aggir3 = TableAggregate(ir3, ApplyAggOp(SeqOp(GetField(GetField(Ref("row", ir3.typ.rowType), "c"), "c1"), I32(0), sig), FastIndexedSeq(), None, sig))
    assert(Interpret[Long](aggir3, optimize = true) == Interpret[Long](aggir3, optimize = false))
  }
}
