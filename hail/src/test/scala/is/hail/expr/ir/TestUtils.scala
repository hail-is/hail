package is.hail.expr.ir

import is.hail.types._
import is.hail.types.virtual._
import is.hail.utils.{FastIndexedSeq, FastSeq}
import is.hail.variant.Call

object TestUtils {
  def rangeKT: TableIR = TableKeyBy(TableRange(20, 4), FastIndexedSeq())

  def collect(tir: TableIR): IR =
    TableAggregate(tir, MakeStruct(FastSeq(
      "rows" -> IRAggCollect(Ref("row", tir.typ.rowType)),
      "global" -> Ref("global", tir.typ.globalType))))
  def collectNoKey(tir: TableIR): IR = TableCollect(tir)

  def toIRInt(i: Integer): IR =
    if (i == null)
      NA(TInt32)
    else
      I32(i)

  def toIRDouble(d: java.lang.Double): IR =
    if (d == null)
      NA(TFloat64)
    else
      F64(d)

  def toIRPair(p: (Integer, Integer)): IR =
    if (p == null)
      NA(TTuple(TInt32, TInt32))
    else
      MakeTuple.ordered(Seq(toIRInt(p._1), toIRInt(p._2)))

  def toIRArray(a: Seq[Integer]): IR =
    if (a == null)
      NA(TArray(TInt32))
    else
      MakeArray(a.map(toIRInt), TArray(TInt32))

  def IRArray(a: Integer*): IR = toIRArray(a)

  def toIRStream(a: Seq[Integer]): IR =
    if (a == null)
      NA(TStream(TInt32))
    else
      MakeStream(a.map(toIRInt), TStream(TInt32))

  def IRStream(a: Integer*): IR = toIRStream(a)

  def toIRStringArray(a: Seq[String]): IR =
    if (a == null)
      NA(TArray(TString))
    else
      MakeArray(a.map(s => Literal.coerce(TString, s)), TArray(TString))

  def IRStringArray(a: String*): IR = toIRStringArray(a)

  def IRStringSet(a: String*): IR = ToSet(ToStream(toIRStringArray(a)))

  def toIRDoubleArray(a: Seq[java.lang.Double]): IR =
    if (a == null)
      NA(TArray(TFloat64))
    else
      MakeArray(a.map(toIRDouble), TArray(TFloat64))

  def IRDoubleArray(a: java.lang.Double*): IR = toIRDoubleArray(a)

  def toIRPairArray(a: Seq[(Integer, Integer)]): IR =
    if (a == null)
      NA(TArray(TTuple(TInt32, TInt32)))
    else
      MakeArray(a.map(toIRPair), TArray(TTuple(TInt32, TInt32)))

  def toIRDict(a: Seq[(Integer, Integer)]): IR =
    if (a == null)
      NA(TDict(TInt32, TInt32))
    else
      ToDict(ToStream(MakeArray(a.map(toIRPair), TArray(TTuple(TInt32, TInt32)))))

  def IRDict(a: (Integer, Integer)*): IR = toIRDict(a)

  def toIRSet(a: Seq[Integer]): IR =
    if (a == null)
      NA(TSet(TInt32))
  else
      ToSet(ToStream(toIRArray(a)))

  def IRSet(a: Integer*): IR = toIRSet(a)

  def IRCall(c: Call): IR = invoke("callFromRepr", TCall, I32(c))

    def IRAggCount: IR = {
    val aggSig = AggSignature(Count(), FastSeq.empty, FastSeq.empty)
    ApplyAggOp(FastIndexedSeq.empty, FastIndexedSeq.empty, aggSig)
  }

  def IRScanCount: IR = {
    val aggSig = AggSignature(Count(), FastSeq.empty, FastSeq.empty)
    ApplyScanOp(FastIndexedSeq.empty, FastIndexedSeq.empty, aggSig)
  }

  def IRAggCollect(ir: IR): IR = {
    val aggSig = AggSignature(Collect(), FastSeq.empty, FastSeq[Type](ir.typ))
    ApplyAggOp(FastIndexedSeq(), FastIndexedSeq(ir), aggSig)
  }

  def IRScanCollect(ir: IR): IR = {
    val aggSig = AggSignature(Collect(), FastSeq.empty, FastSeq[Type](ir.typ))
    ApplyScanOp(FastIndexedSeq(), FastIndexedSeq(ir), aggSig)
  }

  def IRStruct(fields: (String, IR)*): IR =
    MakeStruct(fields)
}
