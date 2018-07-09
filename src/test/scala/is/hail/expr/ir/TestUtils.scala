package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.utils.{FastIndexedSeq, FastSeq}
import is.hail.variant.Call

object TestUtils {
  def toIRInt(i: Integer): IR =
    if (i == null)
      NA(TInt32())
    else
      I32(i)

  def toIRDouble(d: java.lang.Double): IR =
    if (d == null)
      NA(TFloat64())
    else
      F64(d)

  def toIRPair(p: (Integer, Integer)): IR =
    if (p == null)
      NA(TTuple(TInt32(), TInt32()))
    else
      MakeTuple(Seq(toIRInt(p._1), toIRInt(p._2)))

  def toIRArray(a: Seq[Integer]): IR =
    if (a == null)
      NA(TArray(TInt32()))
    else
      MakeArray(a.map(toIRInt), TArray(TInt32()))

  def IRArray(a: Integer*): IR = toIRArray(a)

  def toIRStringArray(a: Seq[String]): IR =
    if (a == null)
      NA(TArray(TString()))
    else
      MakeArray(a.map(s => Literal(s, TString())), TArray(TString()))

  def IRStringArray(a: String*): IR = toIRStringArray(a)

  def IRStringSet(a: String*): IR = ToSet(toIRStringArray(a))

  def toIRDoubleArray(a: Seq[java.lang.Double]): IR =
    if (a == null)
      NA(TArray(TFloat64()))
    else
      MakeArray(a.map(toIRDouble), TArray(TFloat64()))

  def IRDoubleArray(a: java.lang.Double*): IR = toIRDoubleArray(a)

  def toIRPairArray(a: Seq[(Integer, Integer)]): IR =
    if (a == null)
      NA(TArray(TTuple(TInt32(), TInt32())))
    else
      MakeArray(a.map(toIRPair), TArray(TTuple(TInt32(), TInt32())))

  def toIRDict(a: Seq[(Integer, Integer)]): IR =
    if (a == null)
      NA(TDict(TInt32(), TInt32()))
    else
      ToDict(MakeArray(a.map(toIRPair), TArray(TTuple(TInt32(), TInt32()))))

  def IRDict(a: (Integer, Integer)*): IR = toIRDict(a)

  def toIRSet(a: Seq[Integer]): IR =
    if (a == null)
      NA(TSet(TInt32()))
  else
      ToSet(toIRArray(a))

  def IRSet(a: Integer*): IR = toIRSet(a)

  def IRCall(c: Call): IR = Cast(I32(c), TCall())

    def IRAggCount: IR = {
    val aggSig = AggSignature(Count(), FastSeq.empty, None, FastSeq.empty)
    ApplyAggOp(SeqOp(0, FastIndexedSeq.empty, aggSig), FastIndexedSeq.empty, None, aggSig)
  }

  def IRScanCount: IR = {
    val aggSig = AggSignature(Count(), FastSeq.empty, None, FastSeq.empty)
    ApplyScanOp(SeqOp(0, FastIndexedSeq.empty, aggSig), FastIndexedSeq.empty, None, aggSig)
  }

  def IRAggCollect(ir: IR): IR = {
    val aggSig = AggSignature(Collect(), FastSeq.empty, None, FastSeq[Type](ir.typ))
    ApplyAggOp(
      SeqOp(0, FastIndexedSeq(ir), aggSig),
      FastIndexedSeq(), None, aggSig)
  }

  def IRScanCollect(ir: IR): IR = {
    val aggSig = AggSignature(Collect(), FastSeq.empty, None, FastSeq[Type](ir.typ))
    ApplyScanOp(
      SeqOp(0, FastIndexedSeq(ir), aggSig),
      FastIndexedSeq(), None, aggSig)
  }
}
