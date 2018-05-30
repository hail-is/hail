package is.hail.expr.ir

import is.hail.expr.types._

object TestUtils {
  def toIRInt(i: Integer): IR =
    if (i == null)
      NA(TInt32())
    else
      I32(i)

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
}
