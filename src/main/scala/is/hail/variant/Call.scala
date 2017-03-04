package is.hail.variant

import is.hail.check.Gen
import is.hail.expr.{TInt, TStruct, Type}
import is.hail.utils._
import is.hail.variant.GenotypeType.GenotypeType
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.json4s._

trait Call {
  def unboxedGT: Int

  def gt: Option[Int] =
    if (unboxedGT == -1)
      None
    else
      Some(unboxedGT)

  def isHomRef: Boolean = unboxedGT == 0

  def isHet: Boolean = unboxedGT > 0 && {
    val p = Genotype.gtPair(unboxedGT)
    p.j != p.k
  }

  def isHomVar: Boolean = unboxedGT > 0 && {
    val p = Genotype.gtPair(unboxedGT)
    p.j == p.k
  }

  def isCalledNonRef: Boolean = unboxedGT > 0

  def isHetNonRef: Boolean = unboxedGT > 0 && {
    val p = Genotype.gtPair(unboxedGT)
    p.j > 0 && p.j != p.k
  }

  def isHetRef: Boolean = unboxedGT > 0 && {
    val p = Genotype.gtPair(unboxedGT)
    p.j == 0 && p.k > 0
  }

  def isNotCalled: Boolean = unboxedGT == -1

  def isCalled: Boolean = unboxedGT >= 0

  def gtType: GenotypeType =
    if (isHomRef)
      GenotypeType.HomRef
    else if (isHet)
      GenotypeType.Het
    else if (isHomVar)
      GenotypeType.HomVar
    else {
      assert(isNotCalled)
      GenotypeType.NoCall
    }

  def hasNNonRefAlleles: Boolean = unboxedGT != -1

  def nNonRefAlleles_ : Int = Genotype.gtPair(unboxedGT).nNonRefAlleles

  def nNonRefAlleles: Option[Int] =
    if (hasNNonRefAlleles)
      Some(nNonRefAlleles_)
    else
      None

  def oneHotAlleles(nAlleles: Int): Option[IndexedSeq[Int]] = {
    gt.map { call =>
      val gtPair = Genotype.gtPair(call)
      val j = gtPair.j
      val k = gtPair.k
      new IndexedSeq[Int] {
        def length: Int = nAlleles

        def apply(idx: Int): Int = {
          if (idx < 0 || idx >= nAlleles)
            throw new ArrayIndexOutOfBoundsException(idx)
          var r = 0
          if (idx == j)
            r += 1
          if (idx == k)
            r += 1
          r
        }
      }
    }
  }

  def oneHotAlleles(v: Variant): Option[IndexedSeq[Int]] = oneHotAlleles(v.nAlleles)

  def oneHotGenotype(v: Variant): Option[IndexedSeq[Int]] = oneHotGenotype(v.nGenotypes)

  def oneHotGenotype(nGenotypes: Int): Option[IndexedSeq[Int]] = {
    gt.map { call =>
      new IndexedSeq[Int] {
        def length: Int = nGenotypes

        def apply(idx: Int): Int = {
          if (idx < 0 || idx >= nGenotypes)
            throw new ArrayIndexOutOfBoundsException(idx)
          if (idx == call)
            1
          else
            0
        }
      }
    }
  }
}



object Call {
  def apply(gtx: Int) = new GenericCall(gtx)

  def schema: DataType = StructType(Array(
    StructField("gt", IntegerType)))

  def t: Type = TStruct("gt" -> TInt)

  def genArb: Gen[Call] =
    for (v <- Variant.gen;
      nAlleles = v.nAlleles;
      nGenotypes = triangle(nAlleles);
      gtx <- Gen.choose(0, nGenotypes - 1)
    ) yield {
        val c = new GenericCall(gtx)
        c.check(nAlleles)
        c
      }
}

class GenericCall(gtx: Int) extends Serializable with Call {
  def unboxedGT = gtx

  def check(nAlleles: Int) {
    val nGenotypes = triangle(nAlleles)
    assert(gt.forall(i => i >= 0 && i < nGenotypes))
  }

  def toRow: Row = Row(gt.orNull)

  def toJSON: JValue = JObject(("gt", gt.map(JInt(_)).getOrElse(JNull)))
}
