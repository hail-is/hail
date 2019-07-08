package is.hail.expr.ir.agg

import is.hail.HailContext
import is.hail.annotations.{Region, RegionValue, RegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir
import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.CodecSpec
import is.hail.rvd.{RVDContext, RVDType}
import is.hail.utils._

import scala.language.{existentials, postfixOps}


class UnsupportedExtraction(msg: String) extends Exception(msg)

case class Aggs(postAggIR: IR, init: IR, seqPerElt: IR, aggs: Array[AggSignature]) {
  val typ: PTuple = PTuple(aggs.map(Extract.getPType))
  val nAggs: Int = aggs.length

  def readSet(i: Int, path: IR, spec: CodecSpec): IR =
    ReadAggs(i * nAggs, path, spec, aggs)

  def writeSet(i: Int, path: IR, spec: CodecSpec): IR =
    WriteAggs(i * nAggs, path, spec, aggs)

  def eltOp: IR = seqPerElt

  def results: IR = ResultOp2(0, aggs)
}

object Extract {

  def compatible(sig1: AggSignature, sig2: AggSignature): Boolean = (sig1.op, sig2.op) match {
    case (AggElements2(nestedAggs1), AggElements2(nestedAggs2)) =>
      nestedAggs1.zip(nestedAggs2).forall { case (a1, a2) => compatible(a1, a2) }
    case (AggElementsLengthCheck2(nestedAggs1, _), AggElements2(nestedAggs2)) =>
      nestedAggs1.zip(nestedAggs2).forall { case (a1, a2) => compatible(a1, a2) }
    case (AggElements2(nestedAggs1), AggElementsLengthCheck2(nestedAggs2, _)) =>
      nestedAggs1.zip(nestedAggs2).forall { case (a1, a2) => compatible(a1, a2) }
    case (AggElementsLengthCheck2(nestedAggs1, _), AggElementsLengthCheck2(nestedAggs2, _)) =>
      nestedAggs1.zip(nestedAggs2).forall { case (a1, a2) => compatible(a1, a2) }
    case _ => sig1 == sig2
  }

  def getAgg(aggSig: AggSignature): StagedRegionValueAggregator = aggSig match {
    case AggSignature(Sum(), _, _, Seq(t)) =>
      new SumAggregator(t.physicalType)
    case AggSignature(Count(), _, _, _) =>
      CountAggregator
    case AggSignature(AggElementsLengthCheck2(nestedAggs, knownLength), _, _, _) =>
      new ArrayElementLengthCheckAggregator(nestedAggs.map(getAgg).toArray, knownLength)
    case AggSignature(AggElements2(nestedAggs), _, _, _) =>
      new ArrayElementwiseOpAggregator(nestedAggs.map(getAgg).toArray)
    case AggSignature(PrevNonnull(), _, _, Seq(t)) =>
      new PrevNonNullAggregator(t.physicalType)
    case _ => throw new UnsupportedExtraction(aggSig.toString)
  }

  def getPType(aggSig: AggSignature): PType = getAgg(aggSig).resultType

  def getType(aggSig: AggSignature): Type = getPType(aggSig).virtualType
}