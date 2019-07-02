package is.hail.expr.ir.agg

import is.hail.expr.ir._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.CodecSpec
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
  def getAgg(aggSig: AggSignature): StagedRegionValueAggregator = aggSig match {
    case AggSignature(Sum(), _, _, Seq(t)) =>
      new SumAggregator(t.physicalType)
    case AggSignature(Count(), _, _, Seq(t)) =>
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
