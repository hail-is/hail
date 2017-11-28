package is.hail.annotations.aggregators

import is.hail.asm4s._
import is.hail.expr._
import is.hail.expr.ir._
import is.hail.annotations._

object TransformedRegionValueAggregator {
  /**
    * {@code tAggIn} is aggregable type to which
    * TransformedRegionValueAggregator.seqOp will be applied
    *
    * The argument to {@code makeTransform} is an IR that evalutes to the
    * aggregation carrier struct. The output is an IR that evaluates to the
    * desired input to {@code next.seqOp}.
    *
    **/
  def apply(tAggIn: TAggregable,
    makeTransform: IR => IR,
    next: RegionValueAggregator): TransformedRegionValueAggregator = {

    val transform = makeTransform(In(0, tAggIn.carrierStruct))
    // this struct has Hail type
    // TransformedRegionValueAggregator.missingnessCarrier
    val out = MakeStruct(Array(("it", transform.typ, transform)))
    val fb = FunctionBuilder.functionBuilder[MemoryBuffer, Long, Boolean, Long]
    Compile(out, fb)
    new TransformedRegionValueAggregator(fb.result(), transform.typ, next)
  }
}

/**
  * {@code getTransformer} is a thunk of a function that transforms an
  * aggregation carrier struct to the desired input to {@code next.seqOp}.
  *
  **/
class TransformedRegionValueAggregator(
  getTransformer: () => AsmFunction3[MemoryBuffer, Long, Boolean, Long],
  t: Type,
  val next: RegionValueAggregator) extends RegionValueAggregator {

  val typ = next.typ

  private val missingnessCarrier = TStruct(true, "x" -> t)

  private val elementIndex = missingnessCarrier.fieldIdx("x")

  private def extractElement(region: MemoryBuffer, offset: Long): Long =
    missingnessCarrier.loadField(region, offset, elementIndex)

  private def isElementMissing(region: MemoryBuffer, offset: Long): Boolean =
    !missingnessCarrier.isFieldDefined(region, offset, elementIndex)

  def seqOp(region: MemoryBuffer, offset: Long, missing: Boolean) {
    val outOffset = getTransformer()(region, offset, missing)
    next.seqOp(region, extractElement(region, outOffset), isElementMissing(region, outOffset))
  }

  def combOp(agg2: RegionValueAggregator) {
    next.combOp(agg2.asInstanceOf[TransformedRegionValueAggregator].next)
  }

  def result(region: MemoryBuffer): Long = {
    next.result(region)
  }

  def copy(): TransformedRegionValueAggregator =
    new TransformedRegionValueAggregator(getTransformer, t, next)
}

