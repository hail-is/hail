package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.{FilterVariants, FilterSamples, FilterGenotypes, State}
import org.broadinstitute.hail.utils.TestRDDBuilder
import org.testng.annotations.Test

class FilterSuite extends SparkSuite {

  @Test def filterUtilsTest(): Unit = {
    import org.broadinstitute.hail.methods.FilterUtils._

    val foTrue = new FilterOption(Some(true))
    val foFalse = new FilterOption(Some(false))
    val foNone = new FilterOption(None)

    assert(true)
    assert(foTrue.ot.get)

    assert(foNone.ot.isEmpty)
    assert((foNone === foNone).ot.isEmpty)


    assert(true === true)
    //assert((true === foTrue).ot.get)
    assert((foTrue === true).ot.get)
    assert((foTrue === foTrue).ot.get)

    //assert((true === foNone).ot.isEmpty)
    assert((foNone === true).ot.isEmpty)
    //assert((foNone === foNone).ot.isEmpty)

    //assert((true && foTrue).ot.get)
    //assert((foTrue && true).ot.get)
    assert((foTrue && foTrue).ot.get)

    //assert((true && foNone).ot.isEmpty)
    //assert((foNone && true).ot.isEmpty)
    //assert((foTrue && foNone).ot.isEmpty)
    //assert((foNone && foTrue).ot.isEmpty)

    assert((!foFalse).ot.get)
    //assert((!foNone).ot.isEmpty)


    val foStr1 = new FilterOption(Some("1"))
    val foStr1p0 = new FilterOption(Some("1.0"))

    //assert(("1" === foNone).ot.isEmpty)
    assert((foNone === "1").ot.isEmpty)
    //assert((foStr1 === foNone).ot.isEmpty)
    assert((foNone === foStr1).ot.isEmpty)

    //assert((1 === foStr1.toInt).ot.get)
    assert((foStr1.toInt === 1).ot.get)

    //assert((1 === foStr1p0.toDouble).ot.get)
    assert((foStr1p0.toDouble === 1).ot.get)

    //assert((3 === foStr1p0.length).ot.get)
    assert((foStr1p0.length === 3).ot.get)

    //assert(('.' === foStr1p0(1)).ot.isEmpty)
    assert((foStr1p0(1) === '.').ot.get)

    //assert((foStr1p0(1) === foNone).ot.isEmpty)
    //assert((foNone === foStr1p0(1)).ot.isEmpty)

    assert((foStr1 + foStr1p0 === "11.0").ot.get)
    //assert(("11.0" === foStr1 + foStr1p0).ot.get)


    val foZero = new FilterOption(Some(0))
    val foOne = new FilterOption(Some(1))
    val foTwo = new FilterOption(Some(2))
    val foHalf = new FilterOption(Some(.5))
    val foTwoD = new FilterOption(Some(2.0))
    val foThreeD = new FilterOption(Some(3.0))

    val arr = Array(0,1)
    val foArr = new FilterOption(Some(arr))

    //assert(2 === foArr.size)
    assert((foArr.size === 2).ot.get)
    assert((foTwo === foArr.size).ot.get)
    assert((foArr.size === foTwo).ot.get)

    //assert((0 === foArr(0)).ot.get)
    assert((foArr(0) === 0).ot.get)
    assert((foZero === foArr(0)).ot.get)
    assert((foArr(0) === foZero).ot.get)


    //assert((0 < foOne).ot.get)
    assert((foZero < 1).ot.get)
    assert((foZero < foOne).ot.get)

    //assert(0 < foNone).ot.isEmpty)
    //assert((foNone < 0).ot.isEmpty)
    //assert((foZero < foNone).ot.isEmpty)
    //assert((foNone < foZero).ot.isEmpty)


    assert(1 === 1)
    assert(1 === 1.0)
    assert(1.0 === 1)
    assert(1.0 === 1.0)

    //assert((1 + foOne === 2).ot.get)
    //assert((foOne + 1 === 2).ot.get)
    //assert((foOne + foOne === foTwo).ot.get)
    //assert((foOne + foTwo === foThreeD).ot.get)
    //assert((foThreeD === foOne + foTwo).ot.get)


    //assert((1 * foOne === 1).ot.get)
    //assert((foOne * 1 === 1).ot.get)
    //assert((1 * foOne === foOne).ot.get)
    //assert((foOne * 1 === foOne).ot.get)
    //assert((foOne * foOne === foOne).ot.get)

    //assert((1 / foTwoD === .5).ot.get)
    //assert((foOne / 2.0 === .5).ot.get)
    //assert((.5 === foOne / 2.0).ot.get)
    //assert((.5 === 1 / foTwoD).ot.get)
    assert((foOne / foTwoD === foHalf).ot.get)
    assert((foHalf === foOne / foTwoD).ot.get)


    assert(1 === 1.0)
    assert(1.0 === 1)
    assert(1.0 === 1.0)



  }


  /* forAll { (i: Int, j: Int)
    FilterOption(i < j) == i < FilterOption(j) } */
  /* FilterOption(None) < 5 == FilterOption(None) */
  // keep, remove
  // filter options work in Eval (just some)

  // FilterGenotype(val g: Genotype) extends AnyVal { def gq: FilterOption[Int] = filterOptionFromOption(g.gq) }
  // use FilterGenotype in evaluator
  // test these in both cases


  @Test def test() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")
    val state = State("", sc, sqlContext, vds)

    assert(FilterSamples.run(state, Array("--keep", "-c", "\"^HG\" ~ s.id"))
      .vds.nLocalSamples == 63)

    assert(FilterVariants.run(state, Array("--remove", "-c", "v.start >= 14066228"))
      .vds.nVariants == 173)

    val highGQ = FilterGenotypes.run(state, Array("--remove", "-c", "g.call.exists(c => c.gq < 20)"))
      .vds.expand().collect()

    assert(!highGQ.exists { case (v, s, g) => g.call.exists(c => c.gq < 20) })
    assert(highGQ.count{ case (v, s, g) => g.call.exists(c => c.gq >= 20) } == 31260)

    val vds2 = TestRDDBuilder.buildRDD(1, 1, sc)
    val state2 = State("", sc, sqlContext, vds2)
    val nVariants = vds2.nVariants


  }
}
