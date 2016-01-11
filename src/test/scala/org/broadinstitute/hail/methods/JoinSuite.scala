package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.utils.TestRDDBuilder
import org.testng.annotations.Test

class JoinSuite extends SparkSuite {
  @Test def test() {

    def convertToSome[T](x:T):Option[T] = {
      if (x == None) None else Some(x)
    }

    val vsm1 = TestRDDBuilder.buildRDD(5, 3, sc, sampleIds = Some(Array("s1", "s2", "s3", "s4", "s5")))
    val vsm2 = TestRDDBuilder.buildRDD(5, 5, sc, sampleIds = Some(Array("foo", "s7", "beet", "pug", "s3")))
    vsm1.cache()
    vsm2.cache()

    val expectedNumSamples = Map("inner" -> 1, "left" -> 5, "right" -> 5, "outer" -> 9)
    val expectedNumVariants = Map("inner" -> 3, "left" -> 3, "right" -> 5, "outer" -> 5)

    val joinTypes = Array("inner", "left", "right", "outer")

    for (sjt <- joinTypes; vjt <- joinTypes) {
      val nSamples = expectedNumSamples(sjt)
      val nVariants = expectedNumVariants(vjt)

      val mergedVSM = (sjt, vjt) match {
        case ("inner", "inner") => vsm1.joinInnerInner(vsm2)
        case ("inner", "left") => vsm1.joinInnerLeft(vsm2)
        case ("inner", "right") => vsm1.joinInnerRight(vsm2)
        case ("inner", "outer") => vsm1.joinInnerOuter(vsm2)
        case ("left", "inner") => vsm1.joinLeftInner(vsm2)
        case ("left", "left") => vsm1.joinLeftLeft(vsm2)
        case ("left", "right") => vsm1.joinLeftRight(vsm2)
        case ("left", "outer") => vsm1.joinLeftOuter(vsm2)
        case ("right", "inner") => vsm1.joinRightInner(vsm2)
        case ("right", "left") => vsm1.joinRightLeft(vsm2)
        case ("right", "right") => vsm1.joinRightRight(vsm2)
        case ("right", "outer") => vsm1.joinRightOuter(vsm2)
        case ("outer", "inner") => vsm1.joinOuterInner(vsm2)
        case ("outer", "left") => vsm1.joinOuterLeft(vsm2)
        case ("outer", "right") => vsm1.joinOuterRight(vsm2)
        case ("outer", "outer") => vsm1.joinOuterOuter(vsm2)
        case _ => throw new UnsupportedOperationException
      }

      assert(mergedVSM.localSamples.length == nSamples)
      assert(mergedVSM.rdd.filter{case (v,a,g) => g.size == nSamples}.count == nVariants)
      
      val vsm1SampleIdsLocal = vsm1.sampleIds
      val vsm2SampleIdsLocal = vsm2.sampleIds
      val mergeSampleIdsLocal = mergedVSM.sampleIds

      val vsm1ExpandedMap = vsm1.expand().map{case (v,s,g) => ((v,vsm1SampleIdsLocal(s)),g)}.collectAsMap
      val vsm2ExpandedMap = vsm2.expand().map{case (v,s,g) => ((v,vsm2SampleIdsLocal(s)),g)}.collectAsMap
      val mergedExpandedMap = mergedVSM.expand().map{case (v,s,g) => ((v,mergeSampleIdsLocal(s)),g)}.collectAsMap

      for (((v,s),gtMerge) <- mergedExpandedMap) {
        val g1 = vsm1ExpandedMap.get((v, s)) match {
          case Some(x) => x
          case None => None
        }
        val g2 = vsm2ExpandedMap.get((v, s)) match {
          case Some(x) => x
          case None => None
        }

        val gtActual = (sjt,vjt) match {
          case ("inner","inner") => (g1,g2)
          case ("inner","left") => (g1,convertToSome(g2))
          case ("inner","right") => (convertToSome(g1),g2)
          case ("inner","outer") => (convertToSome(g1),convertToSome(g2))
          case ("left","inner") => (g1,convertToSome(g2))
          case ("left","left") => (g1,convertToSome(g2))
          case ("left","right") => (convertToSome(g1),convertToSome(g2))
          case ("left","outer") => (convertToSome(g1),convertToSome(g2))
          case ("right","inner") => (convertToSome(g1),g2)
          case ("right","left") => (convertToSome(g1),convertToSome(g2))
          case ("right","right") => (convertToSome(g1),g2)
          case ("right","outer") => (convertToSome(g1),convertToSome(g2))
          case ("outer","inner") => (convertToSome(g1),convertToSome(g2))
          case ("outer","left") => (convertToSome(g1),convertToSome(g2))
          case ("outer","right") => (convertToSome(g1),convertToSome(g2))
          case ("outer","outer") => (convertToSome(g1),convertToSome(g2))
          case _ => throw new UnsupportedOperationException
        }
        assert(gtMerge == gtActual)
      }
    }
  }
}

