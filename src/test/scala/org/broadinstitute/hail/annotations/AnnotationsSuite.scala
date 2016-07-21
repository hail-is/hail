package org.broadinstitute.hail.annotations

import org.apache.spark.sql.types._
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant.Variant
import org.testng.annotations.Test

import scala.collection.mutable
import scala.language.implicitConversions

/**
  * This testing suite evaluates the functionality of the [[org.broadinstitute.hail.annotations]] package
  */
class AnnotationsSuite extends SparkSuite {
  @Test def test() {
    /*
      The below tests are designed to check for a subset of variants and info fields, that:
          1. the types, emitConversionIdentifier strings, and description strings agree with the VCF
          2. the strings stored in the AnnotationData classes agree with the VCF
          3. the strings stored in the AnnotationData classes convert correctly to the proper type
    */

    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")

    val state = State(sc, sqlContext, vds)
    val vas = vds.vaSignature
    val variantAnnotationMap = vds.variantsAndAnnotations.collect().toMap

    val firstVariant = Variant("20", 10019093, "A", "G")
    val anotherVariant = Variant("20", 10036107, "T", "G")
    assert(variantAnnotationMap.contains(firstVariant))
    assert(variantAnnotationMap.contains(anotherVariant))

    // type Int - info.DP
    val dpQuery = vas.query("info", "DP")
    assert(vas.fieldOption("info", "DP").exists(f =>
      f.`type` == TInt
        && f.attrs == Map("Type" -> "Integer",
        "Number" -> "1",
        "Description" -> "Approximate read depth; some reads may have been filtered")))
    assert(dpQuery(variantAnnotationMap(firstVariant)).contains(77560))
    assert(dpQuery(variantAnnotationMap(anotherVariant)).contains(20271))

    // type Double - info.HWP
    val hwpQuery = vas.query("info", "HWP")
    assert(vas.fieldOption("info", "HWP").exists(f =>
      f.`type` == TDouble
        && f.attrs == Map("Type" -> "Float",
        "Number" -> "1",
        "Description" -> "P value from test of Hardy Weinberg Equilibrium")))
    assert(
      D_==(hwpQuery(variantAnnotationMap(firstVariant))
        .get.asInstanceOf[Double], 0.0001))
    assert(D_==(hwpQuery(variantAnnotationMap(anotherVariant))
      .get.asInstanceOf[Double], 0.8286))

    // type String - info.culprit
    val culpritQuery = vas.query("info", "culprit")
    assert(vas.fieldOption("info", "culprit").exists(f =>
      f.`type` == TString
        && f.attrs == Map("Type" -> "String",
        "Number" -> "1",
        "Description" -> "The annotation which was the worst performing in the Gaussian mixture model, likely the reason why the variant was filtered out")))
    assert(culpritQuery(variantAnnotationMap(firstVariant))
      .contains("FS"))
    assert(culpritQuery(variantAnnotationMap(anotherVariant))
      .contains("FS"))

    // type Array - info.AC (allele count)
    val acQuery = vas.query("info", "AC")
    assert(vas.fieldOption("info", "AC").exists(f =>
      f.`type` == TArray(TInt) &&
        f.attrs == Map("Number" -> "A",
          "Type" -> "Integer",
          "Description" -> "Allele count in genotypes, for each ALT allele, in the same order as listed")))
    assert(acQuery(variantAnnotationMap(firstVariant))
      .contains(Array(89): mutable.WrappedArray[Int]))
    assert(acQuery(variantAnnotationMap(anotherVariant))
      .contains(Array(13): mutable.WrappedArray[Int]))

    // type Boolean/flag - info.DB (dbSNP membership)
    val dbQuery = vas.query("info", "DB")
    assert(vas.fieldOption("info", "DB").exists(f =>
      f.`type` == TBoolean
        && f.attrs == Map("Type" -> "Flag",
        "Number" -> "0",
        "Description" -> "dbSNP Membership")))
    assert(dbQuery(variantAnnotationMap(firstVariant))
      .contains(true))
    assert(dbQuery(variantAnnotationMap(anotherVariant))
      .isEmpty)

    //type Set[String]

    val filtQuery = vas.query("filters")
    assert(vas.fieldOption("filters").exists(f =>
      f.`type` == TSet(TString)
        && f.attrs.nonEmpty))
    assert(filtQuery(variantAnnotationMap(firstVariant))
      contains Set("PASS"))
    assert(filtQuery(variantAnnotationMap(anotherVariant))
      contains Set("VQSRTrancheSNP99.95to100.00"))

    // GATK PASS
    val passQuery = vas.query("pass")
    assert(vas.fieldOption("pass").exists(f => f.`type` == TBoolean
      && f.attrs == Map.empty))
    assert(passQuery(variantAnnotationMap(firstVariant))
      .contains(true))
    assert(passQuery(variantAnnotationMap(anotherVariant))
      .contains(false))

    val vds2 = LoadVCF(sc, "src/test/resources/sample2.vcf")
    val vas2 = vds2.vaSignature

    // Check that VDS can be written to disk and retrieved while staying the same
    val f = tmpDir.createTempFile("sample", extension = ".vds")
    vds2.write(sqlContext, f)
    val readBack = Read.run(state, Array("-i", f))

    assert(readBack.vds.same(vds2))
  }

  @Test def testReadWrite() {
    val vds1 = LoadVCF(sc, "src/test/resources/sample.vcf")
    val s = State(sc, sqlContext, vds1)
    val vds2 = LoadVCF(sc, "src/test/resources/sample.vcf")
    assert(vds1.same(vds2))

    val f = tmpDir.createTempFile("sample", extension = ".vds")
    Write.run(s, Array("-o", f))
    val vds3 = Read.run(s, Array("-i", f)).vds
    assert(vds3.same(vds1))
  }

  @Test def testAnnotationOperations() {

    /*
      This test method performs a number of annotation operations on a vds, and ensures that the signatures
      and annotations in the RDD elements are what is expected after each step.  In particular, we want to
      test overwriting behavior, deleting, appending, and querying.
    */

    var vds = LoadVCF(sc, "src/test/resources/sample.vcf")
      .cache()

    // clear everything
    val (emptyS, d1) = vds.deleteVA()
    vds = vds.mapAnnotations((v, va, gs) => d1(va))
      .copy(vaSignature = emptyS)
    assert(emptyS == TStruct.empty)

    // add to the first layer
    val toAdd = 5
    val toAddSig = TInt
    val (s1, i1) = vds.vaSignature.insert(toAddSig, "I1")
    vds = vds.mapAnnotations((v, va, gs) => i1(va, Some(toAdd)))
      .copy(vaSignature = s1)
    assert(vds.vaSignature.schema ==
      StructType(Array(StructField("I1", IntegerType))))

    val q1 = vds.queryVA("va.I1")._2
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => q1(va) == Some(5) })

    // add another to the first layer
    val toAdd2 = "test"
    val toAdd2Sig = TString
    val (s2, i2) = vds.vaSignature.insert(toAdd2Sig, "S1")
    vds = vds.mapAnnotations((v, va, gs) => i2(va, Some(toAdd2)))
      .copy(vaSignature = s2)
    assert(vds.vaSignature.schema ==
      StructType(Array(
        StructField("I1", IntegerType),
        StructField("S1", StringType))))

    val q2 = vds.queryVA("va.S1")._2
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => q2(va) == Some("test") })

    // overwrite I1 with a row in the second layer
    val toAdd3 = Annotation(1, 3)
    val toAdd3Sig = TStruct("I2" -> TInt,
      "I3" -> TInt)
    val (s3, i3) = vds.vaSignature.insert(toAdd3Sig, "I1")
    vds = vds.mapAnnotations((v, va, gs) => i3(va, Some(toAdd3)))
      .copy(vaSignature = s3)
    assert(vds.vaSignature.schema ==
      StructType(Array(
        StructField("I1", StructType(Array(
          StructField("I2", IntegerType, nullable = true),
          StructField("I3", IntegerType, nullable = true)
        )), nullable = true),
        StructField("S1", StringType))))

    val q3 = vds.queryVA("va.I1")._2
    val q4 = vds.queryVA("va.I1.I2")._2
    val q5 = vds.queryVA("va.I1.I3")._2
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) =>
        (q3(va) == Some(Annotation(1, 3))) &&
          (q4(va) == Some(1)) &&
          (q5(va) == Some(3))
      })

    // add something deep in the tree with an unbuilt structure
    val toAdd4 = "dummy"
    val toAdd4Sig = TString
    val (s4, i4) = vds.insertVA(toAdd4Sig, "a", "b", "c", "d", "e")
    vds = vds.mapAnnotations((v, va, gs) => i4(va, Some(toAdd4)))
      .copy(vaSignature = s4)
    assert(vds.vaSignature.schema ==
      StructType(Array(
        StructField("I1", toAdd3Sig.schema),
        StructField("S1", StringType),
        StructField("a", StructType(Array(
          StructField("b", StructType(Array(
            StructField("c", StructType(Array(
              StructField("d", StructType(Array(
                StructField("e", StringType))))))))))))))))

    val q6 = vds.queryVA("va.a.b.c.d.e")._2
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => q6(va) == Some("dummy") })

    // add something as a sibling deep in the tree
    val toAdd5 = "dummy2"
    val toAdd5Sig = TString
    val (s5, i5) = vds.insertVA(toAdd5Sig, "a", "b", "c", "f")
    vds = vds.mapAnnotations((v, va, gs) => i5(va, Some(toAdd5)))
      .copy(vaSignature = s5)

    assert(vds.vaSignature.schema ==
      StructType(Array(
        StructField("I1", toAdd3Sig.schema),
        StructField("S1", StringType),
        StructField("a", StructType(Array(
          StructField("b", StructType(Array(
            StructField("c", StructType(Array(
              StructField("d", StructType(Array(
                StructField("e", StringType)))),
              StructField("f", StringType)))))))))))))
    val q7 = vds.queryVA("va.a.b.c.f")._2
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => q7(va) == Some("dummy2") })

    // overwrite something deep in the tree
    val toAdd6 = "dummy3"
    val toAdd6Sig = TString
    val (s6, i6) = vds.insertVA(toAdd6Sig, "a", "b", "c", "d")
    vds = vds.mapAnnotations((v, va, gs) => i6(va, Some(toAdd6)))
      .copy(vaSignature = s6)

    assert(vds.vaSignature.schema ==
      StructType(Array(
        StructField("I1", toAdd3Sig.schema),
        StructField("S1", StringType),
        StructField("a", StructType(Array(
          StructField("b", StructType(Array(
            StructField("c", StructType(Array(
              StructField("d", StringType),
              StructField("f", StringType)))))))))))))

    val q8 = vds.queryVA("va.a.b.c.d")._2
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => q8(va) == Some("dummy3") })

    val toAdd7 = "dummy4"
    val toAdd7Sig = TString
    val (s7, i7) = vds.insertVA(toAdd7Sig, "a", "c")
    vds = vds.mapAnnotations((v, va, gs) => i7(va, Some(toAdd7)))
      .copy(vaSignature = s7)

    assert(vds.vaSignature.schema ==
      StructType(Array(
        StructField("I1", toAdd3Sig.schema),
        StructField("S1", StringType),
        StructField("a", StructType(Array(
          StructField("b", StructType(Array(
            StructField("c", StructType(Array(
              StructField("d", StringType),
              StructField("f", StringType))))))),
          StructField("c", StringType)))))))
    val q9 = vds.queryVA("va.a.c")._2
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => q9(va) == Some(toAdd7) })

    // delete a.b.c and ensure that b is deleted and a.c gets shifted over
    val (s8, d2) = vds.deleteVA("a", "b", "c")
    vds = vds.mapAnnotations((v, va, gs) => d2(va))
      .copy(vaSignature = s8)
    assert(vds.vaSignature.schema ==
      StructType(Array(
        StructField("I1", toAdd3Sig.schema),
        StructField("S1", StringType),
        StructField("a", StructType(Array(
          StructField("c", StringType)))))))
    val q10 = vds.queryVA("va.a")._2
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => q10(va) == Some(Annotation(toAdd7)) })

    // delete that part of the tree
    val (s9, d3) = vds.deleteVA("a")
    vds = vds.mapAnnotations((v, va, gs) => d3(va))
      .copy(vaSignature = s9)

    assert(vds.vaSignature.schema ==
      StructType(Array(
        StructField("I1", toAdd3Sig.schema),
        StructField("S1", StringType))))

    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => va == Annotation(toAdd3, "test") })

    // delete the first thing in the row and make sure things are shifted over correctly
    val (s10, d4) = vds.deleteVA("I1")
    vds = vds.mapAnnotations((v, va, gs) => d4(va))
      .copy(vaSignature = s10)

    assert(vds.vaSignature.schema ==
      StructType(Array(
        StructField("S1", StringType))))
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => va == Annotation("test") })

    // remap the head
    val toAdd8 = "dummy"
    val toAdd8Sig = TString
    val (s11, i8) = vds.insertVA(toAdd8Sig, List[String]())
    vds = vds.mapAnnotations((v, va, gs) => i8(va, Some(toAdd8)))
      .copy(vaSignature = s11)

    assert(vds.vaSignature.schema == toAdd8Sig.schema)
    assert(vds.variantsAndAnnotations.collect()
      .forall { case (v, va) => va == "dummy" })
  }

  @Test def testWeirdNamesReadWrite() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")

    var state = State(sc, sqlContext, vds)
    state = SplitMulti.run(state)

    val (newS, ins) = vds.insertVA(TInt, "ThisName(won'twork)=====")
    state = state.copy(vds = vds.mapAnnotations((v, va, gs) => ins(va, Some(5)))
      .copy(vaSignature = newS))

    val f = tmpDir.createTempFile("testwrite", extension = ".vds")

    Write.run(state, Array("-o", f))
    val state2 = Read.run(state, Array("-i", f))
    assert(state.vds.same(state2.vds))
  }
}
