package is.hail.annotations

import is.hail.SparkSuite
import is.hail.expr._
import is.hail.methods.SplitMulti
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant.Variant
import org.apache.spark.sql.types._
import org.testng.annotations.Test

import scala.collection.mutable
import scala.language.implicitConversions

/**
  * This testing suite evaluates the functionality of the [[is.hail.annotations]] package
  */
class AnnotationsSuite extends SparkSuite {
  @Test def test() {
    /*
      The below tests are designed to check for a subset of variants and info fields, that:
          1. the types, emitConversionIdentifier strings, and description strings agree with the VCF
          2. the strings stored in the AnnotationData classes agree with the VCF
          3. the strings stored in the AnnotationData classes convert correctly to the proper type
    */

    val vds = hc.importVCF("src/test/resources/sample.vcf")

    val vas = vds.vaSignature
    val variantAnnotationMap = vds.variantsAndAnnotations.collect().toMap

    val firstVariant = Variant("20", 10019093, "A", "G")
    val anotherVariant = Variant("20", 10036107, "T", "G")
    assert(variantAnnotationMap.contains(firstVariant))
    assert(variantAnnotationMap.contains(anotherVariant))

    // type Int - info.DP
    val dpQuery = vas.query("info", "DP")
    assert(vas.fieldOption("info", "DP").exists(_.typ.isInstanceOf[TInt32]))
    assert(dpQuery(variantAnnotationMap(firstVariant)) == 77560)
    assert(dpQuery(variantAnnotationMap(anotherVariant)) == 20271)

    // type Double - info.HWP
    val hwpQuery = vas.query("info", "HWP")
    assert(vas.fieldOption("info", "HWP").exists(_.typ.isInstanceOf[TFloat64]))
    assert(D_==(hwpQuery(variantAnnotationMap(firstVariant)).asInstanceOf[Double], 0.0001))
    assert(D_==(hwpQuery(variantAnnotationMap(anotherVariant)).asInstanceOf[Double], 0.8286))

    // type String - info.culprit
    val culpritQuery = vas.query("info", "culprit")
    assert(vas.fieldOption("info", "culprit").exists(_.typ.isInstanceOf[TString]))
    assert(culpritQuery(variantAnnotationMap(firstVariant)) == "FS")
    assert(culpritQuery(variantAnnotationMap(anotherVariant)) == "FS")

    // type Array - info.AC (allele count)
    val acQuery = vas.query("info", "AC")
    assert(vas.fieldOption("info", "AC").exists(f => f.typ == TArray(TInt32()) || f.typ == TArray(!TInt32())))
    assert(acQuery(variantAnnotationMap(firstVariant)) == IndexedSeq(89))
    assert(acQuery(variantAnnotationMap(anotherVariant)) == IndexedSeq(13))

    // type Boolean/flag - info.DB (dbSNP membership)
    val dbQuery = vas.query("info", "DB")
    assert(vas.fieldOption("info", "DB").exists(_.typ.isInstanceOf[TBoolean]))
    assert(dbQuery(variantAnnotationMap(firstVariant)) == true)
    assert(dbQuery(variantAnnotationMap(anotherVariant)) == null)

    //type Set[String]
    val filtQuery = vas.query("filters")
    assert(vas.fieldOption("filters").exists(f => f.typ == TSet(TString()) || f.typ == TSet(!TString())))
    assert(filtQuery(variantAnnotationMap(firstVariant)) == Set())
    assert(filtQuery(variantAnnotationMap(anotherVariant)) == Set("VQSRTrancheSNP99.95to100.00"))
  }

  @Test def testReadWrite() {
    val vds1 = hc.importVCF("src/test/resources/sample.vcf")
    val vds2 = hc.importVCF("src/test/resources/sample.vcf")
    assert(vds1.same(vds2))

    val f = tmpDir.createTempFile("sample", extension = ".vds")
    vds1.write(f)
    val vds3 = hc.readVDS(f)
    assert(vds3.same(vds1))
  }

  @Test def testAnnotationOperations() {

    /*
      This test method performs a number of annotation operations on a vds, and ensures that the signatures
      and annotations in the RDD elements are what is expected after each step.  In particular, we want to
      test overwriting behavior, deleting, appending, and querying.
    */

    var vds = hc.importVCF("src/test/resources/sample.vcf").cache()

    // clear everything
    val (emptyS, d1) = vds.deleteVA()
    vds = vds.mapAnnotations(emptyS, (v, va, gs) => d1(va))
    assert(emptyS == TStruct.empty())

    // add to the first layer
    val toAdd = 5
    val toAddSig = TInt32()
    val (s1, i1) = vds.vaSignature.insert(toAddSig, "I1")
    vds = vds.mapAnnotations(s1, (v, va, gs) => i1(va, toAdd))
    assert(vds.vaSignature.schema ==
      StructType(Array(StructField("I1", IntegerType))))

    val (_, q1) = vds.queryVA("va.I1")
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => q1(va) == 5 })

    // add another to the first layer
    val toAdd2 = "test"
    val toAdd2Sig = TString()
    val (s2, i2) = vds.vaSignature.insert(toAdd2Sig, "S1")
    vds = vds.mapAnnotations(s2, (v, va, gs) => i2(va, toAdd2))
    assert(vds.vaSignature.schema ==
      StructType(Array(
        StructField("I1", IntegerType),
        StructField("S1", StringType))))

    val (_, q2) = vds.queryVA("va.S1")
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => q2(va) == "test" })

    // overwrite I1 with a row in the second layer
    val toAdd3 = Annotation(1, 3)
    val toAdd3Sig = TStruct("I2" -> TInt32(),
      "I3" -> TInt32())
    val (s3, i3) = vds.vaSignature.insert(toAdd3Sig, "I1")
    vds = vds.mapAnnotations(s3, (v, va, gs) => i3(va, toAdd3))
    assert(vds.vaSignature.schema ==
      StructType(Array(
        StructField("I1", StructType(Array(
          StructField("I2", IntegerType, nullable = true),
          StructField("I3", IntegerType, nullable = true)
        )), nullable = true),
        StructField("S1", StringType))))

    val (_, q3) = vds.queryVA("va.I1")
    val (_, q4) = vds.queryVA("va.I1.I2")
    val (_, q5) = vds.queryVA("va.I1.I3")
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) =>
        (q3(va) == Annotation(1, 3)) &&
          (q4(va) == 1) &&
          (q5(va) == 3)
      })

    // add something deep in the tree with an unbuilt structure
    val toAdd4 = "dummy"
    val toAdd4Sig = TString()
    val (s4, i4) = vds.insertVA(toAdd4Sig, "a", "b", "c", "d", "e")
    vds = vds.mapAnnotations(s4, (v, va, gs) => i4(va, toAdd4))
    assert(vds.vaSignature.schema ==
      StructType(Array(
        StructField("I1", toAdd3Sig.schema),
        StructField("S1", StringType),
        StructField("a", StructType(Array(
          StructField("b", StructType(Array(
            StructField("c", StructType(Array(
              StructField("d", StructType(Array(
                StructField("e", StringType))))))))))))))))

    val (_, q6) = vds.queryVA("va.a.b.c.d.e")
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => q6(va) == "dummy" })

    // add something as a sibling deep in the tree
    val toAdd5 = "dummy2"
    val toAdd5Sig = TString()
    val (s5, i5) = vds.insertVA(toAdd5Sig, "a", "b", "c", "f")
    vds = vds.mapAnnotations(s5, (v, va, gs) => i5(va, toAdd5))

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
    val (_, q7) = vds.queryVA("va.a.b.c.f")
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => q7(va) == "dummy2" })

    // overwrite something deep in the tree
    val toAdd6 = "dummy3"
    val toAdd6Sig = TString()
    val (s6, i6) = vds.insertVA(toAdd6Sig, "a", "b", "c", "d")
    vds = vds.mapAnnotations(s6, (v, va, gs) => i6(va, toAdd6))

    assert(vds.vaSignature.schema ==
      StructType(Array(
        StructField("I1", toAdd3Sig.schema),
        StructField("S1", StringType),
        StructField("a", StructType(Array(
          StructField("b", StructType(Array(
            StructField("c", StructType(Array(
              StructField("d", StringType),
              StructField("f", StringType)))))))))))))

    val (_, q8) = vds.queryVA("va.a.b.c.d")
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => q8(va) == "dummy3" })

    val toAdd7 = "dummy4"
    val toAdd7Sig = TString()
    val (s7, i7) = vds.insertVA(toAdd7Sig, "a", "c")
    vds = vds.mapAnnotations(s7, (v, va, gs) => i7(va, toAdd7))

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
    val (_, q9) = vds.queryVA("va.a.c")
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => q9(va) == toAdd7 })

    // delete a.b.c and ensure that b is deleted and a.c gets shifted over
    val (s8, d2) = vds.deleteVA("a", "b", "c")
    vds = vds.mapAnnotations(s8, (v, va, gs) => d2(va))
    assert(vds.vaSignature.schema ==
      StructType(Array(
        StructField("I1", toAdd3Sig.schema),
        StructField("S1", StringType),
        StructField("a", StructType(Array(
          StructField("c", StringType)))))))
    val (_, q10) = vds.queryVA("va.a")
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => q10(va) == Annotation(toAdd7) })

    // delete that part of the tree
    val (s9, d3) = vds.deleteVA("a")
    vds = vds.mapAnnotations(s9, (v, va, gs) => d3(va))

    assert(vds.vaSignature.schema ==
      StructType(Array(
        StructField("I1", toAdd3Sig.schema),
        StructField("S1", StringType))))

    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => va == Annotation(toAdd3, "test") })

    // delete the first thing in the row and make sure things are shifted over correctly
    val (s10, d4) = vds.deleteVA("I1")
    vds = vds.mapAnnotations(s10, (v, va, gs) => d4(va))

    assert(vds.vaSignature.schema ==
      StructType(Array(
        StructField("S1", StringType))))
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => va == Annotation("test") })

    // remap the head
    val toAdd8 = "dummy"
    val toAdd8Sig = TString()
    val (s11, i8) = vds.insertVA(toAdd8Sig, List[String]())
    vds = vds.mapAnnotations(s11, (v, va, gs) => i8(va, toAdd8))

    assert(vds.vaSignature.schema == toAdd8Sig.schema)
    assert(vds.variantsAndAnnotations.collect()
      .forall { case (v, va) => va == "dummy" })
  }

  @Test def testWeirdNamesReadWrite() {
    var vds = SplitMulti(hc.importVCF("src/test/resources/sample.vcf"))

    val f = tmpDir.createTempFile("testwrite", extension = ".vds")
    val (newS, ins) = vds.insertVA(TInt32(), "ThisName(won'twork)=====")
    vds = vds.mapAnnotations(newS, (v, va, gs) => ins(va, 5))
    vds.write(f)

    assert(hc.readVDS(f).same(vds))
  }
}
