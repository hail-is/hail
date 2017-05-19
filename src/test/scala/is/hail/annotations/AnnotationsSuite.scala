package is.hail.annotations

import is.hail.SparkSuite
import is.hail.expr._
import is.hail.utils._
import is.hail.variant.{GenotypeMatrixT, Variant}
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
    assert(vas.fieldOption("info", "DP").exists(f =>
      f.typ == TInt
        && f.attrs == Map("Type" -> "Integer",
        "Number" -> "1",
        "Description" -> "Approximate read depth; some reads may have been filtered")))
    assert(dpQuery(variantAnnotationMap(firstVariant)) == 77560)
    assert(dpQuery(variantAnnotationMap(anotherVariant)) == 20271)

    // type Double - info.HWP
    val hwpQuery = vas.query("info", "HWP")
    assert(vas.fieldOption("info", "HWP").exists(f =>
      f.typ == TDouble
        && f.attrs == Map("Type" -> "Float",
        "Number" -> "1",
        "Description" -> "P value from test of Hardy Weinberg Equilibrium")))
    assert(D_==(hwpQuery(variantAnnotationMap(firstVariant)).asInstanceOf[Double], 0.0001))
    assert(D_==(hwpQuery(variantAnnotationMap(anotherVariant)).asInstanceOf[Double], 0.8286))

    // type String - info.culprit
    val culpritQuery = vas.query("info", "culprit")
    assert(vas.fieldOption("info", "culprit").exists(f =>
      f.typ == TString
        && f.attrs == Map("Type" -> "String",
        "Number" -> "1",
        "Description" -> "The annotation which was the worst performing in the Gaussian mixture model, likely the reason why the variant was filtered out")))
    assert(culpritQuery(variantAnnotationMap(firstVariant)) == "FS")
    assert(culpritQuery(variantAnnotationMap(anotherVariant)) == "FS")

    // type Array - info.AC (allele count)
    val acQuery = vas.query("info", "AC")
    assert(vas.fieldOption("info", "AC").exists(f =>
      f.typ == TArray(TInt) &&
        f.attrs == Map("Number" -> "A",
          "Type" -> "Integer",
          "Description" -> "Allele count in genotypes, for each ALT allele, in the same order as listed")))
    assert(acQuery(variantAnnotationMap(firstVariant)) == IndexedSeq(89))
    assert(acQuery(variantAnnotationMap(anotherVariant)) == IndexedSeq(13))

    // type Boolean/flag - info.DB (dbSNP membership)
    val dbQuery = vas.query("info", "DB")
    assert(vas.fieldOption("info", "DB").exists(f =>
      f.typ == TBoolean
        && f.attrs == Map("Type" -> "Flag",
        "Number" -> "0",
        "Description" -> "dbSNP Membership")))
    assert(dbQuery(variantAnnotationMap(firstVariant)) == true)
    assert(dbQuery(variantAnnotationMap(anotherVariant)) == null)

    //type Set[String]

    val filtQuery = vas.query("filters")
    assert(vas.fieldOption("filters").exists(f =>
      f.typ == TSet(TString)
        && f.attrs.nonEmpty))
    assert(filtQuery(variantAnnotationMap(firstVariant)) == Set())
    assert(filtQuery(variantAnnotationMap(anotherVariant)) == Set("VQSRTrancheSNP99.95to100.00"))

    val vds2 = hc.importVCF("src/test/resources/sample2.vcf")
    val vas2 = vds2.vaSignature

    // Check that VDS can be written to disk and retrieved while staying the same
    val f = tmpDir.createTempFile("sample", extension = ".vds")
    vds2.write(f)
    val readBack = hc.readVDS(f)

    assert(readBack.same(vds2))

    //Check that adding attributes to FILTERS / INFO outputs the correct Number/Description
    val vds_attr = vds
      .setVaAttributes("va.filters", Map("testFilter" -> "testFilterDesc"))
      .setVaAttributes("va.info.MQ", Map("Number" -> ".", "Description" -> "testMQ", "foo" -> "bar"))

    assert(vds_attr.vaSignature.fieldOption("filters")
      .map(f => f.attrs.getOrElse("testFilter", "") == "testFilterDesc")
      .getOrElse(false))

    assert(vds_attr.vaSignature.fieldOption(List("info", "MQ"))
      .map(f => f.attrs.getOrElse("foo", "") == "bar")
      .getOrElse(false))
    assert(vds_attr.vaSignature.fieldOption(List("info", "MQ"))
      .map(f => f.attrs.getOrElse("Description", "") == "testMQ")
      .getOrElse(false))
    assert(vds_attr.vaSignature.fieldOption(List("info", "MQ"))
      .map(f => f.attrs.getOrElse("Number", "") == ".")
      .getOrElse(false))


    // Write VCF and check that annotaions are the same
    val f2 = tmpDir.createTempFile("sample2", extension = ".vds")
    vds_attr.write(f2)
    val readBack_attr = hc.readVDS(f2)

    assert(readBack_attr.same(vds_attr))
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
    vds = vds.mapAnnotations((v, va, gs) => d1(va))
      .copy[GenotypeMatrixT](vaSignature = emptyS)
    assert(emptyS == TStruct.empty)

    // add to the first layer
    val toAdd = 5
    val toAddSig = TInt
    val (s1, i1) = vds.vaSignature.insert(toAddSig, "I1")
    vds = vds.mapAnnotations((v, va, gs) => i1(va, toAdd))
      .copy[GenotypeMatrixT](vaSignature = s1)
    assert(vds.vaSignature.schema ==
      StructType(Array(StructField("I1", IntegerType))))

    val (_, q1) = vds.queryVA("va.I1")
    assert(vds.variantsAndAnnotations
      .collect()
      .forall { case (v, va) => q1(va) == 5 })

    // add another to the first layer
    val toAdd2 = "test"
    val toAdd2Sig = TString
    val (s2, i2) = vds.vaSignature.insert(toAdd2Sig, "S1")
    vds = vds.mapAnnotations((v, va, gs) => i2(va, toAdd2))
      .copy[GenotypeMatrixT](vaSignature = s2)
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
    val toAdd3Sig = TStruct("I2" -> TInt,
      "I3" -> TInt)
    val (s3, i3) = vds.vaSignature.insert(toAdd3Sig, "I1")
    vds = vds.mapAnnotations((v, va, gs) => i3(va, toAdd3))
      .copy[GenotypeMatrixT](vaSignature = s3)
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
    val toAdd4Sig = TString
    val (s4, i4) = vds.insertVA(toAdd4Sig, "a", "b", "c", "d", "e")
    vds = vds.mapAnnotations((v, va, gs) => i4(va, toAdd4))
      .copy[GenotypeMatrixT](vaSignature = s4)
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
    val toAdd5Sig = TString
    val (s5, i5) = vds.insertVA(toAdd5Sig, "a", "b", "c", "f")
    vds = vds.mapAnnotations((v, va, gs) => i5(va, toAdd5))
      .copy[GenotypeMatrixT](vaSignature = s5)

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
    val toAdd6Sig = TString
    val (s6, i6) = vds.insertVA(toAdd6Sig, "a", "b", "c", "d")
    vds = vds.mapAnnotations((v, va, gs) => i6(va, toAdd6))
      .copy[GenotypeMatrixT](vaSignature = s6)

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
    val toAdd7Sig = TString
    val (s7, i7) = vds.insertVA(toAdd7Sig, "a", "c")
    vds = vds.mapAnnotations((v, va, gs) => i7(va, toAdd7))
      .copy[GenotypeMatrixT](vaSignature = s7)

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
    vds = vds.mapAnnotations((v, va, gs) => d2(va))
      .copy[GenotypeMatrixT](vaSignature = s8)
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
    vds = vds.mapAnnotations((v, va, gs) => d3(va))
      .copy[GenotypeMatrixT](vaSignature = s9)

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
      .copy[GenotypeMatrixT](vaSignature = s10)

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
    vds = vds.mapAnnotations((v, va, gs) => i8(va, toAdd8))
      .copy[GenotypeMatrixT](vaSignature = s11)

    assert(vds.vaSignature.schema == toAdd8Sig.schema)
    assert(vds.variantsAndAnnotations.collect()
      .forall { case (v, va) => va == "dummy" })
  }

  @Test def testAttributeOperations() {

    /*
      This test method performs a number of attribute annotation operations on a vds, and ensures that the signatures
      and annotations in the RDD elements are what is expected after each step.
    */

    var vds = hc.importVCF("src/test/resources/sample.vcf").cache()

    vds = vds.setVaAttributes("va.info.DP", Map("new_key" -> "new_value"))
    var infoAttr = vds.vaSignature.asInstanceOf[TStruct]
      .fieldOption(Parser.parseAnnotationRoot("va.info.DP", Annotation.VARIANT_HEAD))
      .map(_.attrs)
      .getOrElse(Map[String,String]())
    assert(infoAttr.getOrElse("new_key", "missing_value") == "new_value")

    vds = vds.setVaAttributes("va.info.DP", Map("new_key" -> "modified_value"))
    infoAttr = vds.vaSignature.asInstanceOf[TStruct]
      .fieldOption(Parser.parseAnnotationRoot("va.info.DP", Annotation.VARIANT_HEAD))
      .map(_.attrs)
      .getOrElse(Map[String,String]())
    assert(infoAttr.getOrElse("new_key", "missing_value") == "modified_value")

    vds = vds.setVaAttributes("va.info.DP", Map("key1" -> "value1", "key2" -> "value2"))
    infoAttr = vds.vaSignature.asInstanceOf[TStruct]
      .fieldOption(Parser.parseAnnotationRoot("va.info.DP", Annotation.VARIANT_HEAD))
      .map(_.attrs)
      .getOrElse(Map[String,String]())
    assert(infoAttr.getOrElse("key1", "missing_value") == "value1")
    assert(infoAttr.getOrElse("key2", "missing_value") == "value2")

    vds = vds.deleteVaAttribute("va.info.DP", "new_key")
    infoAttr = vds.vaSignature.asInstanceOf[TStruct]
      .fieldOption(Parser.parseAnnotationRoot("va.info.DP", Annotation.VARIANT_HEAD))
      .map(_.attrs)
      .getOrElse(Map[String,String]())
    assert(!infoAttr.contains("new_key"))

    vds = vds.deleteVaAttribute("va.info.DP", "new_key")

  }

  @Test def testWeirdNamesReadWrite() {
    var vds = hc.importVCF("src/test/resources/sample.vcf")
      .splitMulti()

    val f = tmpDir.createTempFile("testwrite", extension = ".vds")
    val (newS, ins) = vds.insertVA(TInt, "ThisName(won'twork)=====")
    vds = vds.mapAnnotations((v, va, gs) => ins(va, 5))
      .copy[GenotypeMatrixT](vaSignature = newS)
    vds.write(f)

    assert(hc.readVDS(f).same(vds))
  }
}
