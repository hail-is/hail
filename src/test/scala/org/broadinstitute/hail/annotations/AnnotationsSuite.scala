package org.broadinstitute.hail.annotations

import org.apache.spark.sql.Row
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.expr
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.variant.{Genotype, IntervalList, Variant}
import org.testng.annotations.Test
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.variant.RichRow._
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
    val vas = vds.metadata.variantAnnotationSignatures
    val variantAnnotationMap = vds.variantsAndAnnotations.collect().toMap

    val firstVariant = Variant("20", 10019093, "A", "G")
    val anotherVariant = Variant("20", 10036107, "T", "G")
    assert(variantAnnotationMap.contains(firstVariant))
    assert(variantAnnotationMap.contains(anotherVariant))

    // type Int - info.DP
    val dpQuery = vas.query(List("info", "DP"))
    assert(vas.getOption(List("info", "DP")).contains(VCFSignature(expr.TInt, "Integer", "1",
      "Approximate read depth; some reads may have been filtered")))
    assert(dpQuery(variantAnnotationMap(firstVariant))
      .get == 77560)
    assert(dpQuery(variantAnnotationMap(anotherVariant))
      .get == 20271)

    // type Double - info.HWP
    val hwpQuery = vas.query(List("info", "HWP"))
    assert(vas.getOption(List("info", "HWP")).contains(new VCFSignature(expr.TDouble, "Float", "1",
      "P value from test of Hardy Weinberg Equilibrium")))
    assert(
      D_==(hwpQuery(variantAnnotationMap(firstVariant))
        .get.asInstanceOf[Double], 0.0001))
    assert(D_==(hwpQuery(variantAnnotationMap(anotherVariant))
      .get.asInstanceOf[Double], 0.8286))

    // type String - info.culprit
    val culpritQuery = vas.query(List("info", "culprit"))
    assert(vas.getOption(List("info", "culprit")).contains(VCFSignature(expr.TString, "String", "1",
      "The annotation which was the worst performing in the Gaussian mixture model, " +
        "likely the reason why the variant was filtered out")))
    assert(culpritQuery(variantAnnotationMap(firstVariant))
      .contains("FS"))
    assert(culpritQuery(variantAnnotationMap(anotherVariant))
      .contains("FS"))

    // type Array - info.AC (allele count)
    val acQuery = vas.query(List("info", "AC"))
    assert(vas.getOption(List("info", "AC")).contains(VCFSignature(expr.TArray(expr.TInt), "Integer", "A",
      "Allele count in genotypes, for each ALT allele, in the same order as listed")))
    assert(acQuery(variantAnnotationMap(firstVariant))
      .contains(Array(89): mutable.WrappedArray[Int]))
    assert(acQuery(variantAnnotationMap(anotherVariant))
      .contains(Array(13): mutable.WrappedArray[Int]))

    // type Boolean/flag - info.DB (dbSNP membership)
    val dbQuery = vas.query(List("info", "DB"))
    assert(vas.getOption(List("info", "DB")).contains(new VCFSignature(expr.TBoolean, "Flag", "0",
      "dbSNP Membership")))
    assert(dbQuery(variantAnnotationMap(firstVariant))
      .contains(true))
    assert(dbQuery(variantAnnotationMap(anotherVariant))
      .isEmpty)

    //type Set[String]
    val filtQuery = vas.query(List("filters"))
    assert(vas.getOption(List("filters")).contains(new SimpleSignature(expr.TSet(expr.TString))))
    assert(filtQuery(variantAnnotationMap(firstVariant))
        .contains(Array("PASS"): mutable.WrappedArray[String]))
    assert(filtQuery(variantAnnotationMap(anotherVariant))
      contains(Array("VQSRTrancheSNP99.95to100.00"): mutable.WrappedArray[String]))

    // GATK PASS
    val passQuery = vas.query(List("pass"))
    assert(vas.getOption(List("pass")).contains(new SimpleSignature(expr.TBoolean)))
    assert(passQuery(variantAnnotationMap(firstVariant))
      .contains(true))
    assert(passQuery(variantAnnotationMap(anotherVariant))
      .contains(false))

    val vds2 = LoadVCF(sc, "src/test/resources/sample2.vcf")
    val vas2 = vds2.metadata.variantAnnotationSignatures

    // Check that VDS can be written to disk and retrieved while staying the same
    hadoopDelete("/tmp/sample.vds", sc.hadoopConfiguration, recursive = true)
    vds2.write(sqlContext, "/tmp/sample.vds")
    val readBack = Read.run(state, Array("-i", "/tmp/sample.vds"))

    assert(readBack.vds.same(vds2))
  }

  @Test def testReadWrite() {
    val vds1 = LoadVCF(sc, "src/test/resources/sample.vcf")
    val s = State(sc, sqlContext, vds1)
    val vds2 = LoadVCF(sc, "src/test/resources/sample.vcf")
    assert(vds1.same(vds2))
    Write.run(s, Array("-o", "/tmp/sample.vds"))
    val vds3 = Read.run(s, Array("-i", "/tmp/sample.vds")).vds
    println(vds1.metadata == vds3.metadata)
    assert(vds3.same(vds1))
  }

  @Test def testRewrite() {


    val inner = Row.fromSeq(Array(4, 5, 6))
    val middle = Row.fromSeq(Array(2, 3, inner))
    val outer = Row.fromSeq(Array(1, middle))

    val inner2 = Row.fromSeq(Array(100))
    val middle2 = Row.fromSeq(Array(inner2))
    val outer2 = Row.fromSeq(Array(middle2, 55))

    val outer3 = Row.fromSeq(Array(1000))

    val ad1 = outer
    val ad2 = outer2
    val ad3 = outer3

    println("ad1:")
    Annotations.printRow(ad1)
    println("ad2:")
    Annotations.printRow(ad2)
    println("ad3:")
    Annotations.printRow(ad3)


    val innerSigs: StructSignature = StructSignature(Map(
      "d" ->(0, SimpleSignature(expr.TInt)),
      "e" ->(1, SimpleSignature(expr.TInt)),
      "f" ->(2, SimpleSignature(expr.TInt))))

    val middleSigs = StructSignature(Map(
      "b" ->(0, SimpleSignature(expr.TInt)),
      "c" ->(1, SimpleSignature(expr.TInt)),
      "inner" ->(2, innerSigs)))

    val outerSigs = StructSignature(Map(
      "a" ->(0, SimpleSignature(expr.TInt)),
      "middle" ->(1, middleSigs)))

    println("here")
    val signatures = outerSigs
    //    val sigsToAdd = StructSignature(Map(
    //      "middle" -> StructSignature(Map(
    //        "inner" -> StructSignature(Map(
    //          "g" -> SimpleSignature(expr.TInt, 0)
    //        ), 0)
    //      ), 0),
    //      "anotherthing" -> SimpleSignature(expr.TInt, 1))
    //    )
    val sigToAdd = SimpleSignature(expr.TInt)

    val sigsToAdd2 = StructSignature(Map(
      "middle" ->(0, SimpleSignature(expr.TInt))))

    println("sigs before:")
    println(signatures.printSchema("va"))

    val (newSigs, f) = signatures.insert(List("middle", "inner", "g"), sigToAdd)
    println("sigs after:")
    println(newSigs.printSchema("va"))

    val (newSigs2, f2) = newSigs.delete(List("middle", "inner", "g"))
    println("removed g:")
    println(newSigs2.printSchema("va"))

    val (newSigs3, f3) = newSigs2.delete(List("middle", "inner"))
    println("removed inner:")
    println(newSigs3.printSchema("va"))

    val (newSigs4, f4) = newSigs2.delete(List("a"))
    println("removed a:")
    println(newSigs4.printSchema("va"))

    val vds = LoadVCF(sc, file1 = "src/test/resources/sample.vcf")
    val first = vds.variantsAndAnnotations.take(1).head
    Annotations.printRow(first._2.asInstanceOf[Row])
  }
}
