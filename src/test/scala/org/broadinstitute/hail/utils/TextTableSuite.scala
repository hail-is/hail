package org.broadinstitute.hail.utils

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.check._
import org.broadinstitute.hail.driver.{AnnotateVariantsExpr, AnnotateVariantsTable, ExportVariants, State}
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant.{VSMSubgen, VariantDataset, VariantSampleMatrix}
import org.testng.annotations.Test

class TextTableSuite extends SparkSuite {

  @Test def testTypeGuessing() {

    val doubleStrings = Seq("1", ".1", "-1", "-.1", "1e1", "-1e1",
      "1E1", "-1E1", "1.0e2", "-1.0e2", "1e-1", "-1e-1", "-1.0e-2")
    val badDoubleStrings = Seq("1ee1", "1e--2", "1ee2", "1e0.1", "1e-0.1", "1e1.")
    val intStrings = Seq("1", "0", "-1", "12312398", "-123098172398")
    val booleanStrings = Seq("true", "True", "TRUE", "false", "False", "FALSE")
    val variantStrings = Seq("1:1:A:T", "MT:12309123:A:*", "22:1201092:ATTTAC:T,TACC,*")
    val badVariantStrings = Seq("1:X:A:T", "1:1:*:T", "1:1:A", "1:1:A:T,", "1:1:AAAT:*A")

    doubleStrings.foreach(str => assert(str matches TextTableReader.doubleRegex))
    intStrings.foreach(str => assert(str matches TextTableReader.doubleRegex))
    badDoubleStrings.foreach(str => assert(!(str matches TextTableReader.doubleRegex)))

    intStrings.foreach(str => assert(str matches TextTableReader.intRegex))

    booleanStrings.foreach(str => assert(str matches TextTableReader.booleanRegex))

    variantStrings.foreach(str => assert(str matches TextTableReader.variantRegex))
    badVariantStrings.foreach(str => assert(!(str matches TextTableReader.variantRegex)))

    assert(TextTableReader.guessType(Seq("123", ".", "-129", "0", "-200"), ".") == Some(TInt))
    assert(TextTableReader.guessType(Seq("123", ".", "-129", "0", "-200", "-100.0"), ".") == Some(TDouble))
    assert(TextTableReader.guessType(Seq("."), ".").isEmpty)
    assert(TextTableReader.guessType(Seq("gene1", "gene2", "1230192"), ".") == Some(TString))
    assert(TextTableReader.guessType(Seq("1:1:A:T", ".", "1:1:A:AAA"), ".") == Some(TVariant))
    assert(TextTableReader.guessType(Seq("true", ".", "false"), ".") == Some(TBoolean))

    val (schema, _) = TextTableReader.read(sc, Array("src/test/resources/variantAnnotations.tsv"),
      config = TextTableConfiguration().copy(impute = true))
    assert(schema == TStruct(
      "Chromosome" -> TInt,
      "Position" -> TInt,
      "Ref" -> TString,
      "Alt" -> TString,
      "Rand1" -> TDouble,
      "Rand2" -> TDouble,
      "Gene" -> TString))

    val (schema2, _) = TextTableReader.read(sc, Array("src/test/resources/variantAnnotations.tsv"),
      config = TextTableConfiguration().copy(types = Map("Chromosome" -> TString), impute = true))
    assert(schema2 == TStruct(
      "Chromosome" -> TString,
      "Position" -> TInt,
      "Ref" -> TString,
      "Alt" -> TString,
      "Rand1" -> TDouble,
      "Rand2" -> TDouble,
      "Gene" -> TString))

    val (schema3, _) = TextTableReader.read(sc, Array("src/test/resources/variantAnnotations.alternateformat.tsv"),
      config = TextTableConfiguration().copy(impute = true))
    assert(schema3 == TStruct(
      "Chromosome:Position:Ref:Alt" -> TVariant,
      "Rand1" -> TDouble,
      "Rand2" -> TDouble,
      "Gene" -> TString))

    val (schema4, _) = TextTableReader.read(sc, Array("src/test/resources/sampleAnnotations.tsv"),
      config = TextTableConfiguration().copy(impute = true))
    assert(schema4 == TStruct(
      "Sample" -> TString,
      "Status" -> TString,
      "qPhen" -> TInt))
  }

  @Test def testAnnotationsReadWrite() {
    val outPath = tmpDir.createTempFile("annotationOut", ".tsv")
    val outTypesPath = tmpDir.createTempFile("annotationOut", ".types")
    val p = Prop.forAll(VariantSampleMatrix.gen(sc, VSMSubgen.realistic)
      .filter(vds => vds.nVariants > 0 && vds.vaSignature != TDouble)) { vds: VariantDataset =>

      var state = State(sc, sqlContext, vds)
      state = ExportVariants.run(state, Array("-o", outPath, "-c", "v = v, va = va", "-t", outTypesPath))

      state = AnnotateVariantsTable.run(state, Array(outPath,
        "-e", "v",
        "-c", "va = table.va",
        "-t", s"@$outTypesPath"))

      state.vds.same(vds)
    }

    p.check()
  }
}
