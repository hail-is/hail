package is.hail.utils

import is.hail.HailSuite
import is.hail.expr.ir.TextTableReader
import is.hail.expr.types.virtual._
import org.testng.annotations.Test

class TextTableSuite extends HailSuite {

  @Test def testTypeGuessing() {

    val doubleStrings = Seq("1", ".1", "-1", "-.1", "1e1", "-1e1",
      "1E1", "-1E1", "1.0e2", "-1.0e2", "1e-1", "-1e-1", "-1.0e-2")
    val badDoubleStrings = Seq("1ee1", "1e--2", "1ee2", "1e0.1", "1e-0.1", "1e1.")
    val intStrings = Seq("1", "0", "-1", "12312398", "-123092398")
    val longStrings = Seq("11010101010101010", "-9223372036854775808")
    val booleanStrings = Seq("true", "True", "TRUE", "false", "False", "FALSE")

    doubleStrings.foreach(str => assert(TextTableReader.float64Matcher(str)))
    badDoubleStrings.foreach(str => assert(!TextTableReader.float64Matcher(str)))

    intStrings.foreach(str => assert(TextTableReader.int32Matcher(str)))
    intStrings.foreach(str => assert(TextTableReader.int64Matcher(str)))
    intStrings.foreach(str => assert(TextTableReader.float64Matcher(str)))

    longStrings.foreach(str => assert(TextTableReader.int64Matcher(str), str))
    longStrings.foreach(str => assert(!TextTableReader.int32Matcher(str)))

    booleanStrings.foreach(str => assert(TextTableReader.booleanMatcher(str)))

    val rdd = sc.parallelize(Seq(
      "123 123 . gene1 1:1:A:T true MT:123123",
      ". . . gene2 . . .",
      "-129 -129 . 1230192 1:1:A:AAA false GRCH12.1:151515",
      "0 0 . gene123.1 1:100:A:* false GRCH12.1:123",
      "-200 -200.0 . 155.2 GRCH123.2:2:A:T true 1:2"
    ), 3).map { x => WithContext(x, Context(x, "none", None)) }

    val imputed = TextTableReader.imputeTypes(rdd, Array("1", "2", "3", "4", "5", "6", "7"), "\\s+", Set("."), null)

    assert(imputed.sameElements(Array(
      Some(TInt32()),
      Some(TFloat64()),
      None,
      Some(TString()),
      Some(TString()),
      Some(TBoolean()),
      Some(TString())
    )))

    val schema = TextTableReader.read(hc)(Array("src/test/resources/variantAnnotations.tsv"),
      impute = true).signature
    assert(schema == TStruct(
      "Chromosome" -> TInt32(),
      "Position" -> TInt32(),
      "Ref" -> TString(),
      "Alt" -> TString(),
      "Rand1" -> TFloat64(),
      "Rand2" -> TFloat64(),
      "Gene" -> TString()))

    val schema2 = TextTableReader.read(hc)(Array("src/test/resources/variantAnnotations.tsv"),
      types = Map("Chromosome" -> TString()), impute = true).signature
    assert(schema2 == TStruct(
      "Chromosome" -> TString(),
      "Position" -> TInt32(),
      "Ref" -> TString(),
      "Alt" -> TString(),
      "Rand1" -> TFloat64(),
      "Rand2" -> TFloat64(),
      "Gene" -> TString()))

    val schema3 = TextTableReader.read(hc)(Array("src/test/resources/variantAnnotations.alternateformat.tsv"),
      impute = true).signature
    assert(schema3 == TStruct(
      "Chromosome:Position:Ref:Alt" -> TString(),
      "Rand1" -> TFloat64(),
      "Rand2" -> TFloat64(),
      "Gene" -> TString()))

    val schema4 = TextTableReader.read(hc)(Array("src/test/resources/sampleAnnotations.tsv"),
      impute = true).signature
    assert(schema4 == TStruct(
      "Sample" -> TString(),
      "Status" -> TString(),
      "qPhen" -> TInt32()))

    val schema5 = TextTableReader.read(hc)(Array("src/test/resources/integer_imputation.txt"),
      impute = true, separator = "\\s+").signature
    assert(schema5 == TStruct(
      "A" -> TInt64(),
      "B" -> TInt32()))

  }

  @Test def testPipeDelimiter() {
    assert(TextTableReader.splitLine("a|b", "|", '#').toSeq == Seq("a", "b"))
  }
}
