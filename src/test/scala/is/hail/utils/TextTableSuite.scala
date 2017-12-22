package is.hail.utils

import is.hail.SparkSuite
import is.hail.check._
import is.hail.expr._
import is.hail.expr.typ.{TLocus, TStruct, TVariant, Type}
import is.hail.variant.{GenomeReference, MatrixTable, VSMSubgen}
import org.testng.annotations.Test

import scala.io.Source

class TextTableSuite extends SparkSuite {

  @Test def testTypeGuessing() {

    val doubleStrings = Seq("1", ".1", "-1", "-.1", "1e1", "-1e1",
      "1E1", "-1E1", "1.0e2", "-1.0e2", "1e-1", "-1e-1", "-1.0e-2")
    val badDoubleStrings = Seq("1ee1", "1e--2", "1ee2", "1e0.1", "1e-0.1", "1e1.")
    val intStrings = Seq("1", "0", "-1", "12312398", "-123098172398")
    val booleanStrings = Seq("true", "True", "TRUE", "false", "False", "FALSE")
    val variantStrings = Seq("1:1:A:T", "MT:12309123:A:*", "22:1201092:ATTTAC:T,TACC,*")
    val locusStrings = Seq("MT:123123", "1:1", "GRCH12.1:151515", ".")
    val badVariantStrings = Seq("1:X:A:T", "1:1:*:T", "1:1:A", "1:1:A:T,", "1:1:AAAT:*A")

    doubleStrings.foreach(str => assert(str matches TextTableReader.doubleRegex))
    intStrings.foreach(str => assert(str matches TextTableReader.doubleRegex))
    badDoubleStrings.foreach(str => assert(!(str matches TextTableReader.doubleRegex)))

    intStrings.foreach(str => assert(str matches TextTableReader.intRegex))

    booleanStrings.foreach(str => assert(str matches TextTableReader.booleanRegex))

    variantStrings.foreach(str => assert(str matches TextTableReader.variantRegex))
    badVariantStrings.foreach(str => assert(!(str matches TextTableReader.variantRegex)))

    val rdd = sc.parallelize(Seq(
      "123 123 . gene1 1:1:A:T true MT:123123",
      ". . . gene2 . . .",
      "-129 -129 . 1230192 1:1:A:AAA false GRCH12.1:151515",
      "0 0 . gene123.1 1:100:A:* false GRCH12.1:123",
      "-200 -200.0 . 155.2 GRCH123.2:2:A:T true 1:2"
    ), 3).map { x => WithContext(x, Context(x, "none", None)) }

    val imputed = TextTableReader.imputeTypes(rdd, Array("1", "2", "3", "4", "5", "6", "7"), "\\s+", ".", null)

    assert(imputed.sameElements(Array(
      Some(TInt32()),
      Some(TFloat64()),
      None,
      Some(TString()),
      Some(TVariant(GenomeReference.GRCh37)),
      Some(TBoolean()),
      Some(TLocus(GenomeReference.GRCh37))
    )))

    val (schema, _) = TextTableReader.read(sc)(Array("src/test/resources/variantAnnotations.tsv"),
      impute = true)
    assert(schema == TStruct(
      "Chromosome" -> TInt32(),
      "Position" -> TInt32(),
      "Ref" -> TString(),
      "Alt" -> TString(),
      "Rand1" -> TFloat64(),
      "Rand2" -> TFloat64(),
      "Gene" -> TString()))

    val (schema2, _) = TextTableReader.read(sc)(Array("src/test/resources/variantAnnotations.tsv"),
      types = Map("Chromosome" -> TString()), impute = true)
    assert(schema2 == TStruct(
      "Chromosome" -> TString(),
      "Position" -> TInt32(),
      "Ref" -> TString(),
      "Alt" -> TString(),
      "Rand1" -> TFloat64(),
      "Rand2" -> TFloat64(),
      "Gene" -> TString()))

    val (schema3, _) = TextTableReader.read(sc)(Array("src/test/resources/variantAnnotations.alternateformat.tsv"),
      impute = true)
    assert(schema3 == TStruct(
      "Chromosome:Position:Ref:Alt" -> TVariant(GenomeReference.GRCh37),
      "Rand1" -> TFloat64(),
      "Rand2" -> TFloat64(),
      "Gene" -> TString()))

    val (schema4, _) = TextTableReader.read(sc)(Array("src/test/resources/sampleAnnotations.tsv"),
      impute = true)
    assert(schema4 == TStruct(
      "Sample" -> TString(),
      "Status" -> TString(),
      "qPhen" -> TInt32()))
  }

  @Test def testAnnotationsReadWrite() {
    val outPath = tmpDir.createTempFile("annotationOut", ".tsv")
    val p = Prop.forAll(MatrixTable.gen(hc, VSMSubgen.realistic)
      .filter(vds => vds.countVariants > 0 && !vds.vaSignature.isOfType(TFloat64()))) { vds: MatrixTable =>

      vds.variantsKT().export(outPath, typesFile = outPath + ".types")

      val types = Type.parseMap(hadoopConf.readFile(outPath + ".types")(Source.fromInputStream(_).mkString))

      val kt = hc.importTable(outPath, types = types).keyBy("v")
      vds.annotateVariantsTable(kt, root = "va").same(vds)
    }

    p.check()
  }

  @Test def testPipeDelimiter() {
    assert(TextTableReader.splitLine("a|b", "|", '#').toSeq == Seq("a", "b"))
  }
}
