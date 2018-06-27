package is.hail.utils

import is.hail.SparkSuite
import is.hail.check._
import is.hail.expr.ir.{Interpret, Pretty, TableImport}
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.variant.{MatrixTable, ReferenceGenome$, VSMSubgen}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

import scala.io.Source

class TextTableSuite extends SparkSuite {

  @Test def testTypeGuessing() {

    val doubleStrings = Seq("1", ".1", "-1", "-.1", "1e1", "-1e1",
      "1E1", "-1E1", "1.0e2", "-1.0e2", "1e-1", "-1e-1", "-1.0e-2")
    val badDoubleStrings = Seq("1ee1", "1e--2", "1ee2", "1e0.1", "1e-0.1", "1e1.")
    val intStrings = Seq("1", "0", "-1", "12312398", "-123098172398")
    val booleanStrings = Seq("true", "True", "TRUE", "false", "False", "FALSE")

    doubleStrings.foreach(str => assert(str matches TextTableReader.doubleRegex))
    intStrings.foreach(str => assert(str matches TextTableReader.doubleRegex))
    badDoubleStrings.foreach(str => assert(!(str matches TextTableReader.doubleRegex)))

    intStrings.foreach(str => assert(str matches TextTableReader.intRegex))

    booleanStrings.foreach(str => assert(str matches TextTableReader.booleanRegex))

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
  }

  @Test def testPipeDelimiter() {
    assert(TextTableReader.splitLine("a|b", "|", '#').toSeq == Seq("a", "b"))
  }
  
  @Test def testUseColsParameter() {
    val file = "src/test/resources/variantAnnotations.tsv"
    val tbl = TextTableReader.read(hc)(Array(file), impute = true)
    val tir = tbl.tir.asInstanceOf[TableImport]
    
    val selectOneCol = tbl.select("{Gene: row.Gene}", None, None)
    val irSelectOneCol = new Table(hc, TableImport(
      tir.paths,
      selectOneCol.typ,
      tir.readerOpts.copy(useColIndices = Array(6))
    ))
    assert(selectOneCol.same(irSelectOneCol))

    val selectTwoCols = tbl.select("{Chromosome: row.Chromosome, Rand2: row.Rand2}", None, None)
    val irSelectTwoCols = new Table(hc, TableImport(
      tir.paths,
      selectTwoCols.typ,
      tir.readerOpts.copy(useColIndices = Array(0, 5))
    ))
    assert(selectTwoCols.same(irSelectTwoCols))
  }
}
