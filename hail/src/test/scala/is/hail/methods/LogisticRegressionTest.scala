package is.hail.methods

import java.util

import is.hail.{SparkSuite, TestUtils}
import is.hail.annotations.Annotation
import is.hail.check.Prop._
import is.hail.check.{Gen, Properties}
import is.hail.expr.ir.{F64,I32, GetField, InsertFields, MatrixMapCols, Ref}
import is.hail.expr.ir.TestUtils.IRScanCollect
import is.hail.expr.types._
import is.hail.expr.types.virtual.{TFloat64, TInt32, TString}
import is.hail.io.vcf.ExportVCF
import is.hail.utils.AbsoluteFuzzyComparable._
import is.hail.utils.{AbsoluteFuzzyComparable, TextTableReader, _}
import is.hail.variant._
import org.apache.spark.sql.Row
import org.testng.annotations.Test
import is.hail.table.Table





class LogisticRegressionTest extends SparkSuite {

  @Test def testLogisticRegressionSinglePhenotype() {
    hc.indexBgen(FastIndexedSeq("src/test/resources/example.8bits.bgen"), rg = Some("GRCh37"), contigRecoding = Map("01" -> "1"))
    val mt: MatrixTable = TestUtils.importBgens(hc, FastIndexedSeq("src/test/resources/example.8bits.bgen"),
      includeDosage = true)//TestUtils.importVCF(hc, "src/test/resources/regressionLogistic.vcf")
    val covs = hc.importTable("src/test/resources/regressionLogisticMultiPheno.cov", impute = true).keyBy(Array("Sample"))
    val pt = hc.importTable("src/test/resources/regressionLogisticMultiPheno.pheno",types = Map("Sample" -> TString(),
      "Pheno1" -> TInt32(), "Pheno2" -> TInt32()), keyNames=Some(Array("Sample")))
    val mt2 = mt.annotateColsTable(pt,"p")
    val mt3 = mt2.annotateColsTable(covs,"c").cache()
    val oldCol = Ref("sa", mt3.colType)
    val newCol = InsertFields(oldCol, Seq("Pheno1" -> GetField(GetField(oldCol, "p"), "Pheno1").toD,
                                          "Cov1" -> GetField(GetField(oldCol, "c"), "Cov1").toD,
                                          "Cov2" -> GetField(GetField(oldCol, "c"), "Cov2").toD))
    val newMatrixIR = MatrixMapCols(mt3.ast, newCol, None)
    val newMatrix = mt3.copyAST(newMatrixIR)
    //val mt3 = mt2.annotateColsTable(covs, "1")
    var covFields = new util.ArrayList[String]()
    var passFields = new util.ArrayList[String]()
    covFields.add("Cov1")
    covFields.add("Cov2")
    val res = LogisticRegression(newMatrix, "wald", "Pheno1", "dosage", covFields, passFields)
    assert(res.count() > 0, "expecting more than zero results")
    assert(res.typ.rowType.fieldNames.contains("p_value"),"expecting result table to have p_value")
    assert(res.typ.rowType.fieldNames.contains("beta"), "expecting result table to have beta")
  }

  @Test def testLogisticRegressionMultiPhenotype() {
    hc.indexBgen(FastIndexedSeq("src/test/resources/example.8bits.bgen"), rg = Some("GRCh37"), contigRecoding = Map("01" -> "1"))
    val mt: MatrixTable = TestUtils.importBgens(hc, FastIndexedSeq("src/test/resources/example.8bits.bgen"),
      includeDosage = true)//TestUtils.importVCF(hc, "src/test/resources/regressionLogistic.vcf")
    val covs = hc.importTable("src/test/resources/regressionLogisticMultiPheno.cov", impute = true).keyBy(Array("Sample"))
    val pt = hc.importTable("src/test/resources/regressionLogisticMultiPheno.pheno",types = Map("Sample" -> TString(),
      "Pheno1" -> TInt32(), "Pheno2" -> TInt32()), keyNames=Some(Array("Sample")))
    val mt2 = mt.annotateColsTable(pt,"p")
    val mt3 = mt2.annotateColsTable(covs,"c").cache()
    val oldCol = Ref("sa", mt3.colType)
    val newCol = InsertFields(oldCol, Seq("Pheno1" -> GetField(GetField(oldCol, "p"), "Pheno1").toD,
      "Pheno2" -> GetField(GetField(oldCol, "p"), "Pheno2").toD,
      "Cov1" -> GetField(GetField(oldCol, "c"), "Cov1").toD,
      "Cov2" -> GetField(GetField(oldCol, "c"), "Cov2").toD))
    val newMatrixIR = MatrixMapCols(mt3.ast, newCol, None)
    val newMatrix = mt3.copyAST(newMatrixIR)
    //val mt3 = mt2.annotateColsTable(covs, "1")
    var covFields = new util.ArrayList[String]()
    var pheFields = new util.ArrayList[String]()
    var passFields = new util.ArrayList[String]()
    covFields.add("Cov1")
    covFields.add("Cov2")
    pheFields.add("Pheno1")
    pheFields.add("Pheno2")
    val res = LogisticRegression(newMatrix, "wald", pheFields, "dosage", covFields, passFields)
    assert(res.count() > 0, "expecting more than zero results")
    assert(res.typ.rowType.fieldNames.contains("logistic_regression"),"expecting result table to have logistic regression field")
  }
}
