package is.hail.methods

import is.hail.{HailContext, SparkSuite, TestUtils}
import is.hail.expr.ir.{GetField, InsertFields, MatrixMapCols, Ref, TableLiteral}
import is.hail.expr.types.virtual.{TInt32, TString}
import is.hail.variant._
import is.hail.utils._
import org.testng.annotations.Test
import is.hail.table.Table

class LogisticRegressionTest extends SparkSuite {

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
    var covFields = Array[String]("Cov1", "Cov2")
    var pheFields = Array[String]("Pheno1", "Pheno2")
    var passFields = Array[String]()
    val res = new Table(HailContext.get, TableLiteral(LogisticRegression("wald", pheFields, "dosage", covFields, passFields)
      .execute(newMatrix.value)))
    assert(res.count() > 0, "expecting more than zero results")
    assert(res.typ.rowType.fieldNames.contains("logistic_regression"),"expecting result table to have logistic regression field")
  }
}
