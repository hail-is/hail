package is.hail.rest

import _root_.is.hail.variant.VariantDataset
import breeze.linalg._
import com.jayway.restassured.RestAssured._
import com.jayway.restassured.config.JsonConfig
import com.jayway.restassured.path.json.config.JsonPathConfig.NumberReturnType
import is.hail.{HailContext, SparkSuite}
import org.hamcrest.Matchers._
import org.http4s.server.Server
import org.http4s.server.blaze.BlazeBuilder
import org.testng.annotations.Test

class RestServiceScoreCovarianceRunnable(hc: HailContext) extends Runnable {
  var task: Server = _

  override def run() {
    val sampleKT = hc.importTable("src/test/resources/restService.cov", impute = true).keyBy("IID")
    
    val vds: VariantDataset = hc.importVCF("src/test/resources/restService.vcf", nPartitions = Some(2))
      .annotateSamplesTable(sampleKT, root="sa.rest")
   
    val covariates = sampleKT.fieldNames.filterNot(_ == "IID").map("sa.rest." + _)
    
    val restService = new RestServiceScoreCovariance(vds, covariates, useDosages = false, maxWidth = 1000000, hardLimit = 100000)

    task = BlazeBuilder.bindHttp(8080)
      .mountService(restService.service, "/")
      .run

    task
      .awaitShutdown()
  }
}

class RestServiceScoreCovarianceSuite extends SparkSuite {
  
  // run to test server
  def localRestServerTest() {
    val sampleKT = hc.importTable("src/test/resources/restService.cov", impute = true).keyBy("IID")
    val vds: VariantDataset = hc.importVCF("src/test/resources/restService.vcf", nPartitions = Some(2))
      .annotateSamplesTable(sampleKT, root="sa.rest")
    val covariates = sampleKT.fieldNames.filterNot(_ == "IID").map("sa.rest." + _)
    
    vds.restServerScoreCovariance(covariates, port=6060)
  }

  @Test def test() {
    
    val r = new RestServiceScoreCovarianceRunnable(hc)
    val t = new Thread(r)
    t.start()

    Thread.sleep(4000) // Hack to give the server time to initialize
    
    /*
    Sample code for generating p-values in R (below, missing genotypes are imputed using all samples; use subset when using BMI or HEIGHT):
    df = read.table("t2dserverR.tsv", header = TRUE)
    fit <- lm(T2D ~ v1 + SEX, data=df)
    summary(fit)["coefficients"]

    Contents of t2dserverR.tsv (change spaces to tabs, and re-impute missing values based on subset):
    IID v1  v2  v3  v4  v5  v6  v7  v8  v9  v10 T2D SEX PC1 BMI HEIGHT
    A   0   .   0   0   0   0   1   2   1   2   1   0   -1  20  5.4
    B   1   2   .   0   1   0   1   2   1   2   1   2   3   25  5.6
    C   0   .   1   0   1   0   1   2   1   2   2   1   5   NA  6.3
    D   0   2   1   1   2   0   1   2   1   2   2   -2  0   30  NA
    E   0   0   1   1   0   0   1   2   1   2   2   -2  -4  22  6.0
    F   1   0   .   1   0   0   1   2   1   2   2   4   3   19  5.8
    */

    /*
    // Truth data for covariateBMI test
    //
    // yRes and sigma from R, then score = G.t * yRes
    // y = c(1, 1, 2, 2, 2)
    // bmi = c(20, 25, 30, 22, 19)
    // fit = lm(y ~ x)
    // summary(fit)
    
    // covariance from G: rows are samples, columns are variants. Sample C is incomplete for BMI.
    val G = DenseMatrix((0.0, 1.0,  0.0, 0.0, 0.0),
                        (1.0, 2.0, 2/3d, 0.0, 1.0),
                        (0.0, 2.0,  1.0, 1.0, 2.0),
                        (0.0, 0.0,  1.0, 1.0, 0.0),
                        (1.0, 0.0, 2/3d, 1.0, 0.0))
    
    val colMeans = (1.0 / G.rows) * DenseVector(sum(G(::, *)).toArray)
    val centeredG = G(*, ::).map(_ - colMeans)
    val covariance = centeredG.t * centeredG
    */
        
    var response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "CovariateBMI",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "sa.rest.BMI"}
            |                      ],
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "gte", "value": 1, "operand_type": "integer"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("covariance[0]", closeTo(1.2, 1e-5))
        .body("covariance[1]", closeTo(0.0, 1e-5))
        .body("covariance[2]", closeTo(0.0, 1e-5))
        .body("covariance[3]", closeTo(-0.2, 1e-5))
        .body("covariance[4]", closeTo(-0.2, 1e-5))
        .body("covariance[5]", closeTo(4.0, 1e-5))
        .body("covariance[6]", closeTo(0.0, 1e-5))
        .body("covariance[7]", closeTo(-1.0, 1e-5))
        .body("covariance[8]", closeTo(3.0, 1e-5))
        .body("covariance[9]", closeTo(2/3d, 1e-5))
        .body("covariance[10]", closeTo(2/3d, 1e-5))
        .body("covariance[11]", closeTo(2/3d, 1e-5))
        .body("covariance[12]", closeTo(1.2, 1e-5))
        .body("covariance[13]", closeTo(0.2, 1e-5))
        .body("covariance[14]", closeTo(3.2, 1e-5))
        .body("scores[0]", closeTo(-0.157360, 1e-5))
        .body("scores[1]", closeTo(-1.24873, 1e-5))
        .body("scores[2]", closeTo(0.595601, 1e-5))
        .body("scores[3]", closeTo(1.17513, 1e-5))
        .body("scores[4]", closeTo(-0.0736041, 1e-5))
        .body("sigma_sq", closeTo(0.391708, 1e-5))
        .body("nsamples", is(5))
        .body("count", is(5))
        .extract()
        .response()

    // println(response.asString())
    
    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "NoCovariates",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "gte", "value": 1, "operand_type": "integer"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("passback", is("NoCovariates"))
        .body("active_variants[0].chrom", is("1"))
        .body("active_variants[0].pos", is(1))
        .body("active_variants[0].ref", is("C"))
        .body("active_variants[0].alt", is("T"))
        .body("active_variants[4].chrom", is("1"))
        .body("active_variants[4].pos", is(350000))
        .body("active_variants[4].ref", is("C"))
        .body("active_variants[4].alt", is("A"))
        .body("sigma_sq", closeTo(0.266666, 1e-5))
        .body("nsamples", is(6))
        .body("count", is(5))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "macBoundsFilterAll",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "sa.rest.HEIGHT"}
            |                      ],
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"},
            |                        {"operand": "mac", "operator": "lt", "value": 2, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("scores[0]", is(nullValue()))        
        .body("nsamples", is(5))
        .body("count", is(0)) // all AC are 2
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "macBoundsFilterSome",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"},
            |                        {"operand": "mac", "operator": "lt", "value": 4, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))      
        .body("nsamples", is(6))
        .body("count", is(3)) // all AC are 2
        .extract()
        .response()

    /*
    // Truth data for variantList test
    //
    // yRes and sigma from R, then score = G.t * yRes
    // y = c(1, 1, 2, 2, 2)
    // bmi = c(20, 25, 30, 22, 19)
    // v1 = c(0, 1, 0, 0, 1)
    // fit = lm(y ~ bmi + v1)
    // summary(fit)
    
    // covariance from G: rows are samples, columns are variants.
    // sample C is incomplete for BMI. Using variant_list to filter to 2nd and 4th variants.
    val G = DenseMatrix((1.0, 0.0),
                        (2.0, 0.0),
                        (2.0, 1.0),
                        (0.0, 1.0),
                        (0.0, 1.0))
    
    val colMeans = (1.0 / G.rows) * DenseVector(sum(G(::, *)).toArray)
    val centeredG = G(*, ::).map(_ - colMeans)
    val covariance = centeredG.t * centeredG
    */
    
    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "variantList",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "sa.rest.BMI"},
            |                        {"type": "variant", "chrom": "1", "pos": 1, "ref": "C", "alt": "T"}
            |                      ],
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ],
            |  "variant_list"    : [
            |                        {"chrom": "1", "pos": 2, "ref": "C", "alt": "T"},
            |                        {"chrom": "1", "pos": 250000, "ref": "A", "alt": "T"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))    
        .body("covariance[0]", closeTo(4.0, 1e-5))
        .body("covariance[1]", closeTo(-1.0, 1e-5))
        .body("covariance[2]", closeTo(1.2, 1e-5))
        .body("scores[0]", closeTo(-1.18918, 1e-5))
        .body("scores[1]", closeTo(1.15315, 1e-5))
        .body("sigma_sq", closeTo(0.576576, 1e-5))
        .body("nsamples", is(5))
        .body("count", is(2))
        .extract()
        .response()
  }
  
  @Test def lowerTriangleTest() {
    val X = DenseMatrix((1.0, 0.0, 0.0),
                        (2.0, 4.0, 0.0),
                        (3.0, 5.0, 6.0))
    
    assert(RestService.lowerTriangle(X.toArray, 3) sameElements Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
  }
}
