package is.hail.rest

import org.http4s.server.Server
import org.http4s.server.blaze.BlazeBuilder
import org.testng.annotations.Test
import com.jayway.restassured.RestAssured._
import com.jayway.restassured.config.JsonConfig
import com.jayway.restassured.path.json.config.JsonPathConfig.NumberReturnType
import is.hail.{HailContext, SparkSuite}
import org.hamcrest.Matchers._
import _root_.is.hail.variant.VariantDataset
import org.hamcrest.core.AnyOf

class RestServiceLinregRunnable(hc: HailContext) extends Runnable {
  var task: Server = _

  override def run() {
    val sampleKT = hc.importTable("src/test/resources/restServerLinreg.cov", impute = true).keyBy("IID")
    
    val vds: VariantDataset = hc.importVCF("src/test/resources/restServerLinreg.vcf", nPartitions = Some(2))
      .annotateSamplesTable(sampleKT, root="sa.rest")
   
    val covariates = sampleKT.fieldNames.filterNot(_ == "IID").map("sa.rest." + _)
    
    val restService = new RestServiceLinreg(vds, covariates, useDosages = false, maxWidth = 1000000, hardLimit = 100000)

    task = BlazeBuilder.bindHttp(8080)
      .mountService(restService.service, "/")
      .run

    task
      .awaitShutdown()
  }
}

class RestServiceLinregSuite extends SparkSuite {
  
  // run to test server
  def localRestServerTest() {
    val sampleKT = hc.importTable("src/test/resources/restServerLinreg.cov", impute = true).keyBy("IID")
    val vds: VariantDataset = hc.importVCF("src/test/resources/restServerLinreg.vcf", nPartitions = Some(2))
      .annotateSamplesTable(sampleKT, root="sa.rest")
    val covariates = sampleKT.fieldNames.filterNot(_ == "IID").map("sa.rest." + _)
    
    vds.restServerLinreg(covariates, port=6060)
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

    Contents of t2dserverR.tsv (change spaces to tabs):
    IID v1  v2  v3  v4  v5  v6  v7  v8  v9  v10 T2D SEX PC1 BMI HEIGHT
    A   0   1   0   0   0   0   1   2   1   2   1   0   -1  20  5.4
    B   1   2   .75 0   1   0   1   2   1   2   1   2   3   25  5.6
    C   0   1   1   0   1   0   1   2   1   2   2   1   5   NA  6.3
    D   0   2   1   1   2   0   1   2   1   2   2   -2  0   30  NA
    E   0   0   1   1   0   0   1   2   1   2   2   -2  -4  22  6.0
    F   1   0   .75 1   0   0   1   2   1   2   2   4   3   19  5.8
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
        .body("stats[0].chrom", is("1"))
        .body("stats[0].pos", is(1))
        .body("stats[0].p-value", closeTo(0.8632555, 1e-5))
        .body("stats[1].p-value", closeTo(0.06340577, 1e-5))
        .body("stats[2].p-value", closeTo(0.1930786, 1e-5))
        .body("stats[3].p-value", closeTo(0, 1e-5)) // perfect fit, SE approx 0
        .body("stats[4].p-value", closeTo(0.8443759, 1e-5))
        .body("nsamples", is(5))
        .extract()
        .response()

    println(response.asString())

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "CovariateHeight",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "sa.rest.HEIGHT"}
            |                      ],
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].chrom", is("1"))
        .body("stats[0].pos", is(1))
        .body("stats[0].p-value", closeTo(0.8162976, 1e-5))
        .body("stats[1].p-value", closeTo(0.11942193, 1e-5))
        .body("stats[2].p-value", closeTo(0.9536821, 1e-5))
        .body("stats[3].p-value", closeTo(0.08263506, 1e-5))
        .body("stats[4].p-value", closeTo(0.12565524, 1e-5))
        .body("nsamples", is(5))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "PhenotypeBMIAndCovariateHEIGHT",
            |  "api_version"     : 1,
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "sa.rest.HEIGHT"}
            |                      ],
            |  "phenotype"       : "sa.rest.BMI",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].chrom", is("1"))
        .body("stats[0].pos", is(1))
        .body("stats[0].p-value", closeTo(0.8599513, 1e-5))
        .body("stats[1].p-value", closeTo(0.1844617, 1e-5))
        .body("stats[2].p-value", closeTo(0.1400487, 1e-5))
        .body("stats[3].p-value", closeTo(0.1400487, 1e-5))
        .body("stats[4].p-value", closeTo(0.2677205, 1e-5))
        .body("nsamples", is(4))
        .extract()
        .response()
    
    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "CovariateHEIGHTandVariantCovariate",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "sa.rest.HEIGHT"},
            |                        {"type": "variant", "chrom": "1", "pos": 1, "ref": "C", "alt": "T"}
            |                      ],
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].chrom", is("1"))
        .body("stats[0].pos", is(1))
        .body("stats[0].p-value", anyOf(closeTo(1.0, 1e-3), is(nullValue)): AnyOf[java.lang.Double])
        .body("stats[1].p-value", closeTo(0.06066189, 1e-5))
        .body("stats[2].p-value", closeTo(0.9478751, 1e-5))
        .body("stats[3].p-value", closeTo(0.2634229, 1e-5))
        .body("stats[4].p-value", closeTo(0.06779419, 1e-5))
        .body("nsamples", is(5))
        .extract()
        .response()
    
    println(response.asString())

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "noCovariatesChrom1",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].chrom", is("1"))
        .body("stats[0].pos", is(1))
        .body("stats[0].p-value", closeTo(0.63281250, 1e-5))
        .body("stats[1].p-value", closeTo(0.391075888, 1e-5))
        .body("stats[2].p-value", closeTo(0.08593750, 1e-5)) // getting null
        .body("stats[3].p-value", closeTo(0.116116524, 1e-5))
        .body("stats[4].p-value", closeTo(0.764805599, 1e-5))
        .body("stats.size", is(5))
        .body("nsamples", is(6))
        .extract()
        .response()

    println(response.asString())

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "noCovariatesChrom2",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "2", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].chrom", is("2"))
        .body("stats[0].pos", is(1))
        .body("stats[0].p-value", is(nullValue()))
        .body("stats[1].p-value", is(nullValue()))
        .body("stats[2].p-value", is(nullValue()))
        .body("stats[3].p-value", is(nullValue()))
        .body("stats[4].p-value", is(nullValue()))
        .body("stats.size", is(5))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "twoPhenotypeCovariates",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "sa.rest.SEX"},
            |                        {"type": "phenotype", "name": "sa.rest.PC1"}
            |                      ],
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].chrom", is("1"))
        .body("stats[0].pos", is(1))
        .body("stats[0].p-value", closeTo(0.8432711, 1e-5))
        .body("stats[1].p-value", closeTo(0.24728705, 1e-5))
        .body("stats[2].p-value", closeTo(0.2533675, 1e-5))
        .body("stats[3].p-value", closeTo(0.14174710, 1e-5))
        .body("stats[4].p-value", closeTo(0.94999262, 1e-5))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "twoVariantCovariates",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "covariates"      : [
            |                        {"type": "variant", "chrom": "1", "pos": 1, "ref": "C", "alt": "T"},
            |                        {"type": "variant", "chrom": "1", "pos": 2, "ref": "C", "alt": "T"}
            |                      ],
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].chrom", is("1"))
        .body("stats[0].pos", is(1))
        .body("stats[0].p-value", anyOf(closeTo(1.0, 1e-3), is(nullValue)): AnyOf[java.lang.Double])
        .body("stats[1].p-value", anyOf(closeTo(1.0, 1e-3), is(nullValue)): AnyOf[java.lang.Double])
        .body("stats[2].p-value", closeTo(0.13397460, 1e-5))
        .body("stats[3].p-value", closeTo(0.32917961, 1e-5))
        .body("stats[4].p-value", anyOf(closeTo(1.0, 1e-3), is(nullValue)): AnyOf[java.lang.Double]) // NaN
        .extract()
        .response()

    println(response.asString())

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "onePhenotypeCovariateOneVariantCovariate",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "sa.rest.PC1"},
            |                        {"type": "variant", "chrom": "1", "pos": 1, "ref": "C", "alt": "T"}
            |                      ],
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].chrom", is("1"))
        .body("stats[0].pos", is(1))
        .body("stats[0].p-value", anyOf(closeTo(1.0, 1e-3), is(nullValue)): AnyOf[java.lang.Double])
        .body("stats[1].p-value", closeTo(0.48008491, 1e-5))
        .body("stats[2].p-value", closeTo(0.2304332, 1e-5))
        .body("stats[3].p-value", closeTo(0.06501361, 1e-5))
        .body("stats[4].p-value", closeTo(0.92741922, 1e-5))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "alternativePhenotype",
            |  "api_version"     : 1,
            |  "phenotype"      : "sa.rest.SEX",
            |  "covariates"     :  [
            |                        {"type": "phenotype", "name": "sa.rest.T2D"}
            |                      ],
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].chrom", is("1"))
        .body("stats[0].pos", is(1))
        .body("stats[0].p-value", closeTo(0.08810813, 1e-5))
        .body("stats[1].p-value", closeTo(0.6299924, 1e-5))
        .body("stats[2].p-value", closeTo(0.9194733, 1e-5))
        .body("stats[3].p-value", closeTo(0.7878046, 1e-5))
        .body("stats[4].p-value", closeTo(0.6299924, 1e-5))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "posEq",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "eq", "value": 2, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].chrom", is("1"))
        .body("stats[0].pos", is(2))
        .body("stats.size", is(1))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "chromAndPosEq",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "2", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "eq", "value": 2, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].chrom", is("2"))
        .body("stats[0].pos", is(2))
        .body("stats.size", is(1))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "incompatiblePos",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "eq", "value": 1, "operand_type": "integer"},
            |                        {"operand": "pos", "operator": "eq", "value": 2, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("Window is empty"))
        .extract()
        .response()
    
    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "posGteLte",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "gte", "value": 1, "operand_type": "integer"},
            |                        {"operand": "pos", "operator": "lte", "value": 2, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].chrom", is("1"))
        .body("stats[0].pos", is(1))
        .body("stats[1].chrom", is("1"))
        .body("stats[1].pos", is(2))
        .body("stats.size", is(2))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "posGtLte",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "gt", "value": 1, "operand_type": "integer"},
            |                        {"operand": "pos", "operator": "lte", "value": 2, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].chrom", is("1"))
        .body("stats[0].pos", is(2))
        .body("stats.size", is(1))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "posGteLt",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "gte", "value": 1, "operand_type": "integer"},
            |                        {"operand": "pos", "operator": "lt", "value": 2, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].chrom", is("1"))
        .body("stats[0].pos", is(1))
        .body("stats.size", is(1))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "posGtLt",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "gt", "value": 1, "operand_type": "integer"},
            |                        {"operand": "pos", "operator": "lt", "value": 2, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("Window is empty"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "posGtEq",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "gt", "value": 1, "operand_type": "integer"},
            |                        {"operand": "pos", "operator": "eq", "value": 2, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].chrom", is("1"))
        .body("stats[0].pos", is(2))
        .body("stats.size", is(1))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "limit",
            |  "api_version"     : 1,
            |  "limit"           : 3,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats.size", is(3))
        .body("passback", is("limit"))
        .body("count", is(3))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "count",
            |  "api_version"     : 1,
            |  "count"           : true,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats", is(nullValue()))
        .body("count", is(5))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "sortPos",
            |  "api_version"     : 1,
            |  "sort_by"         : [ "pos" ],
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].pos", is(1))
        .body("stats[1].pos", is(2))
        .body("stats[2].pos", is(150000))
        .body("stats[3].pos", is(250000))
        .body("stats[4].pos", is(350000))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "sortPosRef",
            |  "api_version"     : 1,
            |  "sort_by"         : [ "pos", "ref" ],
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].pos", is(1))
        .body("stats[1].pos", is(2))
        .body("stats[2].pos", is(150000))
        .body("stats[3].pos", is(250000))
        .body("stats[4].pos", is(350000))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "sortPosRefAlt",
            |  "api_version"     : 1,
            |  "sort_by"         : [ "pos", "ref", "alt" ],
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].pos", is(1))
        .body("stats[1].pos", is(2))
        .body("stats[2].pos", is(150000))
        .body("stats[3].pos", is(250000))
        .body("stats[4].pos", is(350000))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "sortRef",
            |  "api_version"     : 1,
            |  "sort_by"         : [ "ref" ],
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[2].pos", is(1))
        .body("stats[3].pos", is(2))
        .body("stats[0].pos", is(150000))
        .body("stats[1].pos", is(250000))
        .body("stats[4].pos", is(350000))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "sortRefAlt",
            |  "api_version"     : 1,
            |  "sort_by"         : [ "ref", "alt" ],
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[3].pos", is(1))
        .body("stats[4].pos", is(2))
        .body("stats[0].pos", is(150000))
        .body("stats[1].pos", is(250000))
        .body("stats[2].pos", is(350000))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "sortAlt",
            |  "api_version"     : 1,
            |  "sort_by"         : [ "alt" ],
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[1].pos", is(1))
        .body("stats[2].pos", is(2))
        .body("stats[3].pos", is(150000))
        .body("stats[4].pos", is(250000))
        .body("stats[0].pos", is(350000))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "sortAltRef",
            |  "api_version"     : 1,
            |  "sort_by"         : [ "alt", "ref" ],
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[3].pos", is(1))
        .body("stats[4].pos", is(2))
        .body("stats[1].pos", is(150000))
        .body("stats[2].pos", is(250000))
        .body("stats[0].pos", is(350000))
        .extract()
        .response()


    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "sortRefPosAlt",
            |  "api_version"     : 1,
            |  "sort_by"         : [ "ref", "pos", "alt" ],
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[2].pos", is(1))
        .body("stats[3].pos", is(2))
        .body("stats[0].pos", is(150000))
        .body("stats[1].pos", is(250000))
        .body("stats[4].pos", is(350000))
        .extract()
        .response()


    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "sortRefAltPos",
            |  "api_version"     : 1,
            |  "sort_by"         : [ "ref", "alt", "pos" ],
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[3].pos", is(1))
        .body("stats[4].pos", is(2))
        .body("stats[0].pos", is(150000))
        .body("stats[1].pos", is(250000))
        .body("stats[2].pos", is(350000))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "sortPValue",
            |  "api_version"     : 1,
            |  "sort_by"         : [ "p-value" ],
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[3].pos", is(1))
        .body("stats[2].pos", is(2))
        .body("stats[0].pos", is(150000))
        .body("stats[1].pos", is(250000))
        .body("stats[4].pos", is(350000))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "sortRefPValue",
            |  "api_version"     : 1,
            |  "sort_by"         : [ "ref", "p-value" ],
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[3].pos", is(1))
        .body("stats[2].pos", is(2))
        .body("stats[0].pos", is(150000))
        .body("stats[1].pos", is(250000))
        .body("stats[4].pos", is(350000))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "sortAltPValue",
            |  "api_version"     : 1,
            |  "sort_by"         : [ "alt", "p-value" ],
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[4].pos", is(1))
        .body("stats[3].pos", is(2))
        .body("stats[1].pos", is(150000))
        .body("stats[2].pos", is(250000))
        .body("stats[0].pos", is(350000))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "macSortAltPValue",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"},
            |                        {"operand": "mac", "operator": "gte", "value": 4, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].p-value", is(nullValue()))
        .body("stats[1].p-value", closeTo(0.391075888, 1e-5))
        .body("stats[2].p-value", is(nullValue())) // mac is computed without mean imputation
        .body("stats[3].p-value", is(nullValue()))
        .body("stats[4].p-value", closeTo(0.764805599, 1e-5))
        .extract()
        .response()

    println(response.asString())
    
    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "macBounds",
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
        .body("stats[0].p-value", is(nullValue())) // all AC are 2
        .body("stats[1].p-value", is(nullValue()))
        .body("stats[2].p-value", is(nullValue()))
        .body("stats[3].p-value", is(nullValue()))
        .body("stats[4].p-value", is(nullValue()))
        .extract()
        .response()

    println(response.asString())
    
    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "errorUnknownMDVersion",
            |  "md_version"      : "-1",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("Unknown md_version"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "errorUnsupportedAPIVersion",
            |  "api_version"     : -1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("Unsupported API version"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "errorInvalidPhenotypeName",
            |  "api_version"     : 1,
            |  "phenotype"       : "notAPhenotype",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("not a valid phenotype name"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "errorInvalidCovariateName",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "notACovariate"}
            |                      ],
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("not a valid covariate name"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "errorMissingCovariateName",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "covariates"      : [
            |                        {"type": "phenotype"}
            |                      ],
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("Covariate of type 'phenotype' must include 'name' field in request"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "errorMissingVariantInfo",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "covariates"      : [
            |                        {"type": "variant", "name": "missingVariantInfo"}
            |                      ],
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("Covariate of type 'variant' must include 'chrom', 'pos', 'ref', and 'alt' fields in request"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "errorUnsupportedCovariateType",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "covariates"      : [
            |                        {"type": "notACovariateType"}
            |                      ],
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("Covariate type must be 'phenotype' or 'variant'"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "errorResponseIsCovariate",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "sa.rest.T2D"}
            |                      ],
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("sa.rest.T2D appears as both response phenotype and covariate"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "errorUnsupportedPosOperator",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "notAChromOperator", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("chrom filter operator must be 'eq'"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "errorUnsupportedPosOperator",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "notAPosOperator", "value": 1, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("pos filter operator must be 'gte', 'gt', 'lte', 'lt', or 'eq'"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "errorInvalidSortField",
            |  "api_version"     : 1,
            |  "sort_by"         : [ "notASortField" ],
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("Valid sort_by arguments are `pos', `ref', `alt', and `p-value'"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "errorNegLimit",
            |  "api_version"     : 1,
            |  "limit"           : -1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("limit must be non-negative"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "errorPosString",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "gte", "value": "1", "operand_type": "string"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("pos filter operand_type must be 'integer'"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "errorChromGte",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "gte", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("chrom filter operator must be 'eq' and operand_type must be 'string'"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "errorChromInteger",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": 1, "operand_type": "integer"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("chrom filter operator must be 'eq' and operand_type must be 'string'"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "errorChromIntString",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": 1, "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats.size", is(5))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "notErrorPosValueString",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "eq", "value": "1", "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].pos", is(1))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "notErrorPosValueString",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "eq", "value": 1.5, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("must be a valid non-negative integer"))
        .extract()
        .response()

    println(response.asString())

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "notErrorPosValueString",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "eq", "value": "abc", "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("must be a valid non-negative integer"))
        .extract()
        .response()

    println(response.asString())

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "sortNotDistinct",
            |  "api_version"     : 1,
            |  "sort_by"         : [ "alt", "ref", "alt" ],
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("error_message", containsString("sort_by arguments must be distinct"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "sortNotDistinct",
            |  "api_version"     : 1,
            |  "sort_by"         : [ "invalidSortArg" ],
            |  "phenotype"       : "sa.rest.T2D",
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("error_message", containsString("Valid sort_by arguments are `pos', `ref', `alt', and `p-value'"))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "missingVariantCovariate",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "covariates"      : [
            |                        {"type": "variant", "chrom": "3", "pos": 1, "ref": "C", "alt": "T"}
            |                      ],
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("error_message", containsString("VDS does not contain variant covariate 3:1:C:T"))
        .extract()
        .response()
    
        response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "missingVariantCovariate",
            |  "api_version"     : 1,
            |  "phenotype"       : "sa.rest.T2D",
            |  "covariates"      : [
            |                        {"type": "variant", "chrom": "2", "pos": 1, "ref": "C", "alt": "A"}
            |                      ],
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "lte", "value": 500000, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .extract()
        .response()
    
    r.task.shutdownNow()
    t.join()
  }
}