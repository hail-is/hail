package org.broadinstitute.hail.rest

import com.jayway.restassured.RestAssured
import org.http4s.server.Server
import org.http4s.server.blaze.BlazeBuilder
import org.testng.annotations.{AfterClass, BeforeClass, Test}
import com.jayway.restassured.RestAssured._
import com.jayway.restassured.config.JsonConfig
import com.jayway.restassured.path.json.config.JsonPathConfig.NumberReturnType
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.{ImportVCF, SplitMulti, State}
import org.broadinstitute.hail.variant.HardCallSet
import org.hamcrest.Matchers._

class T2DRunnable(sc: SparkContext, sqlContext: SQLContext) extends Runnable {
  var task: Server = _

  override def run() {

    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/t2dserver.vcf"))
    s = SplitMulti.run(s)

    val hcs = HardCallSet(sqlContext, s.vds)

    println(hcs)

    val hcs1Mb =     hcs.capNVariantsPerBlock(maxPerBlock = 1000, newBlockWidth =  1000000)
    val hcs10Mb = hcs1Mb.capNVariantsPerBlock(maxPerBlock = 1000, newBlockWidth = 10000000)
    val covMap = T2DServer.readCovData(s, "src/test/resources/t2dserver.cov", hcs.sampleIds)

    val service = new T2DService(hcs, hcs1Mb, hcs10Mb, covMap)

    task = BlazeBuilder.bindHttp(8080)
      .mountService(service.service, "/")
      .run

    task
      .awaitShutdown()
  }
}

class T2DServerSuite extends SparkSuite {
  var r: T2DRunnable = null
  var t: Thread = null

  //RestAssured.config = RestAssured.config().jsonConfig(

  @BeforeClass
  override def beforeClass() = {
    super.beforeClass()
    r = new T2DRunnable(sc, sqlContext)
    t = new Thread(r)
    t.start()
    Thread.sleep(2000)
  }

  @Test def test() {
    // FIXME test failure modes (missing fields, mal-formed JSON, etc.)

    var response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "noCovariates",
            |  "api_version"     : 1,
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "gte", "value": 1, "operand_type": "integer"},
            |                        {"operand": "pos", "operator": "lte", "value": 10, "operand_type": "integer"}
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
        .body("stats[2].p-value", closeTo(0.08593750, 1e-5))
        .body("stats[3].p-value", closeTo(0.116116524, 1e-5))
        .body("stats[4].p-value", isEmptyOrNullString)
        .body("stats[5].p-value", isEmptyOrNullString)
        .body("stats[6].p-value", isEmptyOrNullString)
        .body("stats[7].p-value", isEmptyOrNullString)
        .body("stats[8].p-value", isEmptyOrNullString)
        .body("stats[9].p-value", isEmptyOrNullString)
      .extract()
        .response()

      println(response.asString())

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "twoPhenotypeCovariates",
            |  "api_version"     : 1,
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "gte", "value": 1, "operand_type": "integer"},
            |                        {"operand": "pos", "operator": "lte", "value": 10, "operand_type": "integer"}
            |                      ],
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "SEX"},
            |                        {"type": "phenotype", "name": "PC1"}
            |  ]
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
        .body("stats[4].p-value", isEmptyOrNullString)
        .body("stats[5].p-value", isEmptyOrNullString)
        .body("stats[6].p-value", isEmptyOrNullString)
        .body("stats[7].p-value", isEmptyOrNullString)
        .body("stats[8].p-value", isEmptyOrNullString)
        .body("stats[9].p-value", isEmptyOrNullString)
      .extract()
        .response()

    println(response.asString())

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "twoVariantCovariates",
            |  "api_version"     : 1,
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "gte", "value": 1, "operand_type": "integer"},
            |                        {"operand": "pos", "operator": "lte", "value": 10, "operand_type": "integer"}
            |                      ],
            |  "covariates"      : [
            |                        {"type": "variant", "chrom": "1", "pos": 1, "ref": "C", "alt": "T"},
            |                        {"type": "variant", "chrom": "1", "pos": 2, "ref": "C", "alt": "T"}
            |  ]
            |}""".stripMargin)
      .when()
        .post("/getStats")
      .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].chrom", is("1"))
        .body("stats[0].pos", is(1))
//        .body("stats[0].p-value", anyOf(closeTo(1d, 1e-3)))
//        .body("stats[1].p-value", anyOf(closeTo(1d, 1e-3)))
        .body("stats[2].p-value", closeTo(0.13397460, 1e-5))
        .body("stats[3].p-value", closeTo(0.32917961, 1e-5))
        .body("stats[4].p-value", isEmptyOrNullString)
        .body("stats[5].p-value", isEmptyOrNullString)
        .body("stats[6].p-value", isEmptyOrNullString)
        .body("stats[7].p-value", isEmptyOrNullString)
        .body("stats[8].p-value", isEmptyOrNullString)
        .body("stats[9].p-value", isEmptyOrNullString)
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
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
            |                        {"operand": "pos", "operator": "gte", "value": 1, "operand_type": "integer"},
            |                        {"operand": "pos", "operator": "lte", "value": 10, "operand_type": "integer"}
            |                      ],
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "PC1"},
            |                        {"type": "variant", "chrom": "1", "pos": 1, "ref": "C", "alt": "T"}
            |  ]
            |}""".stripMargin)
        .when()
          .post("/getStats")
        .`then`()
          .statusCode(200)
          .body("is_error", is(false))
          .body("stats[0].chrom", is("1"))
          .body("stats[0].pos", is(1))
//          .body("stats[0].p-value", anyOf(closeTo(1.0d, 1e-3d), is(nullValue(Double.getClass))))
          .body("stats[1].p-value", closeTo(0.48008491, 1e-5))
          .body("stats[2].p-value", closeTo(0.2304332, 1e-5))
          .body("stats[3].p-value", closeTo(0.06501361, 1e-5))
          .body("stats[4].p-value", isEmptyOrNullString)
          .body("stats[5].p-value", isEmptyOrNullString)
          .body("stats[6].p-value", isEmptyOrNullString)
          .body("stats[7].p-value", isEmptyOrNullString)
          .body("stats[8].p-value", isEmptyOrNullString)
          .body("stats[9].p-value", isEmptyOrNullString)
        .extract()
          .response()

    println(response.asString())

  }


  @AfterClass(alwaysRun = true)
  override def afterClass() = {
    r.task.shutdownNow()
    t.join()
    super.afterClass()
  }
}