package org.broadinstitute.hail.rest

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
import org.hamcrest.core.AnyOf

class T2DRunnable(sc: SparkContext, sqlContext: SQLContext) extends Runnable {
  var task: Server = _

  override def run() {

    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/t2dserver.vcf"))
    s = SplitMulti.run(s)

    val hcs = HardCallSet(sqlContext, s.vds, sparseCutoff = 0).rangePartition(2)
    val hcs1Mb =  hcs.capNVariantsPerBlock(maxPerBlock = 1000, newBlockWidth =  1000000).rangePartition(2)
    val hcs10Mb = hcs1Mb.capNVariantsPerBlock(maxPerBlock = 1000, newBlockWidth = 10000000).rangePartition(2)
    val covMap = T2DServer.readCovData(s, "src/test/resources/t2dserver.cov", hcs.sampleIds)

//    hcs.write(sqlContext, "src/test/resources/t2dserver100Kb.hcs")
//    hcs.write(sqlContext, "src/test/resources/t2dserver1Mb.hcs")
//    hcs.write(sqlContext, "src/test/resources/t2dserver10Mb.hcs")
//
//    hcs.rdd.foreach(println)
//    println()
//    hcs1Mb.rdd.foreach(println)
//    println()
//    hcs10Mb.rdd.foreach(println)

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

  // RestAssured.config = RestAssured.config().jsonConfig...

  @BeforeClass
  override def beforeClass() = {
    super.beforeClass()
    r = new T2DRunnable(sc, sqlContext)
    t = new Thread(r)
    t.start()
    Thread.sleep(8000)
  }

  @Test def test() {

    /*
    Sample code for generating p-values in R:
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
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "BMI"}
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
        .body("stats[2].p-value", closeTo(0.2337485, 1e-5))
        .body("stats[3].p-value", closeTo(0, 1e-5)) // FIXME: perfect fit, SE approx 0
        .body("stats[4].p-value", closeTo(0.8443759, 1e-5))
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
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "HEIGHT"}
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
        .body("stats[1].p-value", closeTo(0.05123045, 1e-5))
        .body("stats[2].p-value", closeTo(0.9242424, 1e-5))
        .body("stats[3].p-value", closeTo(0.08263506, 1e-5))
        .body("stats[4].p-value", closeTo(0.12565524, 1e-5))
        .extract()
        .response()

    println(response.asString())

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "PhenotypeBMIAndCovariateHEIGHT",
            |  "api_version"     : 1,
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "HEIGHT"}
            |                      ],
            |  "phenotype"       : "BMI"
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].chrom", is("1"))
        .body("stats[0].pos", is(1))
        .body("stats[0].p-value", closeTo(0.8599513, 1e-5))
        .body("stats[1].p-value", closeTo(0.1276718, 1e-5))
        .body("stats[2].p-value", closeTo(0.3953922, 1e-5))
        .body("stats[3].p-value", closeTo(0.1400487, 1e-5))
        .body("stats[4].p-value", closeTo(0.2677205, 1e-5))
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
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "HEIGHT"},
            |                        {"type": "variant", "chrom": "1", "pos": 1, "ref": "C", "alt": "T"}
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
        .body("stats[1].p-value", closeTo(0.04233032, 1e-5))
        .body("stats[2].p-value", closeTo(0.9478751, 1e-5))
        .body("stats[3].p-value", closeTo(0.2634229, 1e-5))
        .body("stats[4].p-value", closeTo(0.06779419, 1e-5))
        .extract()
        .response()

    response =
      given()
        .config(config().jsonConfig(new JsonConfig(NumberReturnType.DOUBLE)))
        .contentType("application/json")
        .body(
          """{
            |  "passback"        : "noCovariatesChrom1",
            |  "api_version"     : 1,
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"}
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
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "2", "operand_type": "string"}
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
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "SEX"},
            |                        {"type": "phenotype", "name": "PC1"}
            |                      ]
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
            |  "covariates"      : [
            |                        {"type": "variant", "chrom": "1", "pos": 1, "ref": "C", "alt": "T"},
            |                        {"type": "variant", "chrom": "1", "pos": 2, "ref": "C", "alt": "T"}
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
        .body("stats[1].p-value", anyOf(closeTo(1.0, 1e-3), is(nullValue)): AnyOf[java.lang.Double])
        .body("stats[2].p-value", closeTo(0.13397460, 1e-5))
        .body("stats[3].p-value", closeTo(0.32917961, 1e-5))
        .body("stats[4].p-value", anyOf(closeTo(1.0, 1e-3), is(nullValue)): AnyOf[java.lang.Double]) // FIXME: getting NaN
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
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "PC1"},
            |                        {"type": "variant", "chrom": "1", "pos": 1, "ref": "C", "alt": "T"}
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
            |  "phenotype"      : "SEX",
            |  "covariates"     :  [
            |                        {"type": "phenotype", "name": "T2D"}
            |                      ]
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
            |  "passback"        : "defaultToChrom1",
            |  "api_version"     : 1
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
        .body("stats[2].chrom", is("1"))
        .body("stats[2].pos", is(900000))
        .body("stats[3].chrom", is("1"))
        .body("stats[3].pos", is(9000000))
        .body("stats[4].chrom", is("1"))
        .body("stats[4].pos", is(90000000))
        .body("stats.size", is(5))
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
            |  "variant_filters" : [
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
            |  "variant_filters" : [
            |                        {"operand": "pos", "operator": "eq", "value": 1, "operand_type": "integer"},
            |                        {"operand": "pos", "operator": "eq", "value": 2, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats.size", is(0))
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
            |  "variant_filters" : [
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
            |  "variant_filters" : [
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
            |  "variant_filters" : [
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
            |  "variant_filters" : [
            |                        {"operand": "pos", "operator": "gt", "value": 1, "operand_type": "integer"},
            |                        {"operand": "pos", "operator": "lt", "value": 2, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats.size", is(0))
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
            |  "variant_filters" : [
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
            |  "limit"           : 3
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
            |  "count"           : true
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
            |  "sort_by"         : [ "pos" ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].pos", is(1))
        .body("stats[1].pos", is(2))
        .body("stats[2].pos", is(900000))
        .body("stats[3].pos", is(9000000))
        .body("stats[4].pos", is(90000000))
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
            |  "sort_by"         : [ "pos", "ref" ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].pos", is(1))
        .body("stats[1].pos", is(2))
        .body("stats[2].pos", is(900000))
        .body("stats[3].pos", is(9000000))
        .body("stats[4].pos", is(90000000))
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
            |  "sort_by"         : [ "pos", "ref", "alt" ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[0].pos", is(1))
        .body("stats[1].pos", is(2))
        .body("stats[2].pos", is(900000))
        .body("stats[3].pos", is(9000000))
        .body("stats[4].pos", is(90000000))
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
            |  "sort_by"         : [ "ref" ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[2].pos", is(1))
        .body("stats[3].pos", is(2))
        .body("stats[0].pos", is(900000))
        .body("stats[1].pos", is(9000000))
        .body("stats[4].pos", is(90000000))
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
            |  "sort_by"         : [ "ref", "alt" ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[3].pos", is(1))
        .body("stats[4].pos", is(2))
        .body("stats[0].pos", is(900000))
        .body("stats[1].pos", is(9000000))
        .body("stats[2].pos", is(90000000))
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
            |  "sort_by"         : [ "alt" ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[1].pos", is(1))
        .body("stats[2].pos", is(2))
        .body("stats[3].pos", is(900000))
        .body("stats[4].pos", is(9000000))
        .body("stats[0].pos", is(90000000))
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
            |  "sort_by"         : [ "alt", "ref" ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[3].pos", is(1))
        .body("stats[4].pos", is(2))
        .body("stats[1].pos", is(900000))
        .body("stats[2].pos", is(9000000))
        .body("stats[0].pos", is(90000000))
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
            |  "sort_by"         : [ "ref", "pos", "alt" ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[2].pos", is(1))
        .body("stats[3].pos", is(2))
        .body("stats[0].pos", is(900000))
        .body("stats[1].pos", is(9000000))
        .body("stats[4].pos", is(90000000))
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
            |  "sort_by"         : [ "ref", "alt", "pos" ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[3].pos", is(1))
        .body("stats[4].pos", is(2))
        .body("stats[0].pos", is(900000))
        .body("stats[1].pos", is(9000000))
        .body("stats[2].pos", is(90000000))
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
            |  "sort_by"         : [ "p-value" ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[3].pos", is(1))
        .body("stats[2].pos", is(2))
        .body("stats[0].pos", is(900000))
        .body("stats[1].pos", is(9000000))
        .body("stats[4].pos", is(90000000))
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
            |  "sort_by"         : [ "ref", "p-value" ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[3].pos", is(1))
        .body("stats[2].pos", is(2))
        .body("stats[0].pos", is(900000))
        .body("stats[1].pos", is(9000000))
        .body("stats[4].pos", is(90000000))
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
            |  "sort_by"         : [ "alt", "p-value" ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(200)
        .body("is_error", is(false))
        .body("stats[4].pos", is(1))
        .body("stats[3].pos", is(2))
        .body("stats[1].pos", is(900000))
        .body("stats[2].pos", is(9000000))
        .body("stats[0].pos", is(90000000))
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
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
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
            |  "passback"        : "errorUnknownMDVersion",
            |  "md_version"      : "-1",
            |  "api_version"     : 1
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
            |  "api_version"     : -1
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
            |  "phenotype"       : "notAPhenotype"
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
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "notACovariate"}
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
            |  "covariates"      : [
            |                        {"type": "phenotype"}
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
            |  "covariates"      : [
            |                        {"type": "variant", "name": "missingVariantInfo"}
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
            |  "covariates"      : [
            |                        {"type": "notACovariateType"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("Supported covariate types are phenotype and variant"))
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
            |  "covariates"      : [
            |                        {"type": "phenotype", "name": "T2D"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("is_error", is(true))
        .body("error_message", containsString("T2D appears as both the response phenotype and a covariate phenotype"))
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
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "notAChromOperator", "value": "1", "operand_type": "string"}
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
            |  "variant_filters" : [
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
            |  "sort_by"         : [ "notASortField" ]
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
            |  "limit"           : -1
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
            |  "variant_filters" : [
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
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "gte", "value": "1", "operand_type": "string"}
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
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": 1, "operand_type": "integer"}
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
            |  "variant_filters" : [
            |                        {"operand": "chrom", "operator": "eq", "value": 1, "operand_type": "string"}
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
            |  "variant_filters" : [
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
            |  "variant_filters" : [
            |                        {"operand": "pos", "operator": "eq", "value": 1.5, "operand_type": "integer"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        //.statusCode(200)
        //.body("is_error", is(false))
        //.body("stats[0].pos", is(1))  FIXME: how to handle?
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
            |  "sort_by"         : [ "alt", "ref", "alt" ]
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
            |  "sort_by"         : [ "invalidSortArg" ]
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
            |  "covariates"      : [
            |                        {"type": "variant", "chrom": "3", "pos": 1, "ref": "C", "alt": "T"}
            |                      ]
            |}""".stripMargin)
        .when()
        .post("/getStats")
        .`then`()
        .statusCode(400)
        .body("error_message", containsString("Variant 3:1:C:T is not in the hard call set"))
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