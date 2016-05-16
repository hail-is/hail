package org.broadinstitute.hail.rest

import org.http4s.server.Server
import org.http4s.server.blaze.BlazeBuilder
import org.testng.annotations.Test
import com.jayway.restassured.RestAssured._
import com.jayway.restassured.matcher.RestAssuredMatchers._
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.{ImportVCF, SplitMulti, State}
import org.broadinstitute.hail.variant.HardCallSet
import org.hamcrest.Matchers._

class T2DRunnable(sc: SparkContext, sqlContext: SQLContext) extends Runnable {
  var task: Server = _

  override def run() {

    println("run1")


    var s = State(sc, sqlContext)
    s = ImportVCF.run(s, Array("src/test/resources/t2dserver.vcf"))
    s = SplitMulti.run(s)

    println("run2")

    val hcs = HardCallSet(sqlContext, s.vds)

    println(hcs)

    println("run3")

    val hcs1Mb =     hcs.capNVariantsPerBlock(maxPerBlock = 1000, newBlockWidth =  1000000)
    val hcs10Mb = hcs1Mb.capNVariantsPerBlock(maxPerBlock = 1000, newBlockWidth = 10000000)
    val covMap = T2DServer.readCovData(s, "src/test/resources/t2dserver.cov", hcs.sampleIds)

    println("run4")


    val service = new T2DService(hcs, hcs1Mb, hcs10Mb, covMap)

    println("run5")

    task = BlazeBuilder.bindHttp(8080)
      .mountService(service.service, "/")
      .run

    println("run6")

    task
      .awaitShutdown()
  }
}

class T2DServerSuite extends SparkSuite {
  @Test def test() {
    val r = new T2DRunnable(sc, sqlContext)
    val t = new Thread(r)
    t.start()

    println("running task")

    Thread.sleep(2000)

    /*
        // FIXME small example in test/resources
        // FIXME test failure modes (missing fields, mal-formed JSON, etc.)

        // {"is_error":false,"stats":[{"chrom":"10","pos":114550600,"ref":"A","alt":"G","p-value":8.6684E-4}]}
        given()
          .contentType("application/json")
          .body(
            """{
              |  "api_version"     : 0.1,
              |  "variant_filters" : [
              |                        {"operand": "chrom", "operator": "eq", "value": "10", "operand_type": "string"},
              |                        {"operand": "pos", "operator": "eq", "value": 114550600, "operand_type": "integer"},
              |                      ],
              |}
            """.stripMargin)
          .when()
          .post("/getStats")
          .`then`()
          .statusCode(200)
          .body("is_error", equalTo(false))
          .body("stats[0].chrom", equalTo("10"))
          .body("stats[0].pos", equalTo(114550600))
          .body("stats[0].p-value", is(0.00086684f))
    */

    given()
      .contentType("application/json")
      .body(
        """{
          |  "api_version"     : 0.1,
          |  "variant_filters" : [
          |                        {"operand": "chrom", "operator": "eq", "value": "1", "operand_type": "string"},
          |                        {"operand": "pos", "operator": "eq", "value": 1, "operand_type": "integer"},
          |                      ],
          |}
        """.stripMargin)
      .when()
      .post("/getStats")
      //.`then`()
      //.statusCode(200)
      //.body("is_error", equalTo(false))


    r.task.shutdownNow()
    t.join()

  }
}