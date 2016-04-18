package org.broadinstitute.hail.rest

import org.apache.spark.sql.Row
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.driver.{Command, State}
import org.broadinstitute.hail.variant.HardCallSet
import org.http4s.server.blaze.BlazeBuilder
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object T2DServer extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-c", aliases= Array("--covariate-file"), usage = "Covariate file")
    var covFile: String = _

    @Args4jOption(required = false, name = "-p", aliases = Array("--port"), usage = "Service port")
    var port: Int = 6062

    @Args4jOption(required = true, name = "-h1", aliases = Array("--hcs100Kb"), usage = ".hcs with 100Kb block")
    var hcsFile: String = _

    @Args4jOption(required = true, name = "-h2", aliases = Array("--hcs1Mb"), usage = ".hcs with 1Mb block")
    var hcs1MbFile: String = _

    @Args4jOption(required = true, name = "-h3", aliases = Array("--hcs10Mb"), usage = ".hcs with 10Mb block")
    var hcs10MbFile: String = _
  }

  def newOptions = new Options

  def name = "t2dserver"

  def description = "Run T2D REST server"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {

    val hcs = HardCallSet.read(state.sqlContext, options.hcsFile)

    val hcs1Mb = HardCallSet.read(state.sqlContext, options.hcs1MbFile)

    val hcs10Mb = HardCallSet.read(state.sqlContext, options.hcs10MbFile)

//    val covTypes = "IID: String, SEX: Double, T2D: Double, CohortBotnia: Double, CohortDGISib: Double, CohortDiabReg: Double, CohortFUSION: Double, CohortKORA: Double, CohortMalmoSib: Double, CohortMPP: Double, CohortSTT: Double, CohortWTCCC: Double, SubCohortFUSION1: Double, SubCohortFUSION2: Double, SubCohortFUSION3: Double, SubCohortFUSION4: Double, SubCohortFUSION5: Double, SubCohortFUSION6: Double, SubCohortKORAF3: Double, SubCohortKORAF4: Double, Age: Double, CountryBotnia: Double, CountryFinland: Double, CountryGermany: Double, CountrySweden: Double, CountryUK: Double, fastingInsulin: Double, BPMEDS: Double, LIPIDMEDS: Double, height: Double, height_adjT2D: Double, height_adjT2D_invn: Double, BMI: Double, BMI_adj_invn_withincohort: Double, BMI_adj_withincohort_invn: Double, BMI_adjT2D_invn: Double, WC: Double, WC_adjT2D_invn: Double, WC_adjT2D_invn_males: Double, WC_adjT2D_invn_females: Double, HIP: Double, HIP_adjT2D_invn: Double, HIP_adjT2D_invn_males: Double, HIP_adjT2D_invn_females: Double, WHR: Double, WHR_adjT2D_invn: Double, WHR_adjT2D_invn_males: Double, WHR_adjT2D_invn_females: Double, LDL: Double, LDL_allvalues_adjT2D_invn: Double, LDL_lipidmeds_divide.7: Double, LDL_lipidmeds_divide.7_adjT2D_invn: Double, LDL_lipidmeds_excluded: Double, LDL_lipidmeds_excluded_adjT2D_invn: Double, HDL: Double, HDL_adjT2D_invn: Double, TC: Double, TC_adjT2D_invn: Double, TG: Double, TG_adjT2D_invn: Double, fastingGlucose: Double, fastingGlucose_adj: Double, fastingGlucose_adj_invn: Double, logfastingInsulin: Double, logfastingInsulin_adj: Double, logfastingInsulin_adj_invn: Double, adjSBP: Double, adjSBP_adjT2D: Double, adjSBP_adjT2D_invn: Double, adjDBP: Double, adjDBP_adjT2D: Double, adjDBP_adjT2D_invn: Double, NUMSING: Double, NUMRARE: Double, PC_1KG_1: Double, PC_1KG_2: Double, PC_1KG_3: Double, PC_1KG_4: Double, PC_1KG_5: Double, PC_1KG_6: Double, PC_1KG_7: Double, PC_1KG_8: Double, PC_1KG_9: Double, PC_1KG_10: Double, PC_1KG_11: Double, PC_1KG_12: Double, PC_1KG_13: Double, PC_1KG_14: Double, PC_1KG_15: Double, PC_1KG_16: Double, PC_1KG_17: Double, PC_1KG_18: Double, PC_1KG_19: Double, PC_1KG_20: Double, PC_1KG_21: Double, PC_1KG_22: Double, PC_1KG_23: Double, PC_1KG_24: Double, PC_1KG_25: Double, PC_1KG_26: Double, PC_1KG_27: Double, PC_1KG_28: Double, PC_1KG_29: Double, PC_1KG_30: Double, PC_1KG_31: Double, PC_1KG_32: Double, PC_1KG_33: Double, PC_1KG_34: Double, PC_1KG_35: Double, PC_1KG_36: Double, PC_1KG_37: Double, PC_1KG_38: Double, PC_1KG_39: Double, PC_1KG_40: Double, PC_1KG_41: Double, PC_1KG_42: Double, PC_1KG_43: Double, PC_1KG_44: Double, PC_1KG_45: Double, PC_1KG_46: Double, PC_1KG_47: Double, PC_1KG_48: Double, PC_1KG_49: Double, PC_1KG_50: Double, PC_1KG_51: Double, PC_1KG_52: Double, PC_1KG_53: Double, PC_1KG_54: Double, PC_1KG_55: Double, PC_1KG_56: Double, PC_1KG_57: Double, PC_1KG_58: Double, PC_1KG_59: Double, PC_1KG_60: Double, PC_1KG_61: Double, PC_1KG_62: Double, PC_1KG_63: Double, PC_1KG_64: Double, PC_1KG_65: Double, PC_1KG_66: Double, PC_1KG_67: Double, PC_1KG_68: Double, PC_1KG_69: Double, PC_1KG_70: Double, PC_1KG_71: Double, PC_1KG_72: Double, PC_1KG_73: Double, PC_1KG_74: Double, PC_1KG_75: Double, PC_1KG_76: Double, PC_1KG_77: Double, PC_1KG_78: Double, PC_1KG_79: Double, PC_1KG_80: Double, PC_1KG_81: Double, PC_1KG_82: Double, PC_1KG_83: Double, PC_1KG_84: Double, PC_1KG_85: Double, PC_1KG_86: Double, PC_1KG_87: Double, PC_1KG_88: Double, PC_1KG_89: Double, PC_1KG_90: Double, PC_1KG_91: Double, PC_1KG_92: Double, PC_1KG_93: Double, PC_1KG_94: Double, PC_1KG_95: Double, PC_1KG_96: Double, PC_1KG_97: Double, PC_1KG_98: Double, PC_1KG_99: Double, PC_1KG_100: Double, PC_GoT2D_1: Double, PC_GoT2D_2: Double, PC_GoT2D_3: Double, PC_GoT2D_4: Double, PC_GoT2D_5: Double, PC_GoT2D_6: Double, PC_GoT2D_7: Double, PC_GoT2D_8: Double, PC_GoT2D_9: Double, PC_GoT2D_10: Double, PC_GoT2D_11: Double, PC_GoT2D_12: Double, PC_GoT2D_13: Double, PC_GoT2D_14: Double, PC_GoT2D_15: Double, PC_GoT2D_16: Double, PC_GoT2D_17: Double, PC_GoT2D_18: Double, PC_GoT2D_19: Double, PC_GoT2D_20: Double, PC_GoT2D_21: Double, PC_GoT2D_22: Double, PC_GoT2D_23: Double, PC_GoT2D_24: Double, PC_GoT2D_25: Double, PC_GoT2D_26: Double, PC_GoT2D_27: Double, PC_GoT2D_28: Double, PC_GoT2D_29: Double, PC_GoT2D_30: Double, PC_GoT2D_31: Double, PC_GoT2D_32: Double, PC_GoT2D_33: Double, PC_GoT2D_34: Double, PC_GoT2D_35: Double, PC_GoT2D_36: Double, PC_GoT2D_37: Double, PC_GoT2D_38: Double, PC_GoT2D_39: Double, PC_GoT2D_40: Double, PC_GoT2D_41: Double, PC_GoT2D_42: Double, PC_GoT2D_43: Double, PC_GoT2D_44: Double, PC_GoT2D_45: Double, PC_GoT2D_46: Double, PC_GoT2D_47: Double, PC_GoT2D_48: Double, PC_GoT2D_49: Double, PC_GoT2D_50: Double, PC_GoT2D_51: Double, PC_GoT2D_52: Double, PC_GoT2D_53: Double, PC_GoT2D_54: Double, PC_GoT2D_55: Double, PC_GoT2D_56: Double, PC_GoT2D_57: Double, PC_GoT2D_58: Double, PC_GoT2D_59: Double, PC_GoT2D_60: Double, PC_GoT2D_61: Double, PC_GoT2D_62: Double, PC_GoT2D_63: Double, PC_GoT2D_64: Double, PC_GoT2D_65: Double, PC_GoT2D_66: Double, PC_GoT2D_67: Double, PC_GoT2D_68: Double, PC_GoT2D_69: Double, PC_GoT2D_70: Double, PC_GoT2D_71: Double, PC_GoT2D_72: Double, PC_GoT2D_73: Double, PC_GoT2D_74: Double, PC_GoT2D_75: Double, PC_GoT2D_76: Double, PC_GoT2D_77: Double, PC_GoT2D_78: Double, PC_GoT2D_79: Double, PC_GoT2D_80: Double, PC_GoT2D_81: Double, PC_GoT2D_82: Double, PC_GoT2D_83: Double, PC_GoT2D_84: Double, PC_GoT2D_85: Double, PC_GoT2D_86: Double, PC_GoT2D_87: Double, PC_GoT2D_88: Double, PC_GoT2D_89: Double, PC_GoT2D_90: Double, PC_GoT2D_91: Double, PC_GoT2D_92: Double, PC_GoT2D_93: Double, PC_GoT2D_94: Double, PC_GoT2D_95: Double, PC_GoT2D_96: Double, PC_GoT2D_97: Double, PC_GoT2D_98: Double, PC_GoT2D_99: Double, PC_GoT2D_100: Double, C1: Double, C2: Double, BEFORE: Double, AFTER: Double, OM1: Double, OM2: Double, OM3: Double, OM4: Double"
//
//    val newState = AnnotateSamplesTSV.run(state, Array(
//      "-i", options.covFile,
//      "-t", covTypes,
//      "-s", "IID",
//      "-r", "got2d"))

    val (covNames, sampleCovs): (Array[String], Map[String, Array[Double]]) =
      readLines(options.covFile, state.hadoopConf) { lines =>
        if (lines.isEmpty)
          fatal("empty TSV file")

        val covNames = lines.next().value.split("\\t").drop(1)
        val nFields = covNames.length + 1

        val sampleCovs = lines.map {
            _.transform { l =>
              val lineSplit = l.value.split("\\t")
              if (lineSplit.length != nFields)
                fatal(s"expected $nFields fields, but got ${lineSplit.length}")
              (lineSplit(0), lineSplit.drop(1).map(_.toDouble))
            }
          }.toMap

        (covNames, sampleCovs)
      }

//    val ab = new mutable.ArrayBuilder.ofDouble()
//    for (s <- hcs.sampleIds) ab ++= sampleCovs(s)

    val covMap: Map[String, Array[Double]] = covNames.zipWithIndex
      .map{ case (name, j) => (name, hcs.sampleIds.map(s => sampleCovs(s)(j)).toArray) }.toMap

    val service = new T2DService(hcs, hcs1Mb, hcs10Mb, covMap)
    val task = BlazeBuilder.bindHttp(options.port, "0.0.0.0")
      .mountService(service.service, "/")
      .run
    task.awaitShutdown()

    state
  }
}
