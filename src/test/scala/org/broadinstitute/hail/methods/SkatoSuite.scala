package org.broadinstitute.hail.methods

import org.broadinstitute.hail.{FatalException, SparkSuite}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.check._
import org.broadinstitute.hail.check.Prop._
import org.testng.annotations.Test
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.driver._
import sys.process._
import scala.language._

class SkatoSuite extends SparkSuite {

  def dichotomousPhenotype(nSamples: Int) = Gen.buildableOfN(nSamples, Gen.option(Gen.arbBoolean, 0.95))
  def quantitativePhenotype(nSamples: Int) = Gen.buildableOfN(nSamples, Gen.option(Gen.choose(-0.5, 0.5), 0.95))
  def covariateMatrix(nSamples: Int, numCovar: Int) = Gen.buildableOfN(numCovar, quantitativePhenotype(nSamples))

  object Spec extends Properties("SKAT-O") {
    val compGen = for (vds: VariantDataset <- VariantSampleMatrix.gen[Genotype](sc, Genotype.gen _);
                       blockSize: Int <- Gen.choose(1, 1000);
                       quantitative: Boolean <- Gen.arbBoolean;
                       nCovar: Int <- Gen.choose(0, 5);
                       nGroups: Int <- Gen.choose(1, 50);
                       y <- if (quantitative) quantitativePhenotype(vds.nSamples) else dichotomousPhenotype(vds.nSamples);
                       x <- covariateMatrix(vds.nSamples, nCovar);
                       seed <- Gen.arbInt;
                       nRho <- Gen.choose(1, 10);
                       rCorr <- Gen.frequency((3, Gen.buildableOfN[Array[Double], Double](1, Gen.const[Double](0.0))),
                         (3, Gen.buildableOfN[Array[Double], Double](1, Gen.const[Double](1.0))),
                         (3, Gen.distinctBuildableOfN[Array[Double], Double](nRho, Gen.choose(0.0, 1.0))));
                       imputeMethod <- Gen.oneOf("fixed", "bestguess"); //cannot test random because order of groups not identical between PLINK and Hail
                       kernel <- Gen.frequency((5,Gen.const("linear")),(5, Gen.const("linear.weighted")), (1, Gen.oneOf("IBS", "IBS.weighted", "quadratic", "2wayIX")));
                       nResampling <- Gen.frequency((5, Gen.const(0)), (5, Gen.choose(0, 100)));
                       typeResampling <- if (quantitative) Gen.oneOf("bootstrap", "permutation") else Gen.oneOf("bootstrap", "bootstrap.fast", "permutation");
                       noAdjustment <- Gen.arbBoolean;
                       method <- Gen.oneOf("davies", "liu", "liu.mod", "optimal.adj", "optimal");
                       missingCutoff <- Gen.choose(0.1, 0.6);
                       weightsBeta <- Gen.buildableOfN(2, Gen.choose(1, 25));
                       estimateMAF <- Gen.oneOf(1, 2)
    )
      yield (vds, blockSize, quantitative, nCovar, nGroups,
        y, x, seed, rCorr, imputeMethod, kernel, nResampling,
        typeResampling, noAdjustment, method, missingCutoff, weightsBeta, estimateMAF)


    property("Hail groups give same result as plink input") =
      forAll(compGen) { case (vds: VariantSampleMatrix[Genotype], blockSize: Int,
      quantitative: Boolean, nCovar: Int, nGroups: Int, y, x,
      seed: Int, rCorr, imputeMethod, kernel, nResampling, typeResampling, noAdjustment,
      method, missingCutoff, weightsBeta, estimateMAF) =>

        val fileRoot = tmpDir.createTempFile("skatoTest")
        val plinkRoot = fileRoot + ".plink"
        val ssd = fileRoot + ".SSD"
        val info = fileRoot + ".info"
        val resultsFile = fileRoot + ".results"
        val sampleAnnotations = fileRoot + ".sampleAnnotations.tsv"
        val variantAnnotations = fileRoot + ".variantAnnotations.tsv"
        val hailOutputFile = fileRoot + ".hailResults.tsv"
        val configSkato = fileRoot + ".config"
        val setID = fileRoot + ".setid"
        val covarFile = fileRoot + ".covar"

        var s = State(sc, sqlContext, vds.filterVariants((v, va, gs) => v.toString.length < 50 && divOption(gs.flatMap(_.nNonRefAlleles).sum, gs.count(_.isCalled) * 2) != Option(0.5))) //hack because SKAT-O won't accept snp ids with > 50 characters
        s = SplitMulti.run(s, Array.empty[String])
        s = VariantQC.run(s, Array.empty[String])
        s = FilterVariantsExpr.run(s, Array("--keep", "-c","va.qc.AF < 0.4")) //problem with MAF flip if too close to 0.5
        s = s.copy(vds = s.vds.filterVariants((v, va, gs) => v.toString.length < 50)) //hack because SKAT-O won't accept snp ids with > 50 characters
        s = ExportPlink.run(s, Array("-o", plinkRoot))

        val nNonMissing =
          if (x.isEmpty)
            y.count(_.isDefined)
          else
            y.zip(x.transpose).count{case (yi, xi) => yi.isDefined && (xi.isEmpty || xi.forall(_.isDefined))}

        if (s.vds.nSamples < 1 || s.vds.nVariants < 1 || nNonMissing < 1) //skat-o from plink files will fail if 0 variants or samples
          true
        else {

          // assign variants to groups
          val groups = s.vds.variants.zipWithIndex.map { case (v, i) =>
            val groupNum = i % nGroups
            (v, groupNum)
          }.collect()

          // Write set ID list
          writeTextFile(setID, sc.hadoopConfiguration) { w =>
            groups.foreach { case (v, i) => w.write(s"group_$i" + "\t" + v.toString + "\n")}}

          // Write covariate file
          writeTextFile(covarFile, sc.hadoopConfiguration) { w =>
            s.vds.sampleIds.zipWithIndex.foreach { case (s, i) =>
              val sb = new StringBuilder()
              for (j <- 0 until nCovar) {
                if (j != 0)
                  sb.append("\t")
                sb.append(x(j)(i).getOrElse(-9))
              }
              sb.append("\n")
              w.write(sb.result())
            }
          }

          // Write phenotype into fam file
          writeTextFile(plinkRoot + ".fam", sc.hadoopConfiguration) { w =>
            s.vds.sampleIds.zipWithIndex.foreach { case (s, i) =>
              val phenoString = y(i).map(_.toString) match {
                case None => -9
                case Some("true") => 2
                case Some("false") => 1
                case Some(p) => p
              }
              w.write(s"0\t$s\t0\t0\t0\t$phenoString\n")
            }
          }

          // Run SKAT-O from PLINK file
          val adjustString = if (noAdjustment) "--no-adjustment" else ""
          val yType = if (quantitative) "C" else "D"

          val plinkCommand = s"""Rscript src/test/resources/testSkatoPlink.R
                                 |--plink-root $plinkRoot
                                 |--covariates $covarFile
                                 |--setid $setID
                                 |--y-type $yType
                                 |--ssd-file $ssd
                                 |--info-file $info
                                 |--results-file $resultsFile
                                 |--ncovar $nCovar
                                 |--seed $seed
                                 |$adjustString
                                 |--kernel $kernel
                                 |--n-resampling $nResampling
                                 |--type-resampling $typeResampling
                                 |--impute-method $imputeMethod
                                 |--method $method
                                 |--r-corr ${rCorr.mkString(",")}
                                 |--missing-cutoff $missingCutoff
                                 |--weights-beta ${weightsBeta.mkString(",")}
                                 |--estimate-maf $estimateMAF
               """.stripMargin

          val plinkSkatExitCode = plinkCommand !

          val plinkResults = sc.parallelize(readLines(resultsFile, sc.hadoopConfiguration) { lines =>
            lines.map { l => l.transform { line =>
              val Array(groupName, pValue, nMarkers, nMarkersTested) = line.value.split( """\s+""")
              (groupName, (pValue, nMarkers, nMarkersTested))}
            }.toArray
          })

          // Annotate samples with covariates and phenotype
          writeTextFile(sampleAnnotations, sc.hadoopConfiguration) { w =>
            val sb = new StringBuilder()
            sb.append("Sample\tPhenotype")
            for (j <- 0 until nCovar) {
              sb.append(s"\tC$j")
            }
            sb.append("\n")
            w.write(sb.result())

            s.vds.sampleIds.zipWithIndex.foreach { case (s, i) =>
              val sb = new StringBuilder()
              sb.append(s)
              sb.append("\t")
              sb.append(y(i).getOrElse("NA"))

              for (j <- 0 until nCovar) {
                sb.append("\t")
                sb.append(x(j)(i).getOrElse("NA"))
              }
              sb.append("\n")
              w.write(sb.result())
            }
          }

          val sb = new StringBuilder()
          if (!quantitative)
            sb.append("Phenotype: Boolean")
          else
            sb.append("Phenotype: Double")

          for (j <- 0 until nCovar) {
            sb.append(s",C$j: Double")
          }
          s = AnnotateSamplesTable.run(s, Array("-i", sampleAnnotations, "-r", "sa.pheno", "-t", sb.result()))

          // Annotate variants with group names
          writeTextFile(variantAnnotations, sc.hadoopConfiguration) { w =>
            w.write("Variant\tGroup\n")
            groups.foreach { case (v, i) => w.write(v.toString + "\t" + s"group_$i\n")
            }
          }
          s = AnnotateVariants.run(s, Array("table", variantAnnotations, "-r", "va.groups", "-v", "Variant"))

          // Run SKAT-O command
          writeTextFile(configSkato, sc.hadoopConfiguration) { w =>
            w.write("hail.skato.Rscript Rscript\n")
            w.write("hail.skato.script src/dist/scripts/skato.r\n")
          }

          var cmd = collection.mutable.ArrayBuffer("-k", "va.groups.Group", "--block-size", blockSize.toString,
            "-y", "sa.pheno.Phenotype", "--config", configSkato, "-o", hailOutputFile, "--seed", seed.toString,
            "--kernel", kernel, "--n-resampling", nResampling.toString, "--type-resampling", typeResampling, "--impute-method", imputeMethod,
            "--method", method, "--r-corr", rCorr.mkString(","), "--missing-cutoff", missingCutoff.toString,
            "--weights-beta", weightsBeta.mkString(","), "--estimate-maf", estimateMAF.toString
          )

          if (quantitative)
            cmd += "-q"

          if (nCovar > 0) {
            cmd += "-c"
            cmd += (0 until nCovar).map{i => s"sa.pheno.C$i"}.mkString(",")
          }

          if (noAdjustment)
            cmd += "--no-adjustment"

          if (!(rCorr.length == 1 && rCorr(0) == 0.0) && kernel != "linear" && kernel != "linear.weighted"){
            try {
              s = GroupTestSKATO.run(s, cmd.toArray)
              false
            } catch {
              case e: FatalException => true
              case _:Throwable => false
            }
          } else {

            s = GroupTestSKATO.run(s, cmd.toArray)

            val hailResults = sc.parallelize(readLines(hailOutputFile, sc.hadoopConfiguration) { lines =>
              lines.drop(1)
              lines.map { l => l.transform { line =>
                val Array(groupName, pValue, pValueResampling, nMarkers, nMarkersTested) = line.value.split( """\s+""")
                (groupName, (pValue, nMarkers, nMarkersTested))
              }
              }.toArray
            })

            val result = plinkResults.fullOuterJoin(hailResults).map { case (g, (p, h)) =>

              val p2 = p match {
                case None => (Double.NaN, 0, 0)
                case Some(res) => (if (res._1 == "NA") -9.0 else res._1.toDouble, if (res._2 == "NA") 0 else res._2.toInt, if (res._3 == "NA") 0 else res._3.toInt)
              }

              val h2 = h match {
                case None => (Double.NaN, 0, 0)
                case Some(res) => (if (res._1 == "null") -9.0 else res._1.toDouble, if (res._2 == "null") 0 else res._2.toInt, if (res._3 == "null") 0 else res._3.toInt)
              }

              if (p2 == h2)
                true
              else if ((math.abs(p2._1 - h2._1) < 1e-3 || p2._1 == h2._1) && p2._2 == h2._2 && p2._3 == h2._3)
                true
              else {
                println(s"group: $g")
                println(s"PValue: plink=${p2._1} hail=${h2._1} ${p2._1 == h2._1}")
                println(s"nMarkers: plink=${p2._2} hail=${h2._2} ${p2._2 == h2._2}")
                println(s"nMarkersTested: plink=${p2._3} hail=${h2._3} ${p2._3 == h2._3}")
                false
              }
            }.fold(true)(_ & _)
            result
          }
        }
      }
  }

  @Test def testSKATO() {
    Spec.check()
  }
}
