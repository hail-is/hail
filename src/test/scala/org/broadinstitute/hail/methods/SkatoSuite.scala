package org.broadinstitute.hail.methods

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check._
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.{FatalException, SparkSuite}
import org.testng.annotations.Test
import org.broadinstitute.hail.check.Arbitrary._

import scala.collection.mutable
import scala.language._
import scala.math.Numeric.Implicits._
import scala.sys.process._

class SkatoSuite extends SparkSuite {

  val testVepSignature = TStruct("transcript_consequences" -> TArray(TStruct("lof_flag" -> TBoolean, "gene_id" -> TString)))
  val sampleSignature = TStruct("phenotype" -> TBoolean)

  def genAnnotation(genes: Array[String]) = for (flag <- arbitrary[Boolean]; gene <- Gen.oneOfSeq(genes)) yield Annotation(flag, gene)

  def genAnnotations(n: Int, genes: Array[String]) = Gen.buildableOfN[IndexedSeq, Annotation](n, genAnnotation(genes))

  def genVariantAnnotation(genes: Array[String]) = for (n <- Gen.choose(0, 10); va <- genAnnotations(n, genes)) yield Annotation(va)

  def dichotomousPhenotype(nSamples: Int) = Gen.buildableOfN[Array, Option[Boolean]](nSamples, Gen.option(arbitrary[Boolean], 0.95))

  def quantitativePhenotype(nSamples: Int) = Gen.buildableOfN[Array, Option[Double]](nSamples, Gen.option(Gen.choose(-0.5, 0.5), 0.95))

  def covariateMatrix(nSamples: Int, numCovar: Int) = Gen.buildableOfN[Array, Array[Option[Double]]](numCovar, quantitativePhenotype(nSamples))

  def readResults(file: String) = {
    readLines(file, sc.hadoopConfiguration) { lines =>
      lines.drop(1).map {
        _.map { line =>
          val Array(groupName, pValue, pValueNoAdj, nMarkers, nMarkersTested) = line.split( """\t""")
          groupName
        }
      }.toArray
    }
  }

  object Spec extends Properties("SKAT-O") {
    val compGen = for (vds: VariantDataset <- VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random);
      blockSize: Int <- Gen.choose(1, 1000);
      quantitative: Boolean <- arbitrary[Boolean];
      nCovar: Int <- Gen.choose(0, 5);
      nGroups: Int <- Gen.choose(1, 50);
      y <- if (quantitative) quantitativePhenotype(vds.nSamples) else dichotomousPhenotype(vds.nSamples);
      x <- covariateMatrix(vds.nSamples, nCovar);
      seed <- Gen.posInt;
      nRho <- Gen.choose(1, 10);
      rCorr <- Gen.frequency(
        (3, Gen.buildableOfN[Array, Double](1, Gen.const[Double](0.0))),
        (3, Gen.buildableOfN[Array, Double](1, Gen.const[Double](1.0))),
        (3, Gen.distinctBuildableOfN[Array, Double](nRho, Gen.choose(0.0, 1.0))));
      imputeMethod <- Gen.oneOf("fixed", "bestguess"); //cannot test random because order of groups not identical between PLINK and Hail
      kernel <- Gen.frequency((5, Gen.const("linear")), (5, Gen.const("linear.weighted")), (1, Gen.oneOf("IBS", "IBS.weighted", "quadratic", "2wayIX")));
      nResampling <- Gen.frequency((5, Gen.const(0)), (5, Gen.choose(0, 100)));
      typeResampling <- if (quantitative) Gen.oneOf("bootstrap", "permutation") else Gen.oneOf("bootstrap", "bootstrap.fast", "permutation");
      noAdjustment <- arbitrary[Boolean];
      method <- Gen.oneOf("davies", "liu", "liu.mod", "optimal.adj", "optimal");
      missingCutoff <- Gen.choose(0.1, 0.6);
      weightsBeta <- Gen.buildableOfN[Array, Int](2, Gen.choose(1, 25));
      estimateMAF <- Gen.oneOf(1, 2)

    ) yield (vds, blockSize, quantitative, nCovar, nGroups,
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
        val setID = fileRoot + ".setid"
        val covarFile = fileRoot + ".covar"

        var s = State(sc, sqlContext, vds.filterVariants((v, va, gs) => v.toString.length < 50 && divOption(gs.flatMap(_.nNonRefAlleles).sum, gs.count(_.isCalled) * 2) != Option(0.5))) //hack because SKAT-O won't accept snp ids with > 50 characters
        s = SplitMulti.run(s, Array.empty[String])
        s = VariantQC.run(s, Array.empty[String])
        s = FilterVariantsExpr.run(s, Array("--keep", "-c", "va.qc.AF < 0.4")) //problem with MAF flip if too close to 0.5
        s = s.copy(vds = s.vds.filterVariants((v, va, gs) => v.toString.length < 50)) //hack because SKAT-O won't accept snp ids with > 50 characters
        s = ExportPlink.run(s, Array("-o", plinkRoot))

        val nNonMissing =
          if (x.isEmpty)
            y.count(_.isDefined)
          else
            y.zip(x.transpose[Option[Double]]).count { case (yi, xi) => yi.isDefined && (xi.isEmpty || xi.forall(_.isDefined)) }

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
            groups.foreach { case (v, i) => w.write(s"group_$i" + "\t" + v.toString + "\n") }
          }

          // Write covariate file
          writeTextFile(covarFile, sc.hadoopConfiguration) { w =>
            s.vds.sampleIds.zipWithIndex.foreach { case (sid, i) =>
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
            s.vds.sampleIds.zipWithIndex.foreach { case (sid, i) =>
              val phenoString = y(i).map(_.toString) match {
                case None => -9
                case Some("true") => 2
                case Some("false") => 1
                case Some(p) => p
              }
              w.write(s"0\t$sid\t0\t0\t0\t$phenoString\n")
            }
          }

          // Run SKAT-O from PLINK file
          val adjustString = if (noAdjustment) "--no-adjustment" else ""
          val yType = if (quantitative) "C" else "D"

          val plinkCommand =
            s"""Rscript src/test/resources/testSkatoPlink.r
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
                |--r-corr ${ rCorr.mkString(",") }
                |--missing-cutoff $missingCutoff
                |--weights-beta ${ weightsBeta.mkString(",") }
                |--estimate-maf $estimateMAF
               """.stripMargin

          val plinkSkatExitCode = plinkCommand !

          val plinkResults = sc.parallelize(readLines(resultsFile, sc.hadoopConfiguration) { lines =>
            lines.map {
              _.map { line =>
                val Array(groupName, pValue, nMarkers, nMarkersTested) = line.split( """\s+""")
                (groupName, (pValue, nMarkers, nMarkersTested))
              }.value
            }.toIndexedSeq
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

            s.vds.sampleIds.zipWithIndex.foreach { case (sid, i) =>
              val sb = new StringBuilder()
              sb.append(sid)
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
          s = AnnotateSamplesTable.run(s, Array("-i", sampleAnnotations, "-e", "Sample", "-r", "sa.pheno", "-t", sb.result()))

          // Annotate variants with group names
          writeTextFile(variantAnnotations, sc.hadoopConfiguration) { w =>
            w.write("Variant\tGroup\n")
            groups.foreach { case (v, i) => w.write(v.toString + "\t" + s"group_$i\n")
            }
          }
          s = AnnotateVariants.run(s, Array("table", variantAnnotations, "-r", "va.groups", "-e", "Variant", "-t", "Variant: Variant"))

          var cmd = collection.mutable.ArrayBuffer("-k", "va.groups.Group", "--block-size", blockSize.toString,
            "-y", "sa.pheno.Phenotype", "-o", hailOutputFile, "--seed", seed.toString,
            "--kernel", kernel, "--n-resampling", nResampling.toString, "--type-resampling", typeResampling, "--impute-method", imputeMethod,
            "--method", method, "--r-corr", rCorr.mkString(","), "--missing-cutoff", missingCutoff.toString,
            "--weights-beta", weightsBeta.mkString(","), "--estimate-maf", estimateMAF.toString
          )

          if (quantitative)
            cmd += "-q"

          if (nCovar > 0) {
            cmd += "-c"
            cmd += (0 until nCovar).map { i => s"sa.pheno.C$i" }.mkString(",")
          }

          if (noAdjustment)
            cmd += "--no-adjustment"

          if (!(rCorr.length == 1 && rCorr(0) == 0.0) && kernel != "linear" && kernel != "linear.weighted") {
            try {
              s = GroupTestSKATO.run(s, cmd.toArray)
              false
            } catch {
              case e: FatalException => true
              case _: Throwable => false
            }
          } else {

            s = GroupTestSKATO.run(s, cmd.toArray)

            val hailResults = sc.parallelize(readLines(hailOutputFile, sc.hadoopConfiguration) { lines =>
              lines.drop(1)
              lines.map {
                _.map { line =>
                  val Array(groupName, pValue, pValueResampling, nMarkers, nMarkersTested) = line.split( """\s+""")
                  (groupName, (pValue, nMarkers, nMarkersTested))
                }.value
              }.toIndexedSeq
            })

            plinkResults.fullOuterJoin(hailResults).forall { case (g, (p, h)) =>

              val p2 = p match {
                case None => (Double.NaN, 0, 0)
                case Some(res) => (if (res._1 == "NA") -9.0 else res._1.toDouble, if (res._2 == "NA") 0 else res._2.toInt, if (res._3 == "NA") 0 else res._3.toInt)
              }

              val h2 = h match {
                case None => (Double.NaN, 0, 0)
                case Some(res) => (if (res._1 == "NA") -9.0 else res._1.toDouble, if (res._2 == "NA") 0 else res._2.toInt, if (res._3 == "NA") 0 else res._3.toInt)
              }

              if (p2 == h2)
                true
              else if ((math.abs(p2._1 - h2._1) < 1e-3 || p2._1 == h2._1) && p2._2 == h2._2 && p2._3 == h2._3)
                true
              else {
                println(s"group: $g")
                println(s"PValue: plink=${ p2._1 } hail=${ h2._1 } ${ p2._1 == h2._1 }")
                println(s"nMarkers: plink=${ p2._2 } hail=${ h2._2 } ${ p2._2 == h2._2 }")
                println(s"nMarkersTested: plink=${ p2._3 } hail=${ h2._3 } ${ p2._3 == h2._3 }")
                false
              }
            }
          }
        }
      }


    def compositeAnnotationGen = for (vds: VariantDataset <- VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random);
      nGroups <- Gen.choose(1, 10);
      genes <- Gen.buildableOfN[Array, String](nGroups, arbitrary[String]);
      phenotypes <- dichotomousPhenotype(vds.nSamples);
      annotations <- Gen.buildableOfN[Array, Annotation](vds.nVariants.toInt, genVariantAnnotation(genes));
      annotType <- Gen.oneOf("Set", "Array")
    ) yield (vds, annotations, genes, phenotypes, annotType)

    property("handle splat") =
      forAll(compositeAnnotationGen) { case (vds, annotations, genes, phenotypes, annotType) =>
        val tmpOutput = tmpDir.createTempFile("splatTest")

        val variantAnnotations = sc.parallelize(vds.variants.collect().zip(annotations))
        val phenotypeMap = vds.sampleIds.zip(phenotypes.map { p => Annotation(p.orNull) }).toMap

        val groupKeyQueryString =
          if (annotType == "Set")
            s"let x = va.vep.transcript_consequences.map(csq => csq.gene_id).toSet in if (x.size == 0) NA: Set[String] else x"
          else
            s"let x = va.vep.transcript_consequences.map(csq => csq.gene_id) in if (x.length == 0) NA: Array[String] else x"

        val splatResults = tmpOutput + ".splat.tsv"
        val noSplatResults = tmpOutput + ".nosplat.tsv"

        val (finalType, inserter): (Type, (Annotation, Option[Annotation]) => Annotation) =
          vds.insertVA(testVepSignature, List("vep"))

        var s = State(sc, sqlContext, vds
          .annotateVariants(variantAnnotations, finalType, inserter)
          .annotateSamples(phenotypeMap, sampleSignature, List("pheno")))
        s = SplitMulti.run(s, Array.empty[String])

        val nNonMissingPheno = phenotypes.count(_.isDefined)
        val nVariants = s.vds.nVariants


        val cmdNoSplat = mutable.ArrayBuffer(
          "-k", groupKeyQueryString,
          "-y", "sa.pheno.phenotype"
        )
        val cmdSplat = mutable.ArrayBuffer(
          "-k", groupKeyQueryString,
          "-y", "sa.pheno.phenotype"
        )

        cmdNoSplat += "-o"
        cmdNoSplat += noSplatResults

        cmdSplat += "-o"
        cmdSplat += splatResults
        cmdSplat += "--splat"

        if (nNonMissingPheno == 0) {
          try {
            s = GroupTestSKATO.run(s, cmdNoSplat.result().toArray)
            s = GroupTestSKATO.run(s, cmdSplat.result().toArray)
            false
          } catch {
            case e: FatalException => true
            case _: Throwable => false
          }
        } else {
          s = GroupTestSKATO.run(s, cmdNoSplat.result().toArray)
          s = GroupTestSKATO.run(s, cmdSplat.result().toArray)

          val (baseType, querier) = s.vds.queryVA(groupKeyQueryString)

          val (nSplatGroupsAnswer, nNoSplatGroupsAnswer) = baseType match {
            case _: TSet =>
              val trueGroups = s.vds.rdd.flatMap { case (v, (va, gs)) => querier(va) }.map(_.asInstanceOf[Set[String]]).collect()
              val nSplatGroupsAnswer = trueGroups.foldLeft(Set.empty[String])((comb, s) => comb.union(s)).size
              val nNoSplatGroupsAnswer = trueGroups.toSet.size
              (nSplatGroupsAnswer, nNoSplatGroupsAnswer)
            case _: TArray =>
              val trueGroups = s.vds.rdd.flatMap { case (v, (va, gs)) => querier(va) }.map(_.asInstanceOf[IndexedSeq[String]]).collect()
              val nSplatGroupsAnswer = trueGroups.foldLeft(Set.empty[String])((comb, sid) => comb.union(sid.toSet)).size
              val nNoSplatGroupsAnswer = trueGroups.toSet.size
              (nSplatGroupsAnswer, nNoSplatGroupsAnswer)
            case _ => fatal(s"Can't test type $baseType for splat")
          }

          val nSplatGroupsOutput = readResults(splatResults).length
          val nNoSplatGroupOutput = readResults(noSplatResults).length

          nSplatGroupsAnswer == nSplatGroupsOutput && nNoSplatGroupsAnswer == nNoSplatGroupsAnswer && nSplatGroupsAnswer <= genes.length
        }
      }
  }

  @Test def testSKATO() {
    Spec.check()
  }
}
