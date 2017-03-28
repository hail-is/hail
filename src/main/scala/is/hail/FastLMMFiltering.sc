import is.hail.HailContext
import is.hail.variant.VariantDataset

val hc = HailContext()

lazy val covariates = hc.importTable("src/test/resources/fastlmmCov.txt",
  noHeader = true, impute = true).keyBy("f1")
lazy val phenotypes = hc.importTable("src/test/resources/fastlmmPheno.txt",
  noHeader = true, impute = true, separator = " ").keyBy("f1")

lazy val vdsFastLMM = hc.importPlink(bed = "src/test/resources/fastlmmTest.bed",
  bim = "src/test/resources/fastlmmTest.bim",
  fam = "src/test/resources/fastlmmTest.fam")
  .annotateSamplesTable(covariates, expr = "sa.cov=table.f2")
  .annotateSamplesTable(phenotypes, expr = "sa.pheno=table.f2")

lazy val vdsChr1: VariantDataset = vdsFastLMM.filterVariantsExpr("""v.contig == "1"""")
  .lmmreg(vdsFastLMM.filterVariantsExpr("""v.contig != "1"""").rrm(), "sa.pheno", Array("sa.cov"), runAssoc = false)

val notChr1VDSDownsampled = vdsFastLMM.filterVariantsExpr("""v.contig != "1"""").sampleVariants(0.24)

println(notChr1VDSDownsampled.countVariants())