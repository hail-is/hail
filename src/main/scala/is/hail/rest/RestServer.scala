package is.hail.rest

import is.hail.variant.VariantDataset
import org.http4s.server.blaze.BlazeBuilder

object RestServerLinreg {
  def apply(vds: VariantDataset, covariates: Array[String], useDosages: Boolean,
    port: Int, maxWidth: Int, hardLimit: Int) {

    val restService = new RestServiceLinreg(vds, covariates, useDosages, maxWidth, hardLimit)
    
    val task = BlazeBuilder.bindHttp(port, "0.0.0.0")
      .mountService(restService.service, "/")
      .run
    task.awaitShutdown()
  }
}

object RestServerScoreCovariance {
  def apply(vds: VariantDataset, covariates: Array[String], useDosages: Boolean,
    port: Int, maxWidth: Int, hardLimit: Int) {

    val restService = new RestServiceScoreCovariance(vds, covariates, useDosages, maxWidth, hardLimit) // Currently useDosages only affects variants as covariates
    
    val task = BlazeBuilder.bindHttp(port, "0.0.0.0")
      .mountService(restService.service, "/")
      .run
    task.awaitShutdown()
  }
}
