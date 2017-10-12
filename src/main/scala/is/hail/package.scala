package is

import java.io.InputStream

package object hail {

  private object HailBuildInfo {

    import java.util.Properties
    import is.hail.utils._

    val (
      hail_build_user: String,
      hail_revision: String,
      hail_branch: String,
      hail_build_date: String,
      hail_repo_url: String,
      hail_spark_version: String,
      hail_version: String) = {

      loadFromResource[(String, String, String, String, String, String, String)]("build-info.properties") {
        (is: InputStream) =>
          val unknownProp = "<unknown>"
          val props = new Properties()
          props.load(is)
          (
            props.getProperty("user", unknownProp),
            props.getProperty("revision", unknownProp),
            props.getProperty("branch", unknownProp),
            props.getProperty("date", unknownProp),
            props.getProperty("url", unknownProp),
            props.getProperty("sparkVersion", unknownProp),
            props.getProperty("hailVersion", unknownProp)
            )
      }
    }
  }

  val HAIL_BUILD_USER = HailBuildInfo.hail_build_user
  val HAIL_REVISION = HailBuildInfo.hail_revision
  val HAIL_BRANCH = HailBuildInfo.hail_branch
  val HAIL_BUILD_DATE = HailBuildInfo.hail_build_date
  val HAIL_REPO_URL = HailBuildInfo.hail_repo_url
  val HAIL_SPARK_VERSION = HailBuildInfo.hail_spark_version
  val HAIL_VERSION = HailBuildInfo.hail_version

  // FIXME: probably should use tags or something to choose English name
  val HAIL_PRETTY_VERSION = HAIL_VERSION + "-" + HAIL_REVISION.substring(0, 7)

}
