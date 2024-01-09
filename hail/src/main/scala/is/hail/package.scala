package is

import java.io.InputStream

package object hail {

  private object HailBuildInfo {

    import is.hail.utils._

    import java.util.Properties

    val (
      hail_build_user: String,
      hail_revision: String,
      hail_branch: String,
      hail_build_date: String,
      hail_spark_version: String,
      hail_pip_version: String,
    ) = {

      loadFromResource[(String, String, String, String, String, String)]("build-info.properties") {
        (is: InputStream) =>
          val unknownProp = "<unknown>"
          val props = new Properties()
          props.load(is)
          (
            props.getProperty("user", unknownProp),
            props.getProperty("revision", unknownProp),
            props.getProperty("branch", unknownProp),
            props.getProperty("date", unknownProp),
            props.getProperty("sparkVersion", unknownProp),
            props.getProperty("hailPipVersion", unknownProp),
          )
      }
    }
  }

  val HAIL_BUILD_USER = HailBuildInfo.hail_build_user
  val HAIL_REVISION = HailBuildInfo.hail_revision
  val HAIL_BRANCH = HailBuildInfo.hail_branch
  val HAIL_BUILD_DATE = HailBuildInfo.hail_build_date
  val HAIL_SPARK_VERSION = HailBuildInfo.hail_spark_version
  val HAIL_PIP_VERSION = HailBuildInfo.hail_pip_version

  // FIXME: probably should use tags or something to choose English name
  val HAIL_PRETTY_VERSION = HAIL_PIP_VERSION + "-" + HAIL_REVISION.substring(0, 12)

}
