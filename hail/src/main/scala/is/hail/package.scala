package is

import java.io.InputStream

package object hail {

  private object HailBuildInfo {

    import is.hail.utils._

    import java.util.Properties

    val (
      hail_revision: String,
      hail_spark_version: String,
      hail_pip_version: String,
    ) =

      loadFromResource[(String, String, String)]("build-info.properties") {
        (is: InputStream) =>
          val unknownProp = "<unknown>"
          val props = new Properties()
          props.load(is)
          (
            props.getProperty("revision", unknownProp),
            props.getProperty("sparkVersion", unknownProp),
            props.getProperty("hailPipVersion", unknownProp),
          )
      }
  }

  val HAIL_REVISION = HailBuildInfo.hail_revision
  val HAIL_SPARK_VERSION = HailBuildInfo.hail_spark_version
  val HAIL_PIP_VERSION = HailBuildInfo.hail_pip_version

  // FIXME: probably should use tags or something to choose English name
  val HAIL_PRETTY_VERSION = HAIL_PIP_VERSION + "-" + HAIL_REVISION.substring(0, 12)

}
