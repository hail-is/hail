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
      hail_build_configuration: BuildConfiguration,
    ) =
      loadFromResource[(String, String, String, BuildConfiguration)]("build-info.properties") {
        (is: InputStream) =>
          val unknownProp = "<unknown>"
          val props = new Properties()
          props.load(is)
          (
            props.getProperty("revision", unknownProp),
            props.getProperty("sparkVersion", unknownProp),
            props.getProperty("hailPipVersion", unknownProp), {
              val c = props.getProperty("hailBuildConfiguration", "debug")
              BuildConfiguration.parseString(c).getOrElse(
                throw new IllegalArgumentException(
                  s"Illegal 'hailBuildConfiguration' entry in 'build-info.properties': '$c'."
                )
              )
            },
          )
      }
  }

  val HAIL_REVISION = HailBuildInfo.hail_revision
  val HAIL_SPARK_VERSION = HailBuildInfo.hail_spark_version
  val HAIL_PIP_VERSION = HailBuildInfo.hail_pip_version

  // FIXME: probably should use tags or something to choose English name
  val HAIL_PRETTY_VERSION = HAIL_PIP_VERSION + "-" + HAIL_REVISION.substring(0, 12)

  val HAIL_BUILD_CONFIGURATION = HailBuildInfo.hail_build_configuration
}

sealed trait BuildConfiguration extends Product with Serializable {
  def isDebug: Boolean
}

object BuildConfiguration {
  case object Release extends BuildConfiguration {
    override def isDebug: Boolean = false
  }

  case object Debug extends BuildConfiguration {
    override def isDebug: Boolean = true
  }

  case object CI extends BuildConfiguration {
    override def isDebug: Boolean = true
  }

  def parseString(c: String): Option[BuildConfiguration] =
    c match {
      case "release" => Some(BuildConfiguration.Release)
      case "dev" => Some(BuildConfiguration.Debug)
      case "ci" => Some(BuildConfiguration.CI)
      case _ => None
    }
}
