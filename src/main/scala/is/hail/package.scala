package is

package object hail {
  private object HailBuildInfo {
    import java.util.Properties

    val (
      hail_build_user: String,
      hail_revision: String,
      hail_branch: String,
      hail_build_date: String,
      hail_repo_url: String,
      hail_spark_version: String) = {

      val resourceStream = Thread.currentThread().getContextClassLoader.
        getResourceAsStream("build-info.properties")

      try {
        if (resourceStream == null) {
          throw new RuntimeException("Could not read properties file build-info.properties")
        }
        val unknownProp = "<unknown>"
        val props = new Properties()
        props.load(resourceStream)
        (
          props.getProperty("user", unknownProp),
          props.getProperty("revision", unknownProp),
          props.getProperty("branch", unknownProp),
          props.getProperty("date", unknownProp),
          props.getProperty("url", unknownProp),
          props.getProperty("sparkVersion", unknownProp)
        )
      } catch {
        case npe: NullPointerException =>
          throw new RuntimeException("Error while locating file build-info.properties", npe)
        case e: Exception =>
          throw new RuntimeException("Error loading properties from build-info.properties", e)
      } finally {
        if (resourceStream != null) {
          try {
            resourceStream.close()
          } catch {
            case e: Exception =>
              throw new RuntimeException("Error closing hail build info resource stream", e)
          }
        }
      }
    }
  }

  val HAIL_BUILD_USER = HailBuildInfo.hail_build_user
  val HAIL_REVISION = HailBuildInfo.hail_revision
  val HAIL_BRANCH = HailBuildInfo.hail_branch
  val HAIL_BUILD_DATE = HailBuildInfo.hail_build_date
  val HAIL_REPO_URL = HailBuildInfo.hail_repo_url
  val HAIL_SPARK_VERSION = HailBuildInfo.hail_spark_version

  // FIXME: probably should use tags or something to choose English name
  val HAIL_PRETTY_VERSION = "0.1-" + HAIL_REVISION.substring(0, 7)

}
