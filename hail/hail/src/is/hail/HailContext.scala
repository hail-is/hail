package is.hail

import is.hail.backend.spark.SparkBackend
import is.hail.io.fs.FS
import is.hail.utils._

object HailContext {

  private var theContext: HailContext = _

  def checkJavaVersion(): Unit = {
    val javaVersion = raw"(\d+)\.(\d+)\.(\d+).*".r
    val versionString = System.getProperty("java.version")
    versionString match {
      // old-style version: 1.MAJOR.MINOR (before JRE 9)
      // new-style version: MAJOR.MINOR.SECURITY (starting with JRE 9)
      /* see:
       * https://docs.oracle.com/javase/9/migrate/toc.htm#JSMIG-GUID-3A71ECEF-5FC5-46FE-9BA9-88CBFCE828CB */
      case javaVersion("1", _, _) =>
        warn(s"Hail is tested against Java 11, found Java $versionString")
      case javaVersion(major, _, _) =>
        if (major.toInt != 11)
          warn(s"Hail is tested against Java 11, found $versionString")
      case _ =>
        fatal(s"Unknown JVM version string: $versionString")
    }
  }

  def getOrCreate: HailContext =
    synchronized {
      if (theContext != null) theContext
      else apply
    }

  def apply: HailContext = synchronized {
    require(theContext == null)
    checkJavaVersion()

    theContext = new HailContext

    info(s"Running Hail version $HAIL_PRETTY_VERSION")

    theContext
  }

  def stop(): Unit =
    synchronized {
      theContext = null
    }
}

class HailContext {
  def stop(): Unit = HailContext.stop()

  private[this] def fileAndLineCounts(
    fs: FS,
    regex: String,
    files: Seq[String],
    maxLines: Int,
  ): Map[String, Array[WithContext[String]]] = {
    val regexp = regex.r
    SparkBackend.sparkContext.textFilesLines(fs.globAll(files).map(_.getPath))
      .filter(line => regexp.findFirstIn(line.value).isDefined)
      .take(maxLines)
      .groupBy(_.source.file)
  }

  def grepPrint(fs: FS, regex: String, files: Seq[String], maxLines: Int): Unit = {
    fileAndLineCounts(fs, regex, files, maxLines).foreach { case (file, lines) =>
      info(s"$file: ${lines.length} ${plural(lines.length, "match", "matches")}:")
      lines.map(_.value).foreach { line =>
        val (screen, logged) = line.truncatable().strings
        log.info("\t" + logged)
        println(s"\t$screen")
      }
    }
  }

  def grepReturn(fs: FS, regex: String, files: Seq[String], maxLines: Int)
    : Array[(String, Array[String])] =
    fileAndLineCounts(fs: FS, regex, files, maxLines).mapValues(_.map(_.value)).toArray

}
