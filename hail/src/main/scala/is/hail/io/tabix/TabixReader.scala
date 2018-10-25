package is.hail.io.tabix

import is.hail.HailContext

import htsjdk.tribble.util.{ParsingUtils, TabixUtils}
import org.apache.hadoop.io.compress.SplitCompressionInputStream

import java.io.InputStream

object TabixReader {
  val MaxBin: Int = 37450;
  val TadLidxShift: Int = 14;
  val DefaultBufferSize: Int = 1000;
}

class TabixReader(val filePath: String, private val idxFilePath: Option[String]) {
  val indexPath: String = idxFilePath match {
    case None => ParsingUtils.appendToPath(filePath, TabixUtils.STANDARD_INDEX_EXTENSION)
    case Some(s) => s
  }

  private val hc = HailContext.get
  private val sc = hc.sc
  private val hConf = sc.hadoopConfiguration
}
