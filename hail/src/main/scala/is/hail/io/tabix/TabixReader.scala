package is.hail.io.tabix

import java.io.InputStream

import org.apache.hadoop.io.compress.SplitCompressionInputStream

object TabixReader {
  val MaxBin: Int = 37450;
  val TadLidxShift: Int = 14;
  val DefaultBufferSize: Int = 1000;
}

class TabixReader(
  private val filePath: String,
  private val indexPath: String,
  private var stream: InputStream
) {
}
