package is.hail

import is.hail.collection.implicits.toRichIterable
import is.hail.io.fs.{readableBytes, FS, FileListEntry}
import is.hail.types.virtual.Type
import is.hail.utils._

import java.io.OutputStreamWriter

package object io extends Logging {
  type VCFFieldAttributes = Map[String, String]
  type VCFAttributes = Map[String, VCFFieldAttributes]
  type VCFMetadata = Map[String, VCFAttributes]

  def exportTypes(filename: String, fs: FS, info: Array[(String, Type)]): Unit = {
    val sb = new StringBuilder
    using(new OutputStreamWriter(fs.create(filename))) { out =>
      info.foreachBetween { case (name, t) =>
        sb.append(prettyIdentifier(name))
        sb.append(":")
        t.pretty(sb, 0, compact = true)
      }(sb += ',')

      out.write(sb.result())
    }
  }

  def checkGzipOfGlobbedFiles(
    globPaths: Seq[String],
    fileListEntries: Array[FileListEntry],
    forceGZ: Boolean,
    gzAsBGZ: Boolean,
    maxSizeMB: Int = 128,
  ): Unit = {
    if (fileListEntries.isEmpty)
      fatal(s"arguments refer to no files: ${globPaths.toIndexedSeq}.")

    if (!gzAsBGZ) {
      fileListEntries.foreach { fileListEntry =>
        val path = fileListEntry.getPath
        if (path.endsWith(".gz"))
          checkGzippedFile(fileListEntry, forceGZ, false, maxSizeMB)
      }
    }
  }

  def checkGzippedFile(
    fileListEntry: FileListEntry,
    forceGZ: Boolean,
    gzAsBGZ: Boolean,
    maxSizeMB: Int = 128,
  ): Unit =
    if (!forceGZ && !gzAsBGZ)
      fatal(
        s"""Cannot load file '${fileListEntry.getPath}'
           |  .gz cannot be loaded in parallel. Is the file actually *block* gzipped?
           |  If the file is actually block gzipped (even though its extension is .gz),
           |  use the 'force_bgz' argument to treat all .gz file extensions as .bgz.
           |  If you are sure that you want to load a non-block-gzipped file serially
           |  on one core, use the 'force' argument.""".stripMargin
      )
    else if (!gzAsBGZ) {
      val fileSize = fileListEntry.getLen
      if (fileSize > 1024 * 1024 * maxSizeMB)
        logger.warn(
          s"""file '${fileListEntry.getPath}' is ${readableBytes(fileSize)}
             |  It will be loaded serially (on one core) due to usage of the 'force' argument.
             |  If it is actually block-gzipped, either rename to .bgz or use the 'force_bgz'
             |  argument.""".stripMargin
        )
    }
}
