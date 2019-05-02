package is.hail

import is.hail.expr.types.virtual.Type
import is.hail.utils._
import is.hail.io.fs.FS

package object io {
  type VCFFieldAttributes = Map[String, String]
  type VCFAttributes = Map[String, VCFFieldAttributes]
  type VCFMetadata = Map[String, VCFAttributes]

  def exportTypes(filename: String, fs: FS, info: Array[(String, Type)]) {
    val sb = new StringBuilder
    fs.writeTextFile(filename) { out =>
      info.foreachBetween { case (name, t) =>
        sb.append(prettyIdentifier(name))
        sb.append(":")
        t.pretty(sb, 0, compact = true)
      } { sb += ',' }

      out.write(sb.result())
    }
  }
}
