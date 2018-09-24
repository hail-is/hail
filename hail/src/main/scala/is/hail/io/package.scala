package is.hail

import is.hail.expr.types.Type
import is.hail.utils._
import org.apache.hadoop.conf.Configuration

package object io {
  type VCFFieldAttributes = Map[String, String]
  type VCFAttributes = Map[String, VCFFieldAttributes]
  type VCFMetadata = Map[String, VCFAttributes]

  def exportTypes(filename: String, hConf: Configuration, info: Array[(String, Type)]) {
    val sb = new StringBuilder
    hConf.writeTextFile(filename) { out =>
      info.foreachBetween { case (name, t) =>
        sb.append(prettyIdentifier(name))
        sb.append(":")
        t.pretty(sb, 0, compact = true)
      } { sb += ',' }

      out.write(sb.result())
    }
  }
}
