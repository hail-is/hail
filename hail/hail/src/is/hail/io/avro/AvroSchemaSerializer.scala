package is.hail.io.avro

import org.apache.avro.Schema
import org.json4s.CustomSerializer
import org.json4s.jackson.{compactJson, parseJson}

class AvroSchemaSerializer extends CustomSerializer[Schema](_ =>
      (
        { case jv =>
          new Schema.Parser().parse(compactJson(jv))
        },
        { case schema: Schema =>
          parseJson(schema.toString)
        },
      )
    )
